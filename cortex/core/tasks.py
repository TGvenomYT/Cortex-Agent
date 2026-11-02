"""
tasks.py — real recurring tasks Cortex schedules for itself, cross-platform.

When the user says "every morning at 8, email me a briefing", the agent calls
the schedule_task tool → we store the task and register it with the OS scheduler
so it runs `run_once.py task <id>` on a cadence, which executes the task's prompt
through Cortex and delivers the result (notification / email).

  macOS   -> launchd  (~/Library/LaunchAgents/com.cortex.task.<id>.plist)
  Linux   -> cron     (a `crontab` line tagged `# cortex-task-<id>`)
  Windows -> schtasks (\\Cortex\\task-<id>)

These show up in Settings → Background and can be removed there. parse_schedule()
produces a normalized schedule, and cortex/core/scheduler.py puts it in the OS scheduler.
"""
from __future__ import annotations

import os
import re
import sys

from cortex.core import scheduler
import cortex.storage.settings_store as settings_store

_DIR = __import__("cortex").ROOT  # project root: run_once shim, logs, working dir live here
_VENV_PY = sys.executable  # whichever python is running us (the venv in normal use)
_RUNNER = os.path.join(_DIR, "run_once.py")


def list_tasks() -> list[dict]:
    return settings_store.load().get("tasks", []) or []


def get_task(tid: str) -> dict | None:
    return next((t for t in list_tasks() if t["id"] == tid), None)


MIN_TASK_SECONDS = 300      # every run wakes the AI model; a 1-minute task keeps a 12 GB model cycling
_DAYS = {"sun": 0, "mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6}
_DAY_NAMES = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
_PARTS_OF_DAY = {"morning": 8, "noon": 12, "afternoon": 13, "evening": 20, "night": 21, "midnight": 0}


def _clock(s: str) -> tuple[int, int] | None:
    """The time of day mentioned in `s` as (hour, minute), or None."""
    m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", s)
    if m:
        if not 1 <= int(m.group(1)) <= 12:
            raise ValueError(f"'{m.group(0)}' isn't a real time of day")
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        hour += 12 if m.group(3) == "pm" else 0
    else:
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", s)
        if m:
            hour, minute = int(m.group(1)), int(m.group(2))
        else:
            for word, h in _PARTS_OF_DAY.items():
                if re.search(rf"\b{word}\b", s):
                    return h, 0
            return None
    if hour > 23 or minute > 59:
        raise ValueError(f"'{m.group(0)}' isn't a real time of day")
    return hour, minute


def parse_schedule(s: str) -> tuple[dict, str]:
    """Return (normalized, human). normalized is one of
    {"type":"interval","seconds":N}, {"type":"daily","hour":H,"minute":M} or
    {"type":"weekly","weekday":W,"hour":H,"minute":M} (W: 0 = Sunday).
    Raises ValueError for anything it can't read: guessing a schedule the user didn't
    ask for is worse than saying so."""
    raw = s
    s = (s or "").strip().lower()
    if re.search(r"\b(in \d+|tomorrow|tonight|once|next)\b", s):
        raise ValueError(f"'{raw}' sounds like a one-off, which is a reminder rather than a recurring task")
    m = re.search(r"every\s+(\d+)\s*(sec|second|min|minute|hour|hr|day)", s)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        secs = n * {"se": 1, "mi": 60, "ho": 3600, "hr": 3600, "da": 86400}[unit[:2]]
        if secs <= MIN_TASK_SECONDS:
            return {"type": "interval", "seconds": MIN_TASK_SECONDS}, "every 5 minutes (the shortest allowed)"
        return {"type": "interval", "seconds": secs}, f"every {n} {unit}s".replace("mins", "minutes").replace("hrs", "hours")
    if "hourly" in s or re.search(r"\bevery hour\b", s):
        return {"type": "interval", "seconds": 3600}, "hourly"
    clock = _clock(s)
    day = re.search(r"\b(sun|mon|tue|wed|thu|fri|sat)[a-z]*\b", s)
    if day or "weekly" in s or re.search(r"\bevery week\b", s):
        hour, minute = clock or (9, 0)
        weekday = _DAYS[day.group(1)] if day else 1
        return ({"type": "weekly", "weekday": weekday, "hour": hour, "minute": minute},
                f"every {_DAY_NAMES[weekday]} at {hour:02d}:{minute:02d}")
    if clock:
        return ({"type": "daily", "hour": clock[0], "minute": clock[1]},
                f"daily at {clock[0]:02d}:{clock[1]:02d}")
    if re.search(r"\b(daily|every day)\b", s):
        return {"type": "daily", "hour": 9, "minute": 0}, "daily at 09:00"
    raise ValueError(f"I can't read the schedule '{raw}'. Try 'daily 08:00', 'every 30 min' or 'every friday at 9am'")


def _install(task: dict):
    scheduler.install(f"task.{task['id']}", [_VENV_PY, _RUNNER, "task", task["id"]],
                      parse_schedule(task["schedule"])[0])


def _uninstall(tid: str):
    scheduler.remove(f"task.{tid}")


def add_task(desc: str, prompt: str, schedule: str, deliver: str = "notify") -> dict:
    # A plain number, because the model has to copy this. It garbles hex ids, and Ollama
    # silently drops a tool call holding one like 9764941c, so nothing happens at all.
    tid = str(max((int(t["id"]) for t in list_tasks() if str(t["id"]).isdigit()), default=0) + 1)
    _, human = parse_schedule(schedule)
    task = {"id": tid, "desc": desc or prompt[:50], "prompt": prompt,
            "schedule": schedule, "human": human, "deliver": deliver}
    # Register with the OS scheduler first; only persist if that succeeds, so the
    # Jobs tab never lists a task that isn't actually scheduled anywhere.
    _install(task)
    settings_store.save({"tasks": list_tasks() + [task]})
    return task


def remove_task(tid: str) -> bool:
    tasks = list_tasks()
    kept = [t for t in tasks if t["id"] != tid]
    settings_store.save({"tasks": kept})
    _uninstall(tid)
    return len(kept) < len(tasks)


def _update(tid: str, **fields) -> None:
    tasks = list_tasks()
    for t in tasks:
        if t["id"] == tid:
            t.update(fields)
    settings_store.save({"tasks": tasks})


def record_failure(tid: str) -> int:
    """One more run in a row that couldn't reach a model. -> how many so far."""
    n = int((get_task(tid) or {}).get("failures", 0)) + 1
    _update(tid, failures=n)
    return n


def clear_failures(tid: str) -> None:
    if (get_task(tid) or {}).get("failures"):
        _update(tid, failures=0)


def is_enabled(task: dict) -> bool:
    return task.get("enabled", True)


def set_task_enabled(tid: str, enabled: bool):
    """Pause or resume one task. Pausing removes it from the OS scheduler (so it stops
    waking the machine) but keeps it listed here; resuming re-registers it."""
    tasks = list_tasks()
    task = next((t for t in tasks if t["id"] == tid), None)
    if task is None:
        raise KeyError(f"no task {tid}")
    if enabled:
        _install(task)           # may raise — in which case we leave the flag untouched
    else:
        _uninstall(tid)
    task["enabled"] = bool(enabled)
    settings_store.save({"tasks": tasks})
