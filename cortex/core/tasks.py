"""
tasks.py — real recurring tasks Cortex schedules for itself, cross-platform.

When the user says "every morning at 8, email me a briefing", the agent calls
the schedule_task tool → we store the task and register it with the OS scheduler
so it runs `run_once.py task <id>` on a cadence, which executes the task's prompt
through Cortex and delivers the result (notification / email).

  macOS   -> launchd  (~/Library/LaunchAgents/com.cortex.task.<id>.plist)
  Linux   -> cron     (a `crontab` line tagged `# cortex-task-<id>`)
  Windows -> schtasks (\\Cortex\\task-<id>)

These show up in Settings → Jobs and can be removed there. parse_schedule()
produces a normalized schedule which each backend translates to its own format.
"""
from __future__ import annotations

import os
import re
import sys
import uuid
import shutil
import subprocess

import cortex.util.osutil as osutil
import cortex.storage.settings_store as settings_store

_DIR = __import__("cortex").ROOT  # project root: run_once shim, logs, working dir live here
_VENV_PY = sys.executable  # whichever python is running us (the venv in normal use)
_RUNNER = os.path.join(_DIR, "run_once.py")
_LA = os.path.expanduser("~/Library/LaunchAgents")


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


# ── macOS: launchd ──────────────────────────────────────────────────────────
def _plist_path(tid: str) -> str:
    return os.path.join(_LA, f"com.cortex.task.{tid}.plist")


def _mac_install(task: dict):
    os.makedirs(_LA, exist_ok=True)
    os.makedirs(os.path.join(_DIR, "logs"), exist_ok=True)
    sched, _ = parse_schedule(task["schedule"])
    if sched["type"] == "interval":
        sched_xml = f"<key>StartInterval</key><integer>{sched['seconds']}</integer>"
    else:
        weekday = f"<key>Weekday</key><integer>{sched['weekday']}</integer>" if sched["type"] == "weekly" else ""
        sched_xml = (f"<key>StartCalendarInterval</key><dict>{weekday}"
                     f"<key>Hour</key><integer>{sched['hour']}</integer>"
                     f"<key>Minute</key><integer>{sched['minute']}</integer></dict>")
    label = f"com.cortex.task.{task['id']}"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>Label</key><string>{label}</string>
 <key>ProgramArguments</key><array>
  <string>{_VENV_PY}</string><string>{_RUNNER}</string><string>task</string><string>{task['id']}</string>
 </array>
 <key>WorkingDirectory</key><string>{_DIR}</string>
 {sched_xml}
 <key>ProcessType</key><string>Background</string><key>LowPriorityIO</key><true/>
 <key>StandardOutPath</key><string>{_DIR}/logs/task-{task['id']}.log</string>
 <key>StandardErrorPath</key><string>{_DIR}/logs/task-{task['id']}-error.log</string>
</dict></plist>"""
    p = _plist_path(task["id"])
    with open(p, "w", encoding="utf-8") as f:
        f.write(plist)
    subprocess.run(["launchctl", "unload", p], capture_output=True)
    subprocess.run(["launchctl", "load", p], capture_output=True)


def _mac_remove(tid: str):
    p = _plist_path(tid)
    subprocess.run(["launchctl", "unload", p], capture_output=True)
    try:
        os.remove(p)
    except OSError:
        pass


# ── Linux: cron ──────────────────────────────────────────────────────────────
def _cron_lines() -> list[str]:
    out = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return out.stdout.splitlines() if out.returncode == 0 else []


def _cron_write(lines: list[str]):
    proc = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True,
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "crontab write failed")


def _linux_install(task: dict):
    sched, _ = parse_schedule(task["schedule"])
    if sched["type"] == "interval":
        mins = max(1, sched["seconds"] // 60)
        hours = mins // 60
        when = (f"*/{mins} * * * *" if mins < 60 else f"0 */{hours} * * *" if hours < 24
                else f"0 0 */{hours // 24} * *")
    elif sched["type"] == "weekly":
        when = f"{sched['minute']} {sched['hour']} * * {sched['weekday']}"
    else:
        when = f"{sched['minute']} {sched['hour']} * * *"
    logdir = os.path.join(_DIR, "logs")
    os.makedirs(logdir, exist_ok=True)
    cmd = (f'cd "{_DIR}" && "{_VENV_PY}" "{_RUNNER}" task {task["id"]} '
           f'>> "{logdir}/task-{task["id"]}.log" 2>&1')
    tag = f"# cortex-task-{task['id']}"
    lines = [l for l in _cron_lines() if f"cortex-task-{task['id']}" not in l]
    lines += [tag, f"{when} {cmd}"]
    _cron_write(lines)


def _linux_remove(tid: str):
    lines = _cron_lines()
    kept, skip_next = [], False
    for l in lines:
        if f"cortex-task-{tid}" in l:
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            continue
        kept.append(l)
    try:
        _cron_write(kept)
    except RuntimeError:
        pass


# ── Windows: schtasks ────────────────────────────────────────────────────────
def _win_name(tid: str) -> str:
    return f"\\Cortex\\task-{tid}"


def _win_install(task: dict):
    sched, _ = parse_schedule(task["schedule"])
    tr = f'"{_VENV_PY}" "{_RUNNER}" task {task["id"]}'
    base = ["schtasks", "/create", "/f", "/tn", _win_name(task["id"]), "/tr", tr]
    if sched["type"] == "interval":
        base += ["/sc", "MINUTE", "/mo", str(max(1, sched["seconds"] // 60))]
    elif sched["type"] == "weekly":
        base += ["/sc", "WEEKLY", "/d", _DAY_NAMES[sched["weekday"]][:3].upper(),
                 "/st", f"{sched['hour']:02d}:{sched['minute']:02d}"]
    else:
        base += ["/sc", "DAILY", "/st", f"{sched['hour']:02d}:{sched['minute']:02d}"]
    proc = subprocess.run(base, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "schtasks create failed")


def _win_remove(tid: str):
    subprocess.run(["schtasks", "/delete", "/f", "/tn", _win_name(tid)], capture_output=True)


# ── public API (OS dispatch) ─────────────────────────────────────────────────
def _install(task: dict):
    if osutil.IS_MAC:
        _mac_install(task)
    elif osutil.IS_WINDOWS:
        _win_install(task)
    elif osutil.IS_LINUX and shutil.which("crontab"):
        _linux_install(task)
    else:
        raise RuntimeError(f"no supported scheduler on {osutil.OS_NAME} (need cron/launchd/schtasks)")


def _uninstall(tid: str):
    if osutil.IS_MAC:
        _mac_remove(tid)
    elif osutil.IS_WINDOWS:
        _win_remove(tid)
    elif osutil.IS_LINUX:
        _linux_remove(tid)


def add_task(desc: str, prompt: str, schedule: str, deliver: str = "notify") -> dict:
    tid = uuid.uuid4().hex[:8]
    _, human = parse_schedule(schedule)
    task = {"id": tid, "desc": desc or prompt[:50], "prompt": prompt,
            "schedule": schedule, "human": human, "deliver": deliver}
    # Register with the OS scheduler first; only persist if that succeeds, so the
    # Jobs tab never lists a task that isn't actually scheduled anywhere.
    _install(task)
    settings_store.save({"tasks": list_tasks() + [task]})
    return task


def remove_task(tid: str):
    settings_store.save({"tasks": [t for t in list_tasks() if t["id"] != tid]})
    _uninstall(tid)


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
