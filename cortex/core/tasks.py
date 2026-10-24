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


def parse_schedule(s: str) -> tuple[dict, str]:
    """Return (normalized, human). normalized is either
    {"type":"interval","seconds":N} or {"type":"daily","hour":H,"minute":M}.
    Supports 'every N min/hours', 'hourly', 'daily HH:MM' / 'at HH:MM',
    'every morning/evening/noon'. Defaults to daily 09:00."""
    s = (s or "").strip().lower()
    m = re.search(r"every\s+(\d+)\s*(min|minute|hour|hr)", s)
    if m:
        n = int(m.group(1))
        secs = n * (60 if m.group(2).startswith("min") else 3600)
        return {"type": "interval", "seconds": max(60, secs)}, f"every {n} {m.group(2)}s"
    if "hourly" in s:
        return {"type": "interval", "seconds": 3600}, "hourly"
    tm = re.search(r"(\d{1,2}):(\d{2})", s)
    hour, minute = 9, 0
    if tm:
        hour, minute = int(tm.group(1)), int(tm.group(2))
    elif "morning" in s:
        hour, minute = 8, 0
    elif "evening" in s or "night" in s:
        hour, minute = 20, 0
    elif "noon" in s or "afternoon" in s:
        hour, minute = 13, 0
    return ({"type": "daily", "hour": hour, "minute": minute},
            f"daily at {hour:02d}:{minute:02d}")


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
        sched_xml = ("<key>StartCalendarInterval</key><dict>"
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
        when = f"*/{mins} * * * *" if mins < 60 else f"0 */{max(1, mins // 60)} * * *"
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
