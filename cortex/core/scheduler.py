"""
scheduler.py — put a job in the operating system's own scheduler.

    macOS    launchd   ~/Library/LaunchAgents/com.cortex.<name>.plist
    Linux    cron      a crontab line tagged  # cortex-<name>
    Windows  schtasks  \\Cortex\\<name>

A schedule is {"type":"interval","seconds":N}, {"type":"daily","hour":H,"minute":M} or
{"type":"weekly","weekday":W,"hour":H,"minute":M} (W: 0 = Sunday). Each job runs one
command and exits, so nothing stays in the background between runs.
"""
from __future__ import annotations

import os
import plistlib
import re
import shlex
import shutil
import subprocess

import cortex
import cortex.util.osutil as osutil

_LA = "~/Library/LaunchAgents"
_DAYS = ("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT")
PATH = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


def _log(name: str) -> str:
    logs = os.path.expanduser("~/.cortex/logs")
    os.makedirs(logs, exist_ok=True)
    return os.path.join(logs, f"{_slug(name)}.log")


# ── macOS ─────────────────────────────────────────────────────────────────
def plist_path(name: str) -> str:
    return os.path.join(os.path.expanduser(_LA), f"com.cortex.{name}.plist")


def write_plist(name: str, argv: list, schedule: dict) -> str:
    """Write the launchd job without loading it. -> its path."""
    job = {"Label": f"com.cortex.{name}", "ProgramArguments": argv, "WorkingDirectory": cortex.ROOT,
           "ProcessType": "Background", "LowPriorityIO": True, "Nice": 5,
           "StandardOutPath": _log(name), "StandardErrorPath": _log(name),
           "EnvironmentVariables": {"PATH": PATH}}
    if schedule["type"] == "interval":
        job["StartInterval"] = schedule["seconds"]
    else:
        when = {"Hour": schedule["hour"], "Minute": schedule["minute"]}
        if schedule["type"] == "weekly":
            when["Weekday"] = schedule["weekday"]
        job["StartCalendarInterval"] = when
    path = plist_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        plistlib.dump(job, f)
    return path


# ── Linux ─────────────────────────────────────────────────────────────────
def _cron_lines() -> list[str]:
    out = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    return out.stdout.splitlines() if out.returncode == 0 else []


def _cron_write(lines: list[str]) -> None:
    proc = subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "crontab write failed")


def cron_when(schedule: dict) -> str:
    if schedule["type"] == "interval":
        mins = max(1, schedule["seconds"] // 60)
        hours = mins // 60
        if mins < 60:
            return f"*/{mins} * * * *"
        return f"0 */{hours} * * *" if hours < 24 else f"0 0 */{hours // 24} * *"
    if schedule["type"] == "weekly":
        return f"{schedule['minute']} {schedule['hour']} * * {schedule['weekday']}"
    return f"{schedule['minute']} {schedule['hour']} * * *"


def _cron_install(name: str, argv: list, schedule: dict) -> None:
    tag = f"# cortex-{_slug(name)}"
    cmd = " ".join(shlex.quote(a) for a in argv)
    line = f"{cron_when(schedule)} cd {shlex.quote(cortex.ROOT)} && {cmd} >> {shlex.quote(_log(name))} 2>&1"
    kept = _cron_without(name)
    _cron_write(kept + [tag, line])


def _cron_without(name: str) -> list[str]:
    tag = f"# cortex-{_slug(name)}"
    kept, skip = [], False
    for line in _cron_lines():
        if line.strip() == tag:
            skip = True
            continue
        if skip:
            skip = False
            continue
        kept.append(line)
    return kept


# ── Windows ───────────────────────────────────────────────────────────────
def _win_name(name: str) -> str:
    return f"\\Cortex\\{_slug(name)}"


def schtasks_args(name: str, argv: list, schedule: dict) -> list:
    command = subprocess.list2cmdline(argv)
    base = ["schtasks", "/create", "/f", "/tn", _win_name(name), "/tr", command]
    clock = f"{schedule.get('hour', 0):02d}:{schedule.get('minute', 0):02d}"
    if schedule["type"] == "interval":
        return base + ["/sc", "MINUTE", "/mo", str(max(1, schedule["seconds"] // 60))]
    if schedule["type"] == "weekly":
        return base + ["/sc", "WEEKLY", "/d", _DAYS[schedule["weekday"]], "/st", clock]
    return base + ["/sc", "DAILY", "/st", clock]


# ── the same three operations everywhere ──────────────────────────────────
def install(name: str, argv: list, schedule: dict) -> None:
    """Schedule a job, replacing any earlier one. Raises RuntimeError if the OS refuses."""
    if osutil.IS_MAC:
        path = write_plist(name, argv, schedule)
        subprocess.run(["launchctl", "unload", path], capture_output=True)
        subprocess.run(["launchctl", "load", path], capture_output=True)
    elif osutil.IS_WINDOWS:
        proc = subprocess.run(schtasks_args(name, argv, schedule), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "schtasks create failed")
    elif osutil.IS_LINUX and shutil.which("crontab"):
        _cron_install(name, argv, schedule)
    else:
        raise RuntimeError(f"no supported scheduler on {osutil.OS_NAME} (need cron, launchd or schtasks)")


def remove(name: str) -> None:
    if osutil.IS_MAC:
        path = plist_path(name)
        subprocess.run(["launchctl", "unload", path], capture_output=True)
        try:
            os.remove(path)
        except OSError:
            pass
    elif osutil.IS_WINDOWS:
        subprocess.run(["schtasks", "/delete", "/f", "/tn", _win_name(name)], capture_output=True)
    elif osutil.IS_LINUX and shutil.which("crontab"):
        try:
            _cron_write(_cron_without(name))
        except RuntimeError:
            pass


def installed(name: str) -> bool:
    if osutil.IS_MAC:
        return os.path.exists(plist_path(name))
    if osutil.IS_WINDOWS:
        return subprocess.run(["schtasks", "/query", "/tn", _win_name(name)], capture_output=True).returncode == 0
    if osutil.IS_LINUX and shutil.which("crontab"):
        return f"# cortex-{_slug(name)}" in [line.strip() for line in _cron_lines()]
    return False
