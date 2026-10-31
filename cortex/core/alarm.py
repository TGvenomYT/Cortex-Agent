"""
alarm.py — rings reminders on time without anything polling for them.

Nothing runs while the next reminder is far off. Whatever changes a reminder calls
sync(), which notes when the next one is due. Once that's close, a small process sleeps
until the moment and rings it; if it's far off, the tick starts that process when the
time gets near. Works the same on every OS. A one-off launchd job couldn't do this: to
set the next alarm it would have to reload itself, and launchd kills a job it unloads.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import cortex
from cortex.core import scheduler
import cortex.storage.settings_store as settings_store

logger = logging.getLogger("cortex.alarm")

HORIZON = 900       # must be longer than the slowest tick, or a reminder could slip past it
_NAP = 15           # looks again this often: a sooner reminder may have been added meanwhile
_RING_TIMEOUT = 120


def _path(name: str) -> str:
    return os.path.expanduser(f"~/.cortex/state/{name}")


def _stamp(due) -> float:
    return (datetime.fromisoformat(due) if isinstance(due, str) else due).timestamp()


def next_due() -> float | None:
    try:
        with open(_path("next_reminder"), encoding="utf-8") as f:
            return float(f.read())
    except (OSError, ValueError):
        return None


def _runner() -> str:
    return os.path.join(cortex.ROOT, "run_once.py")


def _reminders_on() -> bool:
    return "reminders" in (settings_store.load().get("workers_on") or [])


def _running() -> bool:
    try:
        return time.time() - os.path.getmtime(_path("alarm.beat")) < _NAP * 3
    except OSError:
        return False


def _beat() -> None:
    with open(_path("alarm.beat"), "w", encoding="utf-8"):
        pass


def sync(db) -> None:
    """Note when the next reminder is due, and start waiting for it if that's soon."""
    try:
        times = [_stamp(r["due_at"]) for r in db.get_all_reminders(include_fired=False) if r.get("due_at")]
        os.makedirs(_path(""), exist_ok=True)
        tmp = _path(f"next_reminder.{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(str(min(times)) if times else "")
        os.replace(tmp, _path("next_reminder"))
        check()
    except Exception as e:      # a reminder was saved; the tick will still find it
        logger.warning(f"couldn't set the next alarm: {e}")


def check() -> None:
    """Start the sleeper if a reminder is close. Cheap, and starts nothing otherwise."""
    due = next_due()
    if due is not None and due - time.time() <= HORIZON and _reminders_on() and not _running():
        scheduler.detach([sys.executable, _runner(), "alarm"], "alarm")


def _ring() -> None:
    with open(os.path.expanduser("~/.cortex/logs/reminders.log"), "ab") as log:
        subprocess.run([sys.executable, _runner(), "--force", "reminders"], cwd=cortex.ROOT,
                       stdout=log, stderr=log, timeout=_RING_TIMEOUT)


def main() -> int:
    """The sleeper: wait for each reminder that's close, ring it, and exit when none are."""
    if _running():
        return 0
    try:
        while _reminders_on():
            _beat()
            due = next_due()
            if due is None or due - time.time() > HORIZON:
                return 0
            wait = due - time.time()
            if wait > 0:
                time.sleep(min(wait, _NAP))
                continue
            _ring()
            if next_due() == due:   # it couldn't be cleared: leave it to the tick, don't spin
                return 0
        return 0
    finally:
        try:
            os.remove(_path("alarm.beat"))
        except OSError:
            pass
