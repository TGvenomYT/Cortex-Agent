"""
tick.py — the one background job.

Giving each worker its own OS job suits a server, but on a laptop it meant about 185
separate wake-ups an hour, and those are what drain a battery: a run costs almost
nothing, but each one stops the CPU resting. So the OS starts this instead, every 10
minutes on a laptop and every minute on a server. It runs whatever is due for how the
machine is powered right now, one thing at a time at low priority, then exits. On
battery that's reminders and rules; the rest waits for the charger or for Cortex to be
opened, when refresh_context() catches up on what it needs. With nothing switched on
the job doesn't exist at all.

Keep what this imports light: it runs all day. Worker on/off and cadences live in workers.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta

import cortex
from cortex.core import alarm, scheduler, workers
import cortex.util.power as power

TIMEOUT = 180           # a stuck collector mustn't hold up the rest
AI_TIMEOUT = 1800
GRACE = 6 * 3600        # a briefing that's more than this late isn't worth writing
SLACK = 0.1             # of a tick: 29:59 since the last run counts as the 30 minutes


def _path(name: str) -> str:
    return os.path.expanduser(f"~/.cortex/state/{name}")


def load_state() -> dict:
    try:
        with open(_path("tick.json"), encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        state = {}
    state.setdefault("last", {})
    state.setdefault("exit", {})
    return state


def save_state(state: dict) -> None:
    os.makedirs(_path(""), exist_ok=True)
    tmp = _path(f"tick.{os.getpid()}.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp, _path("tick.json"))


def stamp(keys, now: float | None = None) -> None:
    """Count these as just run, so switching one on doesn't start it straight away."""
    state = load_state()
    state["last"].update({k: now or time.time() for k in keys})
    save_state(state)


def _locked() -> bool:
    try:
        return time.time() - os.path.getmtime(_path("tick.lock")) < AI_TIMEOUT + 60
    except OSError:
        return False


def _lock() -> bool:
    """launchd never overlaps a job with itself, but cron and Task Scheduler can."""
    if _locked():
        return False
    os.makedirs(_path(""), exist_ok=True)
    try:
        os.remove(_path("tick.lock"))       # one left by a run that was killed
    except OSError:
        pass
    try:
        os.close(os.open(_path("tick.lock"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except OSError:
        return False


def _unlock() -> None:
    try:
        os.remove(_path("tick.lock"))
    except OSError:
        pass


def is_due(w: dict, last: float, now: float, mode: str) -> bool:
    if "daily" in w:
        slot = datetime.fromtimestamp(now).replace(hour=w["daily"][0], minute=w["daily"][1], second=0, microsecond=0)
        if slot.timestamp() > now:
            slot -= timedelta(days=1)
        return last < slot.timestamp() and now - slot.timestamp() <= GRACE
    every = workers.cadence(w, mode)
    return every is not None and now - last >= every - SLACK * workers.TICK["server" if mode == "server" else "laptop"]


def due_now(keys, state: dict, now: float | None = None, batt: dict | None = None) -> list[dict]:
    """What to run this tick, in order. Skipped work stays due, so it runs later instead of being lost."""
    now = now or time.time()
    batt = batt or power.battery_state()
    mode = workers.power_mode(batt)
    due = [w for w in workers.BUILTIN
           if w["key"] in keys and is_due(w, state["last"].get(w["key"], 0), now, mode)]
    if not due or not power.should_run("cheap")[0]:
        return []
    plain, ai = [w for w in due if not w["ai"]], [w for w in due if w["ai"]]
    return plain + ai[:1] if ai and power.should_run("heavy")[0] else plain     # one model load a tick


def stale(keys, state: dict, now: float | None = None) -> list[dict]:
    """Light collectors that have gone past their server cadence."""
    now = now or time.time()
    return [w for w in workers.BUILTIN if w["key"] in keys and w["group"] == "awareness" and w["cost"] == "light"
            and now - state["last"].get(w["key"], 0) >= w["every"]]


def refresh_context() -> bool:
    """Someone's using Cortex, so catch up on what it knows, whatever the power. This is why
    collectors can stop on battery. Detached, so nobody waits; a no-op when nothing's stale."""
    if not stale(workers.enabled_keys(), load_state()) or _locked():
        return False
    scheduler.detach([sys.executable, os.path.join(cortex.ROOT, "run_once.py"), "tick", "--context"], "tick")
    return True


def _env() -> dict:
    # cron and launchd start with a bare PATH, and collectors need docker, git and gh
    return os.environ if os.name == "nt" else {**os.environ, "PATH": f"{scheduler.PATH}:{os.environ.get('PATH', '')}"}


def run(w: dict, state: dict) -> None:
    state["running"] = w["key"]
    save_state(state)
    with open(scheduler._log(w["key"]), "ab") as log:
        try:
            code = subprocess.run(workers.command(w), cwd=cortex.ROOT, env=_env(), stdout=log, stderr=log,
                                  timeout=AI_TIMEOUT if w["ai"] else TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            code = 124
    state["last"][w["key"]], state["exit"][w["key"]] = time.time(), code
    state.pop("running", None)
    save_state(state)


def main() -> int:
    keys = workers.enabled_keys()
    if not keys:
        return 0
    context = "--context" in sys.argv
    if not context:
        alarm.check()
    if not _lock():
        return 0
    try:
        if hasattr(os, "nice"):
            os.nice(10)
        state = load_state()
        for w in (stale if context else due_now)(keys, state):
            run(w, state)
    finally:
        _unlock()
    return 0
