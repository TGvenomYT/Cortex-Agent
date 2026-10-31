"""
workers.py — the control panel for everything Cortex does in the background.

Cortex's background work is a set of small scheduled jobs (macOS launchd): reminder
and automation checks, nine "awareness" collectors, an AI autonomy session, two daily
briefings and a cloud sync — plus any recurring tasks the user (or the agent) created.
This module is the single place that knows about all of them. It can list them with
live status, switch each one on/off (really unloading it from launchd, so it stops
waking the machine), run one right now, and apply a whole-profile preset.

Source of truth is launchd itself (`launchctl list` / `print-disabled`), so what the UI
shows is what the OS is actually doing — nothing is cached or remembered separately.
"Off" means bootout + disable, so a worker also stays off across logins/reboots.

Built-in workers are macOS-only (that's where install_scheduled.sh puts them). User
tasks go through tasks.py and work on macOS/Linux/Windows.
"""
from __future__ import annotations

import os
import sys
import time
import plistlib
import logging
import subprocess

import cortex
import cortex.util.osutil as osutil

logger = logging.getLogger("cortex.workers")

_LA = os.path.expanduser("~/Library/LaunchAgents")
_UID = os.getuid() if hasattr(os, "getuid") else 0
_DOMAIN = f"gui/{_UID}"

# cost: light  = a few ms of CPU, negligible
#       medium = a second or two of CPU / ~150 MB while it runs
#       heavy  = wakes the AI model (GBs of RAM/GPU) — the real resource cost
BUILTIN = [
    dict(key="reminders", label="com.cortex.reminders", name="Reminders", group="essentials",
         desc="Fires your reminders when they're due.", cost="light", ai=False,
         run=["run_once.py", "reminders"]),
    dict(key="automation", label="com.cortex.automation", name="Automation rules", group="essentials",
         desc="Checks your if-this-then-that rules.", cost="light", ai=False,
         run=["run_once.py", "automation"]),

    dict(key="collector.system", label="com.cortex.collector.system", name="System stats", group="awareness",
         desc="Disk, memory, CPU and network snapshot.", cost="light", ai=False,
         run=["collectors/system_collector.py"]),
    dict(key="collector.process", label="com.cortex.collector.process", name="Running services", group="awareness",
         desc="Docker containers, open ports, services.", cost="light", ai=False,
         run=["collectors/process_collector.py"]),
    dict(key="collector.metrics", label="com.cortex.collector.metrics", name="Performance metrics", group="awareness",
         desc="CPU / memory history.", cost="light", ai=False,
         run=["collectors/metrics_collector.py"]),
    dict(key="collector.shell", label="com.cortex.collector.shell", name="Shell history", group="awareness",
         desc="Learns from the commands you run.", cost="light", ai=False,
         run=["collectors/shell_collector.py"]),
    dict(key="collector.git", label="com.cortex.collector.git", name="Git status", group="awareness",
         desc="Branch / changes across your tracked projects.", cost="light", ai=False,
         run=["collectors/git_collector.py"]),
    dict(key="collector.project", label="com.cortex.collector.project", name="Project scanner", group="awareness",
         desc="Project types, dependencies, TODOs.", cost="light", ai=False,
         run=["collectors/project_collector.py"]),
    dict(key="collector.owner", label="com.cortex.collector.owner", name="Machine profile", group="awareness",
         desc="Full machine + owner profile — the heaviest collector.", cost="medium", ai=False,
         run=["collectors/owner_collector.py"]),
    dict(key="collector.remote", label="com.cortex.collector.remote", name="Remote hosts", group="awareness",
         desc="Checks your server over SSH / Tailscale.", cost="light", ai=False,
         run=["collectors/remote_collector.py"]),
    dict(key="collector.github", label="com.cortex.collector.github", name="GitHub", group="awareness",
         desc="Repos, pull requests and issues.", cost="light", ai=False,
         run=["collectors/github_collector.py"]),

    dict(key="autonomy", label="com.cortex.autonomy", name="Autonomy", group="ai",
         desc="Cortex works on your goals by itself: plans, acts, reports. Wakes the AI model.",
         cost="heavy", ai=True, run=["run_once.py", "autonomy"]),
    dict(key="report-morning", label="com.cortex.report-morning", name="Morning briefing", group="ai",
         desc="Writes your daily briefing. Wakes the AI model.", cost="heavy", ai=True,
         run=["run_once.py", "report", "Morning Briefing"]),
    dict(key="report-evening", label="com.cortex.report-evening", name="Evening wrap", group="ai",
         desc="Writes your end-of-day wrap-up. Wakes the AI model.", cost="heavy", ai=True,
         run=["run_once.py", "report", "Evening Wrap"]),

    dict(key="sync", label="com.cortex.sync", name="Cloud sync", group="sync",
         desc="Pushes your local data to your server when it's reachable.", cost="light", ai=False,
         run=["run_once.py", "sync"]),
]
_BY_KEY = {w["key"]: w for w in BUILTIN}

GROUPS = [
    ("essentials", "Essentials", "Small checks that keep reminders and rules working."),
    ("awareness", "Machine awareness", "Collectors that let Cortex know what's going on on this Mac."),
    ("ai", "AI jobs", "These wake the language model — the part that actually uses a lot of memory and battery."),
    ("sync", "Sync", ""),
    ("services", "Services", "Engines Cortex starts on demand."),
    ("tasks", "Your recurring tasks", "Jobs you (or Cortex, on request) scheduled."),
]

PROFILES = {
    "off": [],
    "essential": ["reminders", "automation"],
    "balanced": ["reminders", "automation", "collector.system", "collector.process",
                 "collector.metrics", "collector.shell", "collector.git", "collector.project", "sync"],
    "full": [w["key"] for w in BUILTIN],
}
PROFILE_BLURB = {
    "off": "Nothing runs in the background. Reminders won't fire.",
    "essential": "Only reminders and rules. Practically free.",
    "balanced": "Plus machine awareness and sync. No AI model in the background.",
    "full": "Everything, including AI autonomy and briefings.",
}


# ── launchd plumbing ──────────────────────────────────────────────────────
def _launchctl(*args, timeout: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True, timeout=timeout)


def _loaded() -> dict:
    """label -> {pid, exit} for every job launchd currently has loaded."""
    out = {}
    try:
        for line in _launchctl("list").stdout.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            pid, status, label = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if label.startswith("com.cortex."):
                out[label] = {"pid": int(pid) if pid.isdigit() else None,
                              "exit": int(status) if status.lstrip("-").isdigit() else None}
    except Exception as e:
        logger.warning(f"launchctl list failed: {e}")
    return out


def _disabled() -> set:
    out = set()
    try:
        for line in _launchctl("print-disabled", _DOMAIN).stdout.splitlines():
            line = line.strip()
            if "=> disabled" in line or "=> true" in line:
                out.add(line.split('"')[1] if '"' in line else line.split()[0])
    except Exception as e:
        logger.warning(f"launchctl print-disabled failed: {e}")
    return out


def _plist_path(label: str) -> str:
    return os.path.join(_LA, f"{label}.plist")


def _plist_info(label: str) -> dict:
    try:
        with open(_plist_path(label), "rb") as f:
            return plistlib.load(f)
    except Exception:
        return {}


def _fmt_secs(n: int) -> str:
    if n < 120:
        return f"{n} s"
    if n < 7200:
        return f"{round(n / 60)} min"
    return f"{round(n / 3600)} h"


def _cadence(info: dict) -> str:
    if "StartInterval" in info:
        return f"every {_fmt_secs(int(info['StartInterval']))}"
    sci = info.get("StartCalendarInterval")
    if isinstance(sci, list):
        sci = sci[0] if sci else None
    if isinstance(sci, dict) and "Hour" in sci:
        return f"daily {int(sci['Hour']):02d}:{int(sci.get('Minute', 0)):02d}"
    return "on demand"


def _last_run(info: dict) -> float | None:
    """Most recent write to the job's log files ≈ when it last ran."""
    best = None
    for k in ("StandardErrorPath", "StandardOutPath"):
        p = info.get(k)
        try:
            if p and os.path.exists(p):
                m = os.path.getmtime(p)
                best = m if best is None or m > best else best
        except OSError:
            pass
    return best


# ── listing ───────────────────────────────────────────────────────────────
def _builtin_rows(loaded: dict, disabled: set) -> list[dict]:
    rows = []
    for w in BUILTIN:
        info = _plist_info(w["label"])
        installed = bool(info)
        st = loaded.get(w["label"])
        rows.append({
            **{k: w[k] for k in ("key", "name", "desc", "group", "cost", "ai")},
            "installed": installed,
            "available": osutil.IS_MAC and installed,
            "on": bool(st) and w["label"] not in disabled,
            "running": bool(st and st["pid"]),
            "last_exit": st["exit"] if st else None,
            "cadence": _cadence(info),
            "last_run": _last_run(info),
            "removable": False,
        })
    return rows


def _task_rows(loaded: dict) -> list[dict]:
    import cortex.core.tasks as tasks_mod
    rows = []
    for t in tasks_mod.list_tasks():
        label = f"com.cortex.task.{t['id']}"
        st = loaded.get(label)
        enabled = t.get("enabled", True)
        rows.append({
            "key": f"task.{t['id']}", "name": t.get("desc") or t.get("prompt", "")[:50],
            "desc": t.get("prompt", "")[:140], "group": "tasks", "cost": "heavy", "ai": True,
            "installed": True, "available": True,
            "on": enabled and (bool(st) if osutil.IS_MAC else True),
            "running": bool(st and st["pid"]), "last_exit": st["exit"] if st else None,
            "cadence": t.get("human") or t.get("schedule", ""),
            "last_run": _last_run(_plist_info(label)), "removable": True,
        })
    return rows


def _voice_row() -> dict:
    try:
        import cortex.voice.lifecycle as life
        up = life.is_up()
        idle = life.idle_seconds()
    except Exception:
        up, idle = False, None
    return {
        "key": "voice", "name": "Voice engine",
        "desc": "On-device speech-to-text and text-to-speech (loads ~2–3 GB). Starts when you tap the mic "
                "or speaker and shuts itself down when idle.",
        "group": "services", "cost": "medium", "ai": True, "installed": True, "available": True,
        "on": up, "running": up, "last_exit": None,
        "cadence": "on demand", "last_run": (time.time() - idle) if (up and idle is not None) else None,
        "removable": False,
    }


def list_workers() -> list[dict]:
    loaded, disabled = (_loaded(), _disabled()) if osutil.IS_MAC else ({}, set())
    rows = _builtin_rows(loaded, disabled) if osutil.IS_MAC else []
    rows += [_voice_row()]
    try:
        rows += _task_rows(loaded)
    except Exception as e:
        logger.warning(f"task listing failed: {e}")
    return rows


def is_on(key: str) -> bool:
    """Is this built-in worker actually scheduled right now? True where there's nothing to check."""
    if not osutil.IS_MAC:
        return True
    w = _BY_KEY.get(key)
    return bool(w) and w["label"] in _loaded() and w["label"] not in _disabled()


def current_profile(rows: list[dict] | None = None) -> str:
    rows = rows if rows is not None else list_workers()
    on = {r["key"] for r in rows if r["group"] not in ("tasks", "services") and r["on"]}
    tasks_on = any(r["on"] for r in rows if r["group"] == "tasks")
    for name, keys in PROFILES.items():
        if on == set(keys):
            # "off" means nothing runs, so it can't be the answer while a task is still on.
            return "custom" if name == "off" and tasks_on else name
    return "custom"


def summary(rows: list[dict] | None = None) -> dict:
    rows = rows if rows is not None else list_workers()
    builtin = [r for r in rows if r["group"] not in ("tasks", "services")]
    tasks = [r for r in rows if r["group"] == "tasks"]
    return {
        "profile": current_profile(rows),
        "on": sum(1 for r in builtin if r["on"]),
        "total": len(builtin),
        "ai_on": sum(1 for r in rows if r["on"] and r["ai"] and r["group"] != "services"),
        "tasks_on": sum(1 for r in tasks if r["on"]),
        "tasks_total": len(tasks),
        "mac": osutil.IS_MAC,
    }


# ── control ───────────────────────────────────────────────────────────────
def _find(key: str) -> dict | None:
    return next((r for r in list_workers() if r["key"] == key), None)


def resolve_key(text: str) -> str | None:
    """Map what the user/model said ('autonomy', 'Morning briefing', 'system stats') to a worker key."""
    t = (text or "").strip().lower()
    if not t:
        return None
    rows = list_workers()
    for r in rows:                                   # exact key or exact name
        if t in (r["key"].lower(), r["name"].lower()):
            return r["key"]
    for r in rows:                                   # key suffix, e.g. 'system' -> collector.system
        if r["key"].lower().endswith("." + t):
            return r["key"]
    hits = [r for r in rows if t in r["name"].lower() or t in r["key"].lower()]
    return hits[0]["key"] if len(hits) == 1 else None


def set_enabled(key: str, on: bool) -> dict:
    """Turn one worker on/off. Returns {ok, on, error?}."""
    if key == "voice":
        import cortex.voice.lifecycle as life
        if on:
            return {"ok": bool(life.start()), "on": life.is_up()}
        life.stop()
        return {"ok": True, "on": life.is_up()}

    if key.startswith("task."):
        import cortex.core.tasks as tasks_mod
        try:
            tasks_mod.set_task_enabled(key[5:], bool(on))
            return {"ok": True, "on": bool(on)}
        except Exception as e:
            return {"ok": False, "on": not on, "error": str(e)}

    w = _BY_KEY.get(key)
    if not w:
        return {"ok": False, "on": False, "error": f"unknown worker '{key}'"}
    if not osutil.IS_MAC:
        return {"ok": False, "on": False, "error": "built-in background workers are macOS-only"}
    plist = _plist_path(w["label"])
    if not os.path.exists(plist):
        return {"ok": False, "on": False,
                "error": "not installed — run install_scheduled.sh to set up the scheduled jobs"}
    svc = f"{_DOMAIN}/{w['label']}"
    try:
        if on:
            _launchctl("enable", svc)
            if w["label"] not in _loaded():
                r = _launchctl("bootstrap", _DOMAIN, plist)
                if r.returncode != 0 and w["label"] not in _loaded():
                    # last resort: the legacy loader
                    _launchctl("load", "-w", plist)
        else:
            _launchctl("bootout", svc)          # also stops it if it's mid-run
            _launchctl("disable", svc)          # …and keeps it off across logins/reboots
            if w["label"] in _loaded():
                _launchctl("unload", "-w", plist)
    except Exception as e:
        return {"ok": False, "on": not on, "error": str(e)}
    now_on = w["label"] in _loaded() and w["label"] not in _disabled()
    return {"ok": now_on == bool(on), "on": now_on}


def apply_profile(name: str) -> dict:
    """Switch the built-in workers to a preset. 'off' also pauses user tasks."""
    if name not in PROFILES:
        return {"ok": False, "error": f"unknown profile '{name}' (use {', '.join(PROFILES)})"}
    want = set(PROFILES[name])
    errors = []
    for w in BUILTIN:
        r = set_enabled(w["key"], w["key"] in want)
        if not r.get("ok") and r.get("error"):
            errors.append(f"{w['key']}: {r['error']}")
    if name == "off":
        try:
            import cortex.core.tasks as tasks_mod
            for t in tasks_mod.list_tasks():
                tasks_mod.set_task_enabled(t["id"], False)
        except Exception as e:
            errors.append(f"tasks: {e}")
    return {"ok": not errors, "profile": current_profile(), "errors": errors[:5]}


def run_now(key: str) -> dict:
    """Run one worker once, right now, detached (ignores the battery gate)."""
    if key.startswith("task."):
        cmd = [sys.executable, os.path.join(cortex.ROOT, "run_once.py"), "--force", "task", key[5:]]
    elif key in _BY_KEY:
        run = _BY_KEY[key]["run"]
        script = os.path.join(cortex.ROOT, run[0])
        cmd = [sys.executable, script] + (["--force"] + run[1:] if run[0] == "run_once.py" else [])
    else:
        return {"ok": False, "error": f"can't run '{key}' on demand"}
    try:
        logdir = os.path.join(cortex.ROOT, "logs")
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, f"run-now-{key}.log"), "ab") as lf:
            subprocess.Popen(cmd, cwd=cortex.ROOT, stdout=lf, stderr=lf, start_new_session=True)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def is_allowed(job: str, arg: str | None = None) -> tuple[bool, str]:
    """Guard for job entrypoints: should this scheduled invocation do anything?
    launchd is the real off-switch; this is a belt-and-braces check for the case where
    a job still fires (e.g. a stale plist) after the user switched it off."""
    if job == "task":
        import cortex.core.tasks as tasks_mod
        task = tasks_mod.get_task(arg or "")
        if task and not tasks_mod.is_enabled(task):
            return False, f"task {arg} is paused"
        return True, "task enabled"
    if not osutil.IS_MAC:
        return True, "not macOS"
    want = ["run_once.py", job] + ([arg] if arg else [])
    w = next((x for x in BUILTIN if x["run"][:len(want)] == want), None)
    if not w:
        return True, "unmanaged job"
    if w["label"] in _disabled():
        return False, f"{w['name']} is switched off"
    return True, "enabled"


def describe(rows: list[dict] | None = None) -> str:
    """Plain-language status — what the chat agent / CLI shows."""
    rows = rows if rows is not None else list_workers()
    s = summary(rows)
    lines = [f"Background profile: {s['profile']} — {s['on']}/{s['total']} built-in workers on, "
             f"{s['ai_on']} AI job(s) on, {s['tasks_on']}/{s['tasks_total']} of your tasks on."]
    for gid, gname, _ in GROUPS:
        grp = [r for r in rows if r["group"] == gid]
        if not grp:
            continue
        lines.append(f"{gname}:")
        for r in grp:
            lines.append(f"  [{'ON ' if r['on'] else 'off'}] {r['key']:<18} {r['name']} · {r['cadence']} · {r['cost']}")
    return "\n".join(lines)
