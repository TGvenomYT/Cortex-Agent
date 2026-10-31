"""
workers.py — the control panel for everything Cortex does in the background.

Cortex's background work is a set of small jobs: reminder and automation checks, nine
"awareness" collectors, an AI autonomy session, two daily briefings and a cloud sync —
plus any recurring tasks the user (or the agent) created. This module is the single
place that knows about all of them. It can list them with live status, switch each one
on/off, run one right now, and apply a whole-profile preset.

The built-in workers share one OS job, the tick (see tick.py), which decides what's due.
What's on is kept in settings as `workers_on`. How often each runs depends on the
machine: a server keeps the original cadences, a laptop runs far less often when plugged
in and only reminders and rules on battery. With nothing on, the tick job is removed, so
nothing wakes the machine at all. Recurring tasks keep their own job, because they run
at a time someone chose.

Nothing runs until someone chooses it: a fresh install starts with every worker off, and
the setup step asks what Cortex may keep track of (see TRACKING). What people type at
the shell and what's installed on their machine stay opt-in.
"""
from __future__ import annotations

import os
import sys
import time
import plistlib
import logging
import subprocess

import cortex
from cortex.core import scheduler
import cortex.storage.settings_store as settings_store
import cortex.util.osutil as osutil
import cortex.util.power as power

logger = logging.getLogger("cortex.workers")

_LA = os.path.expanduser("~/Library/LaunchAgents")
_UID = os.getuid() if hasattr(os, "getuid") else 0
_DOMAIN = f"gui/{_UID}"

# cost: light  = a few ms of CPU, negligible
#       medium = a second or two of CPU / ~150 MB while it runs
#       heavy  = wakes the AI model (GBs of RAM/GPU) — the real resource cost
# every   = seconds between runs on a server or a desktop
# ac      = on a laptop that's plugged in
# battery = on a laptop on battery; None means it waits for the charger. AI jobs still
#           need power.should_run("heavy") to agree, which by default means plugged in.
BUILTIN = [
    dict(key="reminders", label="com.cortex.reminders", name="Reminders", group="essentials",
         desc="Fires your reminders when they're due.", cost="light", ai=False,
         run=["run_once.py", "reminders"], every=60, ac=1800, battery=1800),
    dict(key="automation", label="com.cortex.automation", name="Automation rules", group="essentials",
         desc="Checks your if-this-then-that rules.", cost="light", ai=False,
         run=["run_once.py", "automation"], every=120, ac=600, battery=1800),

    dict(key="collector.system", label="com.cortex.collector.system", name="System stats", group="awareness",
         desc="Disk, memory, CPU and network snapshot.", cost="light", ai=False,
         run=["collectors/system_collector.py"], every=600, ac=1800, battery=None),
    dict(key="collector.process", label="com.cortex.collector.process", name="Running services", group="awareness",
         desc="Docker containers, open ports, services.", cost="light", ai=False,
         run=["collectors/process_collector.py"], every=180, ac=1800, battery=None),
    dict(key="collector.metrics", label="com.cortex.collector.metrics", name="Performance metrics", group="awareness",
         desc="CPU / memory history.", cost="light", ai=False,
         run=["collectors/metrics_collector.py"], every=180, ac=1800, battery=None),
    dict(key="collector.shell", label="com.cortex.collector.shell", name="Shell history", group="awareness",
         desc="Learns from the commands you run.", cost="light", ai=False,
         run=["collectors/shell_collector.py"], every=180, ac=1800, battery=None),
    dict(key="collector.git", label="com.cortex.collector.git", name="Git status", group="awareness",
         desc="Branch / changes across your tracked projects.", cost="light", ai=False,
         run=["collectors/git_collector.py"], every=300, ac=1800, battery=None),
    dict(key="collector.project", label="com.cortex.collector.project", name="Project scanner", group="awareness",
         desc="Project types, dependencies, TODOs.", cost="light", ai=False,
         run=["collectors/project_collector.py"], every=1800, ac=7200, battery=None),
    dict(key="collector.owner", label="com.cortex.collector.owner", name="Machine profile", group="awareness",
         desc="Full machine + owner profile — the heaviest collector.", cost="medium", ai=False,
         run=["collectors/owner_collector.py"], every=3600, ac=86400, battery=None),
    dict(key="collector.remote", label="com.cortex.collector.remote", name="Remote hosts", group="awareness",
         desc="Checks your server over SSH / Tailscale.", cost="light", ai=False,
         run=["collectors/remote_collector.py"], every=600, ac=1800, battery=None),
    dict(key="collector.github", label="com.cortex.collector.github", name="GitHub", group="awareness",
         desc="Repos, pull requests and issues.", cost="light", ai=False,
         run=["collectors/github_collector.py"], every=900, ac=3600, battery=None),

    dict(key="autonomy", label="com.cortex.autonomy", name="Autonomy", group="ai",
         desc="Cortex works on your goals by itself: plans, acts, reports. Wakes the AI model.",
         cost="heavy", ai=True, run=["run_once.py", "autonomy"], every=1800, ac=3600, battery=3600),
    dict(key="report-morning", label="com.cortex.report-morning", name="Morning briefing", group="ai",
         desc="Writes your daily briefing. Wakes the AI model.", cost="heavy", ai=True,
         run=["run_once.py", "report", "Morning Briefing"], daily=(9, 0)),
    dict(key="report-evening", label="com.cortex.report-evening", name="Evening wrap", group="ai",
         desc="Writes your end-of-day wrap-up. Wakes the AI model.", cost="heavy", ai=True,
         run=["run_once.py", "report", "Evening Wrap"], daily=(18, 0)),

    dict(key="sync", label="com.cortex.sync", name="Cloud sync", group="sync",
         desc="Pushes your local data to your server when it's reachable.", cost="light", ai=False,
         run=["run_once.py", "sync"], every=1800, ac=3600, battery=None),
]
_BY_KEY = {w["key"]: w for w in BUILTIN}

TICK = {"laptop": 600, "server": 60}       # seconds between ticks

GROUPS = [
    ("essentials", "Essentials", "Small checks that keep reminders and rules working."),
    ("awareness", "Machine awareness", "Collectors that let Cortex know what's going on on this computer. "
                                       "On a laptop they run when it's plugged in, and whenever you open Cortex."),
    ("ai", "AI jobs", "These wake the language model — the part that actually uses a lot of memory and battery."),
    ("sync", "Sync", ""),
    ("services", "Services", "Engines Cortex starts on demand."),
    ("tasks", "Your recurring tasks", "Jobs you (or Cortex, on request) scheduled."),
]

PROFILES = {
    "off": [],
    "essential": ["reminders", "automation"],
    "balanced": ["reminders", "automation", "collector.system", "collector.process",
                 "collector.metrics", "collector.git", "collector.project"],
    "full": [w["key"] for w in BUILTIN],
}
PROFILE_BLURB = {
    "off": "Nothing runs in the background. Reminders won't fire.",
    "essential": "Only reminders and rules. Practically free.",
    "balanced": "Plus this computer's health and your projects. No AI model in the background.",
    "full": "Everything, including AI autonomy and briefings.",
}


# What the setup step offers. Things that reveal what someone does stay off until they say so.
TRACKING = [
    dict(id="essentials", name="Reminders and alerts", on=True, keys=["reminders", "automation"],
         desc="Fires your reminders, and warns you if the disk or CPU gets tight."),
    dict(id="health", name="This computer's health", on=True,
         keys=["collector.system", "collector.process", "collector.metrics"],
         desc="Disk, memory, what's running. So it can answer without checking every time."),
    dict(id="projects", name="Your projects", on=True, keys=["collector.git", "collector.project"],
         desc="Git status and project details, for the folders you list below."),
    dict(id="shell", name="Commands you type", on=False, keys=["collector.shell"],
         desc="Reads your shell history so it knows what you've been doing. Passwords in it are masked."),
    dict(id="machine", name="Apps and setup", on=False, keys=["collector.owner"],
         desc="Installed apps, tools, Wi-Fi network names and SSH key names."),
]


def command(w: dict) -> list:
    """How to run a worker once. --force: whoever calls this has made the power and on/off checks."""
    run = w["run"]
    script = os.path.join(cortex.ROOT, run[0])
    return [sys.executable, script] + (["--force"] + run[1:] if run[0] == "run_once.py" else [])


def machine_kind(batt: dict | None = None) -> str:
    """'laptop' or 'server'. A battery means laptop, unless the user said otherwise."""
    choice = settings_store.section("resources").get("machine", "auto")
    if choice in ("laptop", "server"):
        return choice
    return "laptop" if (batt or power.battery_state())["has_battery"] else "server"


def power_mode(batt: dict | None = None) -> str:
    """'server', 'plugged' or 'battery': which cadence applies right now."""
    batt = batt or power.battery_state()
    if machine_kind(batt) == "server":
        return "server"
    return "plugged" if batt["on_ac"] else "battery"


def cadence(w: dict, mode: str) -> int | None:
    return {"server": w["every"], "plugged": w["ac"], "battery": w["battery"]}[mode]


def power_info() -> dict:
    """For the settings page: what's in force on this machine, in plain words."""
    batt = power.battery_state()
    mode = power_mode(batt)
    return {"choice": settings_store.section("resources").get("machine", "auto"), "kind": machine_kind(batt),
            "mode": mode, "note": {
                "server": "Runs as a server: background work keeps its full schedule.",
                "plugged": "Plugged in: background work runs less often than on a server, to keep the laptop idle.",
                "battery": "On battery: only reminders and rules run. The rest waits until you plug in or open Cortex.",
            }[mode]}


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
    if n < 3600:
        return f"{round(n / 60)} min"
    return f"{round(n / 3600)} h"


def _cadence_text(w: dict, kind: str) -> str:
    if w["key"] == "reminders":
        return "exactly on time"
    if "daily" in w:
        text = f"daily {w['daily'][0]:02d}:{w['daily'][1]:02d}"
    else:
        text = f"every {_fmt_secs(w['every'] if kind == 'server' else w['ac'])}"
    if kind == "server":
        return text
    if w.get("battery") is None and "daily" not in w or w["ai"] and power.ai_jobs_only_on_ac():
        return f"{text}, plugged in only"
    return text if w.get("battery") in (None, w.get("ac")) else f"{text}, every {_fmt_secs(w['battery'])} on battery"


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
def _builtin_rows(loaded: dict) -> list[dict]:
    from cortex.core import tick
    keys, up, state, kind = enabled_keys(), _tick_up(loaded), tick.load_state(), machine_kind()
    return [{**{k: w[k] for k in ("key", "name", "desc", "group", "cost", "ai")},
             "installed": True, "available": True, "on": w["key"] in keys and up,
             "running": state.get("running") == w["key"], "last_exit": state["exit"].get(w["key"]),
             "cadence": _cadence_text(w, kind), "last_run": state["last"].get(w["key"]), "removable": False}
            for w in BUILTIN]


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
    _ready()
    loaded = _loaded() if osutil.IS_MAC else {}
    rows = _builtin_rows(loaded) + [_voice_row()]
    try:
        rows += _task_rows(loaded)
    except Exception as e:
        logger.warning(f"task listing failed: {e}")
    return rows


def enabled_keys() -> set:
    return set(settings_store.load().get("workers_on") or [])


def is_on(key: str) -> bool:
    """Is this built-in worker switched on, with the tick there to run it?"""
    _ready()
    return key in enabled_keys() and _tick_up()


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

    if key not in _BY_KEY:
        return {"ok": False, "on": False, "error": f"unknown worker '{key}'"}
    _ready()
    r = _apply(enabled_keys() | {key} if on else enabled_keys() - {key})
    return {**r, "on": key in enabled_keys()}


def _tick_up(loaded: dict | None = None) -> bool:
    if osutil.IS_MAC:
        return "com.cortex.tick" in (_loaded() if loaded is None else loaded)
    return scheduler.installed("tick")


def _sync_tick(keys: set) -> None:
    """Keep the one OS job in step with what's on. Nothing on, no job, nothing wakes."""
    if not keys:
        scheduler.remove("tick")
        return
    seconds = TICK[machine_kind()]
    if not _tick_up() or settings_store.load().get("workers_tick") != seconds:
        scheduler.install("tick", [sys.executable, os.path.join(cortex.ROOT, "run_once.py"), "tick"],
                          {"type": "interval", "seconds": seconds, "at_load": True})
        settings_store.save({"workers_tick": seconds})


def _apply(keys: set) -> dict:
    """Make this the set of workers that are on."""
    from cortex.core import tick
    added = keys - enabled_keys()
    try:
        _sync_tick(keys)
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    if keys and not _tick_up():
        return {"ok": False, "error": "the operating system wouldn't accept the background job"}
    tick.stamp(k for k in added if _BY_KEY[k]["ai"])    # AI jobs wait for their turn, as they always did
    settings_store.save({"workers_on": sorted(keys)})
    if keys:        # run what's due now, and pick up any reminder that's already close
        scheduler.detach([sys.executable, os.path.join(cortex.ROOT, "run_once.py"), "tick"], "tick")
    return {"ok": True}


def _ready() -> None:
    """The first run after moving to one job: keep what was on, and retire the old jobs."""
    if "workers_on" in settings_store.load():
        return
    if osutil.IS_MAC:
        loaded, disabled = _loaded(), _disabled()
        keys = {w["key"] for w in BUILTIN if w["label"] in loaded and w["label"] not in disabled}
    else:
        keys = {w["key"] for w in BUILTIN if scheduler.installed(w["key"])}
    for w in BUILTIN:
        scheduler.remove(w["key"])
    if keys:
        _apply(keys)
    settings_store.save({"workers_on": sorted(keys)})


def ensure() -> None:
    """When Cortex starts: finish any move off the old jobs, make sure the tick matches
    what's on, and catch up on anything out of date."""
    from cortex.core import tick
    try:
        _ready()
        _sync_tick(enabled_keys())
        tick.refresh_context()
    except Exception as e:
        logger.warning(f"background startup: {e}")


def set_machine(choice: str) -> dict:
    """auto, laptop or server."""
    if choice not in ("auto", "laptop", "server"):
        return {"ok": False, "error": "use auto, laptop or server"}
    settings_store.update_section("resources", {"machine": choice})
    try:
        _sync_tick(enabled_keys())
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, **power_info()}


def choose_tracking(chosen: list) -> dict:
    """The setup step: turn on what the person agreed to, and nothing else."""
    _ready()
    offered = {k for g in TRACKING for k in g["keys"]}
    want = {k for g in TRACKING if g["id"] in chosen for k in g["keys"]}
    r = _apply((enabled_keys() - offered) | want)
    return {"ok": r["ok"], "errors": [r["error"]] if r.get("error") else []}


def apply_profile(name: str) -> dict:
    """Switch the built-in workers to a preset. 'off' also pauses user tasks."""
    if name not in PROFILES:
        return {"ok": False, "error": f"unknown profile '{name}' (use {', '.join(PROFILES)})"}
    _ready()
    errors = []
    r = _apply(set(PROFILES[name]))
    if r.get("error"):
        errors.append(r["error"])
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
        cmd = command(_BY_KEY[key])
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
    """Guard for job entrypoints: should this invocation do anything? The tick has already
    checked; this catches an old per-worker job firing before Cortex has been opened again."""
    if job == "task":
        import cortex.core.tasks as tasks_mod
        task = tasks_mod.get_task(arg or "")
        if task and not tasks_mod.is_enabled(task):
            return False, f"task {arg} is paused"
        return True, "task enabled"
    want = ["run_once.py", job] + ([arg] if arg else [])
    w = next((x for x in BUILTIN if x["run"][:len(want)] == want), None)
    on = settings_store.load().get("workers_on")
    if not w or on is None:
        return True, "unmanaged job"
    if w["key"] not in on:
        return False, f"{w['name']} is switched off"
    return True, "enabled"


def describe(rows: list[dict] | None = None) -> str:
    """Plain-language status — what the chat agent / CLI shows."""
    rows = rows if rows is not None else list_workers()
    s = summary(rows)
    lines = [f"Background profile: {s['profile']} — {s['on']}/{s['total']} built-in workers on, "
             f"{s['ai_on']} AI job(s) on, {s['tasks_on']}/{s['tasks_total']} of your tasks on.",
             power_info()["note"]]
    for gid, gname, _ in GROUPS:
        grp = [r for r in rows if r["group"] == gid]
        if not grp:
            continue
        lines.append(f"{gname}:")
        for r in grp:
            lines.append(f"  [{'ON ' if r['on'] else 'off'}] {r['key']:<18} {r['name']} · {r['cadence']} · {r['cost']}")
    return "\n".join(lines)
