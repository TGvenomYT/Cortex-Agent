"""
projects.py — detect what kind of project a folder is and run its standard steps.

The run_project tool asks for a step by name (dev/start/build/test/install); this turns
that into the real command for the project in that folder. dev/start are long-running
servers, so they're started in the background with a log instead of blocking the agent.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time

STEPS = ("dev", "start", "build", "test", "install")
_LONG_RUNNING = ("dev", "start")


def tracked(rows, config: dict) -> list[dict]:
    """Projects from the database plus any listed under `projects:` in config.yaml."""
    merged = {r["path"]: dict(r) for r in rows}
    for p in config.get("projects") or []:
        if p.get("path") and p["path"] not in merged:
            merged[p["path"]] = {"name": p.get("name") or os.path.basename(p["path"].rstrip("/")), "path": p["path"]}
    return list(merged.values())


def _node(path: str, step: str) -> str | None:
    try:
        with open(os.path.join(path, "package.json"), encoding="utf-8") as f:
            scripts = json.load(f).get("scripts") or {}
    except (OSError, ValueError):
        scripts = {}
    pm = ("pnpm" if os.path.exists(os.path.join(path, "pnpm-lock.yaml"))
          else "yarn" if os.path.exists(os.path.join(path, "yarn.lock")) else "npm")
    if step == "install":
        return f"{pm} install"
    if step in scripts:
        return f"{pm} run {step}"
    if step == "start":
        return f"{pm} start"
    return None


def _python(path: str, step: str) -> str | None:
    if step == "install":
        if os.path.exists(os.path.join(path, "requirements.txt")):
            return "python3 -m pip install -r requirements.txt"
        return "python3 -m pip install -e ."
    if step == "test":
        return "python3 -m pytest -q"
    for entry in ("manage.py", "main.py", "app.py", "run.py"):
        if os.path.exists(os.path.join(path, entry)) and step in ("dev", "start"):
            return f"python3 manage.py runserver" if entry == "manage.py" else f"python3 {entry}"
    return None


def _make_has(path: str, target: str) -> bool:
    try:
        with open(os.path.join(path, "Makefile"), encoding="utf-8") as f:
            return any(line.startswith(target + ":") for line in f)
    except OSError:
        return False


def command_for(path: str, step: str) -> str | None:
    """The shell command for `step` in the project at `path`, or None if there isn't one."""
    has = lambda name: os.path.exists(os.path.join(path, name))
    if has("package.json"):
        return _node(path, step)
    if has("pubspec.yaml"):
        return {"dev": "flutter run", "start": "flutter run", "build": "flutter build apk",
                "test": "flutter test", "install": "flutter pub get"}.get(step)
    if has("Cargo.toml"):
        return {"dev": "cargo run", "start": "cargo run", "build": "cargo build --release",
                "test": "cargo test", "install": "cargo fetch"}.get(step)
    if has("go.mod"):
        return {"dev": "go run .", "start": "go run .", "build": "go build ./...",
                "test": "go test ./...", "install": "go mod download"}.get(step)
    if has("pyproject.toml") or has("requirements.txt") or has("setup.py"):
        cmd = _python(path, step)
        if cmd:
            return cmd
    if has("Makefile") and _make_has(path, step):
        return f"make {step}"
    return None


def run(path: str, step: str, executor) -> str:
    """Run one step. -> a result line for the model."""
    folder = os.path.abspath(os.path.expanduser(path or "."))
    if not os.path.isdir(folder):
        return f"there's no folder at {folder}"
    if step not in STEPS:
        return f"unknown project step '{step}' (use one of {', '.join(STEPS)})"
    cmd = command_for(folder, step)
    if not cmd:
        return f"I can't tell how to {step} the project in {folder}"

    if step in _LONG_RUNNING:
        logdir = os.path.join(os.path.expanduser("~/.cortex"), "logs")
        os.makedirs(logdir, exist_ok=True)
        log = os.path.join(logdir, f"project-{os.path.basename(folder)}.log")
        with open(log, "ab") as lf:
            p = subprocess.Popen(shlex.split(cmd), cwd=folder, stdout=lf, stderr=lf, start_new_session=True)
        time.sleep(3)
        if p.poll() is not None:
            return f"ran `{cmd}` -> exit {p.returncode}; it stopped straight away, see {log}"
        return f"ran `{cmd}` -> exit 0; running in the background as process {p.pid}, log at {log}"

    res = executor.execute(command=cmd, cwd=folder, timeout=900, source="cortex-core")
    out = (res.get("stdout") or "").strip()[-2500:]
    err = (res.get("stderr") or "").strip()[-800:]
    return f"ran `{cmd}` in {folder} -> exit {res.get('exit_code')}; {out}\n{err}".strip()
