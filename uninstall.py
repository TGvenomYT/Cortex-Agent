#!/usr/bin/env python3
"""
uninstall.py — cleanly remove Cortex from macOS, Windows or Linux.

    python3 uninstall.py          # macOS / Linux
    py uninstall.py               # Windows

By default it removes everything Cortex *installed* but KEEPS your personal data
(conversations, memory, settings in ~/.cortex) unless you ask it to purge. It
never deletes the project source folder itself or Ollama (your other apps may
use it) unless you opt in.

What it removes:
  • scheduled background jobs   (launchd / cron / Task Scheduler)
  • stored secrets             (Keychain / Credential Manager / Secret Service)
  • the `cx` command-line shim (/usr/local/bin/cx)
  • the Python virtualenv      (.venv)

Flags:
  --purge           ALSO delete your data in ~/.cortex (irreversible)
  --remove-ollama   ALSO uninstall Ollama and its pulled models
  --yes             don't prompt; use the safe defaults (keeps data + Ollama)
  --dry-run         show what would be removed, change nothing
"""
import os
import sys
import glob
import json
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
CORTEX_DIR = os.path.expanduser(os.path.join("~", ".cortex"))
LA = os.path.expanduser(os.path.join("~", "Library", "LaunchAgents"))
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
OSNAME = "macOS" if IS_MAC else "Windows" if IS_WIN else "Linux" if IS_LINUX else sys.platform

ARGS = sys.argv[1:]
DRY = "--dry-run" in ARGS
ASSUME_YES = "--yes" in ARGS


def say(m=""):
    print(m, flush=True)


def step(n, m):
    say(f"\n[{n}] {m}")


def run(cmd, shell=False):
    if DRY:
        say("    would run: " + (cmd if shell else " ".join(cmd)))
        return 0
    return subprocess.run(cmd, shell=shell, capture_output=True, text=True).returncode


def rm_path(p):
    if not os.path.lexists(p):
        return
    if DRY:
        say("    would remove: " + p)
        return
    try:
        if os.path.isdir(p) and not os.path.islink(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            os.remove(p)
        say("    removed " + p)
    except PermissionError:
        say(f"    ⚠ no permission to remove {p} — try: sudo rm -rf '{p}'")
    except OSError as e:
        say(f"    ⚠ could not remove {p}: {e}")


def ask_yes(prompt, default=False):
    if ASSUME_YES or not sys.stdin or not sys.stdin.isatty():
        return default
    d = "Y/n" if default else "y/N"
    try:
        ans = input(f"    {prompt} [{d}] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    return default if not ans else ans in ("y", "yes")


# ── scheduled jobs ────────────────────────────────────────────────────────────
def _email_accounts():
    try:
        with open(os.path.join(CORTEX_DIR, "settings.json")) as f:
            return [a.get("email") for a in json.load(f).get("emails", []) if a.get("email")]
    except (FileNotFoundError, ValueError):
        return []


def _task_ids():
    try:
        with open(os.path.join(CORTEX_DIR, "settings.json")) as f:
            return [t.get("id") for t in json.load(f).get("tasks", []) if t.get("id")]
    except (FileNotFoundError, ValueError):
        return []


def remove_jobs():
    step(1, "Scheduled background jobs")
    if IS_MAC:
        plists = glob.glob(os.path.join(LA, "com.cortex.*.plist")) + \
                 glob.glob(os.path.join(LA, "com.cortex.*.plist.retired"))
        # unload live ones first
        out = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
        for line in out.stdout.splitlines():
            if "com.cortex." in line:
                label = line.split("\t")[-1].strip()
                run(["launchctl", "remove", label])
        for p in plists:
            if not DRY:
                subprocess.run(["launchctl", "unload", p], capture_output=True)
            rm_path(p)
        if not plists:
            say("    no launchd jobs found ✓")
    elif IS_LINUX:
        if not shutil.which("crontab"):
            say("    no crontab — nothing to clean")
            return
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        lines = cur.stdout.splitlines() if cur.returncode == 0 else []
        kept, removed, skip = [], 0, False
        for l in lines:
            if "cortex-task-" in l or "run_once.py" in l and "Cortex" in l:
                removed += 1
                skip = True
                continue
            if skip and l.strip().startswith(os.path.join(HERE, "")):
                skip = False
                removed += 1
                continue
            skip = False
            kept.append(l)
        if removed and not DRY:
            subprocess.run(["crontab", "-"], input="\n".join(kept) + "\n", text=True, capture_output=True)
        say(f"    removed {removed} cron line(s)" if removed else "    no cron jobs found ✓")
    elif IS_WIN:
        ids = _task_ids()
        for tid in ids:
            run(["schtasks", "/delete", "/f", "/tn", f"\\Cortex\\task-{tid}"])
        # best-effort: any remaining \Cortex\ tasks
        q = subprocess.run(["schtasks", "/query", "/fo", "LIST"], capture_output=True, text=True)
        for line in q.stdout.splitlines():
            if line.strip().startswith("TaskName:") and "\\Cortex\\" in line:
                name = line.split("TaskName:", 1)[1].strip()
                run(["schtasks", "/delete", "/f", "/tn", name])
        say(f"    cleared {len(ids)} scheduled task(s)" if ids else "    no scheduled tasks found ✓")


# ── secrets ───────────────────────────────────────────────────────────────────
def _secret_python():
    vp = os.path.join(VENV, "Scripts", "python.exe") if IS_WIN else os.path.join(VENV, "bin", "python")
    return vp if os.path.exists(vp) else sys.executable


def remove_secrets():
    step(2, "Stored secrets")
    accounts = (["cortex-db-password",
                 "apikey:anthropic", "apikey:openai", "apikey:gemini"]
                + [f"mail:{e}" for e in _email_accounts()])
    if DRY:
        for a in accounts:
            say(f"    would delete secret: {a}")
        return
    # Use osutil (same logic as the app) via the venv python so keyring-backed
    # stores on Windows/Linux are handled, and the macOS Keychain on macOS.
    code = ("from cortex.util import osutil;import sys;"
            "[osutil.secret_delete(a) for a in sys.argv[1:]];"
            "print('ok')")
    try:
        r = subprocess.run([_secret_python(), "-c", code, *accounts],
                           cwd=HERE, capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            say(f"    deleted {len(accounts)} secret entry(ies) ✓")
        else:
            raise RuntimeError(r.stderr.strip())
    except Exception as e:
        say(f"    ⚠ couldn't clear secrets automatically ({e}).")
        if IS_MAC:
            say("      manual: security delete-generic-password -s cortex -a cortex-db-password")


# ── cli shim ──────────────────────────────────────────────────────────────────
def remove_cli():
    step(3, "Command-line shim (cx)")
    if IS_WIN:
        say("    nothing to remove (no cx shim is installed on Windows)")
        return
    cx = "/usr/local/bin/cx"
    if not os.path.lexists(cx):
        say("    no cx shim found ✓")
        return
    # only remove if it points into THIS project
    try:
        target = os.path.realpath(cx)
    except OSError:
        target = ""
    if HERE not in target:
        say(f"    {cx} points elsewhere ({target}) — leaving it alone")
        return
    if DRY:
        say(f"    would remove: {cx}")
        return
    try:
        os.remove(cx)
        say(f"    removed {cx}")
    except PermissionError:
        say(f"    ⚠ {cx} is root-owned — run: sudo rm '{cx}'")


# ── venv + data + ollama ──────────────────────────────────────────────────────
def remove_venv():
    step(4, "Python virtualenv")
    if os.path.isdir(VENV):
        rm_path(VENV)
    else:
        say("    no .venv found ✓")


def remove_data():
    step(5, "Your data (~/.cortex)")
    if not os.path.isdir(CORTEX_DIR):
        say("    nothing to remove ✓")
        return
    do = "--purge" in ARGS or ask_yes(
        "Delete ALL Cortex data (conversations, memory, settings)? This is irreversible.",
        default=False)
    if do:
        rm_path(CORTEX_DIR)
    else:
        say("    keeping your data in ~/.cortex (re-run with --purge to delete it)")


def remove_ollama():
    step(6, "Ollama + models (optional)")
    if not shutil.which("ollama"):
        say("    Ollama isn't installed ✓")
        return
    do = "--remove-ollama" in ARGS or ask_yes(
        "Also uninstall Ollama and delete its downloaded models?", default=False)
    if not do:
        say("    leaving Ollama installed (other apps may use it)")
        return
    if IS_MAC and shutil.which("brew"):
        run(["brew", "uninstall", "ollama"])
    elif IS_WIN and shutil.which("winget"):
        run(["winget", "uninstall", "--id", "Ollama.Ollama", "-e", "--silent"])
    elif IS_LINUX:
        say("    removing the Ollama binary + service (may need sudo)…")
        run("sudo systemctl stop ollama; sudo rm -f $(command -v ollama); "
            "sudo rm -rf /usr/share/ollama", shell=True)
    else:
        say("    couldn't auto-uninstall — remove Ollama from your system settings.")
    # model blobs
    om = os.path.expanduser(os.path.join("~", ".ollama"))
    if ask_yes(f"Delete the model cache at {om} (frees disk)?", default=False):
        rm_path(om)


def main():
    say(f"Cortex ✦  uninstall  ({OSNAME})" + ("   [DRY RUN — nothing will change]" if DRY else ""))
    if not ("--yes" in ARGS or DRY):
        say("\nThis will remove Cortex's scheduled jobs, secrets, cx shim and virtualenv.")
        say("Your data in ~/.cortex and Ollama are kept unless you say otherwise.")
        if not ask_yes("Continue?", default=True):
            say("Aborted — nothing changed.")
            return 0
    remove_jobs()
    remove_secrets()
    remove_cli()
    remove_venv()
    remove_data()
    remove_ollama()
    say("\nDone. The project folder itself is left in place — delete it manually if you like:")
    say(f"    {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
