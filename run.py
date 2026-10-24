#!/usr/bin/env python3
"""
run.py — dynamic one-command installer + launcher for Cortex.

    python3 run.py          # macOS / Linux
    py run.py               # Windows

It figures out what your machine is missing and sets it up, in order:
  1. Python virtualenv (.venv) + Python requirements
  2. Ollama (the local LLM engine) — installs it if absent
  3. A usable model — pulls one if the configured model isn't available
  4. Launches the Cortex GUI, whose first run walks you through onboarding.

Everything is incremental: anything already present is left untouched, so
re-running is safe and running it on a machine that's already set up just
launches the app.

Flags:
    --no-venv       use the current interpreter instead of creating .venv
    --reinstall     force-reinstall the Python requirements
    --voice         also install the optional voice/audio extras
    --skip-ollama   don't touch Ollama or models (just venv + launch)
    --yes           don't prompt; accept the recommended defaults
"""
import os
import sys
import json
import shutil
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
OSNAME = "macOS" if IS_MAC else "Windows" if IS_WIN else "Linux" if IS_LINUX else sys.platform
STAMP = os.path.join(VENV, ".cortex-deps-installed")
OLLAMA_API = "http://127.0.0.1:11434/api/tags"
DEFAULT_MODEL = "gemma3:12b"        # solid tool-use default, ~8GB
LIGHT_MODEL = "gemma3:4b"           # lighter alt for low-RAM machines (~3GB)
EMBED_MODEL = "nomic-embed-text"    # required for PDF/doc search (RAG), ~275MB

ARGS = sys.argv[1:]
ASSUME_YES = "--yes" in ARGS


def say(msg=""):
    print(msg, flush=True)


def step(n, msg):
    say(f"\n[{n}] {msg}")


def sh(cmd, **kw):
    say("    $ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    subprocess.check_call(cmd, **kw)


def sh_optional(cmd, label):
    """Run an install that's allowed to fail (optional feature). Returns True/False."""
    say("    $ " + " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError:
        say(f"    ⚠ couldn't install {label} on this Python — Cortex still runs; "
            f"that feature stays off until the wheels are available.")
        return False


def ask_yes(prompt, default=True):
    if ASSUME_YES or not sys.stdin or not sys.stdin.isatty():
        return default
    d = "Y/n" if default else "y/N"
    try:
        ans = input(f"    {prompt} [{d}] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    if not ans:
        return default
    return ans in ("y", "yes")


def _read_reqs(path):
    out = []
    for line in open(path):
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(line)
    return out


def install_requirements(py, path):
    """Install core deps. Try the fast batch install; if it fails (e.g. one package
    has no wheel for this Python), fall back to installing each one individually so a
    single bad package can't block the whole app."""
    try:
        sh([py, "-m", "pip", "install", "-q", "-r", path])
        return
    except subprocess.CalledProcessError:
        say("    batch install failed — retrying package-by-package so the rest still land…")
    failed = []
    for pkg in _read_reqs(path):
        if not sh_optional([py, "-m", "pip", "install", "-q", pkg], pkg):
            failed.append(pkg)
    if failed:
        say(f"    ⚠ couldn't install: {', '.join(failed)}. Cortex will run with those "
            f"features degraded. Tip: a stable Python (3.12 / 3.13) has wheels for everything.")


# ── 1. venv + python deps ────────────────────────────────────────────────────
def venv_python():
    return os.path.join(VENV, "Scripts", "python.exe") if IS_WIN \
        else os.path.join(VENV, "bin", "python")


def ensure_python_env():
    step(1, "Python environment")
    py = sys.executable
    if "--no-venv" not in ARGS:
        if not os.path.isdir(VENV):
            say("    creating virtualenv (.venv)…")
            sh([sys.executable, "-m", "venv", VENV])
        py = venv_python()
    if "--reinstall" in ARGS or not os.path.exists(STAMP):
        say("    installing core requirements (first run can take a minute)…")
        sh([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
        install_requirements(py, os.path.join(HERE, "requirements.txt"))
        # Optional extras — best-effort. A missing wheel (e.g. matplotlib on a
        # brand-new ARM Python) must NOT fail the whole install.
        say("    installing chart support (optional)…")
        sh_optional([py, "-m", "pip", "install", "-q", "-r",
                     os.path.join(HERE, "requirements-charts.txt")], "chart support (matplotlib)")
        if "--voice" in ARGS:
            sh_optional([py, "-m", "pip", "install", "-q", "-r",
                         os.path.join(HERE, "requirements-voice.txt")], "voice support")
        open(STAMP, "w").close()
    else:
        say("    dependencies already installed ✓")
    return py


# ── 2. Ollama runtime ────────────────────────────────────────────────────────
def ollama_up():
    try:
        with urllib.request.urlopen(OLLAMA_API, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def install_ollama():
    say("    Ollama not found — installing it…")
    try:
        if IS_MAC:
            if shutil.which("brew"):
                sh(["brew", "install", "ollama"])
            else:
                say("    Homebrew isn't installed. Download Ollama from https://ollama.com/download")
                _open("https://ollama.com/download")
                return False
        elif IS_LINUX:
            if shutil.which("curl"):
                say("    running the official installer (may ask for sudo)…")
                sh("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
            else:
                say("    Install curl, or get Ollama from https://ollama.com/download")
                return False
        elif IS_WIN:
            if shutil.which("winget"):
                sh(["winget", "install", "--id", "Ollama.Ollama", "-e", "--silent",
                    "--accept-package-agreements", "--accept-source-agreements"])
            else:
                say("    Download the Ollama installer from https://ollama.com/download")
                _open("https://ollama.com/download")
                return False
    except subprocess.CalledProcessError as e:
        say(f"    automatic install failed ({e}). Get it from https://ollama.com/download")
        return False
    return shutil.which("ollama") is not None


def ensure_ollama():
    step(2, "Ollama (local LLM engine)")
    if not shutil.which("ollama"):
        if not ask_yes("Install Ollama now?", default=True):
            say("    skipping — install it yourself from https://ollama.com, then re-run.")
            return False
        if not install_ollama():
            return False
    else:
        say("    Ollama is installed ✓")
    if not ollama_up():
        say("    starting the Ollama service…")
        try:
            subprocess.Popen(["ollama", "serve"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            say(f"    couldn't auto-start it ({e}); start it manually with: ollama serve")
        # wait briefly for it to come up
        import time
        for _ in range(20):
            if ollama_up():
                break
            time.sleep(0.5)
    say("    Ollama service reachable ✓" if ollama_up() else "    ⚠ Ollama service not reachable yet")
    return True


# ── 3. a usable model ────────────────────────────────────────────────────────
def configured_models(py):
    """Read primary+fallback model names from config.yaml (uses the venv's yaml)."""
    code = ("import yaml;c=yaml.safe_load(open(r'%s'));"
            "o=c.get('llm',c);"
            "import json;print(json.dumps([(o.get('ollama') or {}).get('model'),"
            "(o.get('ollama_fallback') or {}).get('model')]))"
            % os.path.join(HERE, "config.yaml"))
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=15)
        prim, fb = json.loads(out.stdout.strip() or "[null,null]")
        return prim, fb
    except Exception:
        return None, None


def installed_models():
    try:
        with urllib.request.urlopen(OLLAMA_API, timeout=5) as r:
            data = json.load(r)
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def _pullable(model):
    """MLX / obviously-custom tags aren't in the public registry."""
    if not model:
        return False
    m = model.lower()
    return "mlx" not in m and "agent" not in m and "custom" not in m


def set_config_model(py, new_model, old_primary, old_fallback):
    """Point config.yaml at new_model, preserving comments (targeted text swap)."""
    path = os.path.join(HERE, "config.yaml")
    txt = open(path, encoding="utf-8").read()
    for old in (old_primary, old_fallback):
        if old and old in txt:
            txt = txt.replace(f'"{old}"', f'"{new_model}"').replace(f"'{old}'", f'"{new_model}"')
    open(path, "w", encoding="utf-8").write(txt)
    say(f"    config.yaml now uses: {new_model}")


def ensure_model(py):
    step(3, "Language model")
    have = installed_models()
    prim, fb = configured_models(py)
    if prim and any(prim == h or h.startswith(prim + ":") or h == prim + ":latest" for h in have):
        say(f"    configured model '{prim}' is already installed ✓")
        return True

    # Decide what to pull. Keep the configured one only if it's actually pullable
    # on this OS; otherwise fall back to a registry default.
    target = prim if _pullable(prim) else DEFAULT_MODEL
    if prim and not _pullable(prim):
        say(f"    configured model '{prim}' can't be pulled here (custom/MLX-only).")

    if have:
        say(f"    models already present: {', '.join(have)}")
        if not any(_pullable(prim) and prim == h.split(':')[0] for h in have):
            # offer to just use an existing one instead of downloading
            if not ASSUME_YES and sys.stdin and sys.stdin.isatty():
                pick = input(f"    pull '{target}' (~several GB), or type an installed name to use it "
                             f"[{target}]: ").strip()
                if pick:
                    target = pick

    if target in have:
        set_config_model(py, target, prim, fb)
        return True

    if not ask_yes(f"Download model '{target}' now?", default=True):
        alt = LIGHT_MODEL
        if ask_yes(f"Download the lighter '{alt}' instead?", default=True):
            target = alt
        else:
            say("    skipping model download — set one up with `ollama pull <model>` then edit config.yaml.")
            return False
    try:
        sh(["ollama", "pull", target])
    except subprocess.CalledProcessError as e:
        say(f"    pull failed ({e}).")
        return False
    set_config_model(py, target, prim, fb)
    return True


def ensure_embed_model():
    """Pull the embedding model used for PDF/document search (RAG). Without it,
    uploads appear to succeed but can never be retrieved."""
    step("3b", "Embedding model (for PDF / document search)")
    have = installed_models()
    if any(h == EMBED_MODEL or h.startswith(EMBED_MODEL + ":") for h in have):
        say(f"    '{EMBED_MODEL}' already installed ✓")
        return True
    say(f"    pulling '{EMBED_MODEL}' (~275MB) so attached files are searchable…")
    try:
        sh(["ollama", "pull", EMBED_MODEL])
        return True
    except subprocess.CalledProcessError as e:
        say(f"    ⚠ couldn't pull {EMBED_MODEL} ({e}) — PDF search won't work until you run: "
            f"ollama pull {EMBED_MODEL}")
        return False


# ── 4. launch ────────────────────────────────────────────────────────────────
def _open(url):
    try:
        if IS_MAC:
            subprocess.Popen(["open", url])
        elif IS_WIN:
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def launch(py):
    step(4, "Launching Cortex")
    say("    opening the GUI at http://127.0.0.1:7860  (first run will ask your name)")
    say("    Ctrl+C to stop.\n")
    env = dict(os.environ, CORTEX_GUI_OPEN="1")  # the installer opens a tab once; `cx gui` won't
    try:
        return subprocess.call([py, "-m", "cortex.server.gui_server"], cwd=HERE, env=env)
    except KeyboardInterrupt:
        return 0


def main():
    say(f"Cortex ✦  setting up on {OSNAME}")
    if sys.version_info < (3, 10):
        say(f"Cortex needs Python 3.10+ (you have {sys.version.split()[0]}).")
        return 1
    py = ensure_python_env()
    if "--skip-ollama" not in ARGS:
        if ensure_ollama():
            ensure_model(py)
            ensure_embed_model()
        else:
            say("\n    (continuing without Ollama — chat won't work until it's installed)")
    return launch(py)


if __name__ == "__main__":
    sys.exit(main())
