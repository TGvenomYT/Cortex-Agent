"""
lifecycle.py — start / stop / probe the on-device voice engine.

The engine (mlx-whisper speech-to-text + kokoro-mlx text-to-speech, see voice_server.py)
holds ~2–3 GB of models once it loads. It used to be started on the first mic tap and then
live forever. Now it is started on demand and a small reaper thread shuts it down after
`resources.voice_idle_minutes` of silence (default 5), so a quiet Cortex holds no models.

Stdlib + httpx only — no numpy/mlx imports — so the GUI and the Workers panel can manage
the engine without paying for it.
"""
from __future__ import annotations

import os
import sys
import time
import signal
import threading
import subprocess
import logging

import httpx
import cortex

logger = logging.getLogger("cortex.voice")

PORT = int(os.environ.get("CORTEX_VOICE_PORT", "8005"))
URL = f"http://127.0.0.1:{PORT}"
_OURS = ("cortex.voice.voice_server", "voice_server.py")   # never touch someone else's :8005 server

_proc: subprocess.Popen | None = None
_last_used = time.time()
_reaper_started = False


def is_up() -> bool:
    try:
        return httpx.get(URL + "/health", timeout=1.5).status_code < 500
    except Exception:
        return False


def touch() -> None:
    """Record that the engine was just used (resets the idle clock)."""
    global _last_used
    _last_used = time.time()


def idle_seconds() -> float:
    return time.time() - _last_used


def start(timeout: float = 120.0) -> bool:
    """Make sure the engine is up. Fails fast if the process dies instead of polling blindly."""
    global _proc
    if is_up():
        touch()
        return True
    env = dict(os.environ, CORTEX_VOICE_PORT=str(PORT))
    logdir = os.path.expanduser("~/.cortex/logs")
    os.makedirs(logdir, exist_ok=True)
    log = open(os.path.join(logdir, "voice-server.log"), "ab")
    _proc = subprocess.Popen([sys.executable, "-m", "cortex.voice.voice_server"], cwd=cortex.ROOT,
                             env=env, stdout=log, stderr=log, start_new_session=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _proc.poll() is not None:
            logger.warning(f"voice server exited early (code {_proc.returncode}) — see ~/.cortex/logs/voice-server.log")
            return False
        if is_up():
            touch()
            _ensure_reaper()
            return True
        time.sleep(0.5)
    return False


def _listening_pids() -> list[int]:
    try:
        out = subprocess.run(["lsof", "-ti", f"tcp:{PORT}", "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=5).stdout.split()
        return [int(p) for p in out if p.isdigit()]
    except Exception:
        return []


def _is_ours(pid: int) -> bool:
    try:
        cmd = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                             capture_output=True, text=True, timeout=5).stdout
        return any(tag in cmd for tag in _OURS)
    except Exception:
        return False


def stop() -> bool:
    """Shut the engine down and free its models. Only ever kills a Cortex voice server."""
    global _proc
    killed = False
    pids = {p for p in _listening_pids() if _is_ours(p)}
    if _proc is not None and _proc.poll() is None:
        pids.add(_proc.pid)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            killed = True
        except OSError:
            pass
    for _ in range(20):
        if not is_up():
            break
        time.sleep(0.2)
    for pid in pids:                      # still alive? force it
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    _proc = None
    return killed


def _idle_limit_seconds() -> float | None:
    try:
        import cortex.storage.settings_store as settings_store
        m = float(settings_store.section("resources").get("voice_idle_minutes", 5))
        return m * 60 if m > 0 else None      # 0 = never auto-stop
    except Exception:
        return 300.0


def _reap_loop() -> None:
    while True:
        time.sleep(30)
        try:
            limit = _idle_limit_seconds()
            if limit and is_up() and idle_seconds() > limit and any(_is_ours(p) for p in _listening_pids()):
                logger.info(f"voice engine idle {int(idle_seconds())}s — shutting it down to free memory")
                stop()
        except Exception as e:
            logger.debug(f"voice reaper: {e}")


def _ensure_reaper() -> None:
    global _reaper_started
    if not _reaper_started:
        _reaper_started = True
        threading.Thread(target=_reap_loop, name="voice-reaper", daemon=True).start()
