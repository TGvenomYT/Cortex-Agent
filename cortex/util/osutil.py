"""
osutil.py — the one place that knows which OS Cortex is running on.

Cortex is local-first and cross-platform: the chat, GUI and agent loop are pure
Python and run anywhere. Only a handful of touchpoints differ per OS —
notifications, opening a browser, storing secrets, and scheduling background
jobs — so every one of them goes through here. Each helper degrades gracefully
(logs, never crashes) when a platform can't do the thing, so the core app keeps
working on a machine that has no notifier, no keyring daemon, etc.
"""
from __future__ import annotations

import os
import sys
import json
import shutil
import logging
import platform
import subprocess

logger = logging.getLogger("cortex.osutil")

IS_MAC = sys.platform == "darwin"
IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")

OS_NAME = "macOS" if IS_MAC else "Windows" if IS_WINDOWS else "Linux" if IS_LINUX else platform.system()


def data_dir() -> str:
    """Cortex's per-user data dir (~/.cortex on every OS). Created 0700 on POSIX."""
    d = os.path.expanduser(os.path.join("~", ".cortex"))
    try:
        os.makedirs(d, exist_ok=True)
        if not IS_WINDOWS:
            os.chmod(d, 0o700)
    except OSError:
        pass
    return d


# ── opening a browser / URL ────────────────────────────────────────────────
def open_url(url: str) -> bool:
    """Open a URL (or local file) in the user's default browser, cross-platform.
    NB: we avoid stdlib `webbrowser` because the project ships its own
    webbrowser.py which shadows it on import."""
    try:
        if IS_MAC:
            subprocess.Popen(["open", url])
        elif IS_WINDOWS:
            os.startfile(url)  # type: ignore[attr-defined]
        else:
            opener = shutil.which("xdg-open") or shutil.which("gio")
            if opener:
                subprocess.Popen([opener, url],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                logger.info(f"no URL opener found; open manually: {url}")
                return False
        return True
    except Exception as e:
        logger.warning(f"could not open {url}: {e}")
        return False


# ── desktop notifications ──────────────────────────────────────────────────
def notify(title: str, message: str, sound: bool = False) -> bool:
    """Show a desktop notification. macOS: osascript; Linux: notify-send;
    Windows: PowerShell toast. Falls back to a log line if none is available."""
    msg = (message or "")[:400]
    ttl = f"Cortex: {title}" if title else "Cortex"
    try:
        if IS_MAC:
            safe = msg.replace('"', '\\"')
            st = ttl.replace('"', '\\"')
            snd = ' sound name "Purr"' if sound else ""
            subprocess.run(["osascript", "-e",
                            f'display notification "{safe}" with title "{st}"{snd}'],
                           capture_output=True, timeout=5)
            return True
        if IS_LINUX and shutil.which("notify-send"):
            subprocess.run(["notify-send", ttl, msg], capture_output=True, timeout=5)
            return True
        if IS_WINDOWS:
            ps = (
                '[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime] | Out-Null;'
                '[Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom, ContentType=WindowsRuntime] | Out-Null;'
                '$t=[Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02);'
                f'$t.GetElementsByTagName("text").Item(0).AppendChild($t.CreateTextNode({json.dumps(ttl)})) | Out-Null;'
                f'$t.GetElementsByTagName("text").Item(1).AppendChild($t.CreateTextNode({json.dumps(msg)})) | Out-Null;'
                '$n=[Windows.UI.Notifications.ToastNotification]::new($t);'
                '[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("Cortex").Show($n);'
            )
            subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                           capture_output=True, timeout=8)
            return True
    except Exception as e:
        logger.debug(f"notify failed: {e}")
    logger.info(f"[notify] {ttl}: {msg}")
    return False


def speak(text: str, voice: str = None, rate: int = 175) -> bool:
    """Speak text aloud via the OS TTS (best-effort, non-blocking)."""
    try:
        if IS_MAC:
            cmd = ["say", "-r", str(rate)]
            if voice:
                cmd += ["-v", voice]
            subprocess.Popen(cmd + [text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        if IS_LINUX:
            tts = shutil.which("spd-say") or shutil.which("espeak-ng") or shutil.which("espeak")
            if tts:
                subprocess.Popen([tts, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
        if IS_WINDOWS:
            ps = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak({json.dumps(text)})'
            subprocess.Popen(["powershell", "-NoProfile", "-Command", ps],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
    except Exception as e:
        logger.debug(f"speak failed: {e}")
    return False


# ── secrets (cross-platform) ───────────────────────────────────────────────
# macOS -> login Keychain (via `security`). Elsewhere -> the `keyring` library
# if installed (Windows Credential Manager / Linux Secret Service), falling back
# to a 0600 JSON file so the app still works on a headless box with no keyring.
_SERVICE = "cortex"
_SECRETS_FILE = os.path.join(data_dir(), "secrets.json")


def _file_secret_get(account: str) -> str | None:
    try:
        with open(_SECRETS_FILE) as f:
            return json.load(f).get(account)
    except (FileNotFoundError, ValueError):
        return None


def _file_secret_set(account: str, secret: str) -> bool:
    try:
        data = {}
        if os.path.exists(_SECRETS_FILE):
            with open(_SECRETS_FILE) as f:
                data = json.load(f)
        data[account] = secret
        with open(_SECRETS_FILE, "w") as f:
            json.dump(data, f)
        if not IS_WINDOWS:
            os.chmod(_SECRETS_FILE, 0o600)
        return True
    except Exception as e:
        logger.warning(f"file secret store failed: {e}")
        return False


def _file_secret_delete(account: str) -> None:
    try:
        with open(_SECRETS_FILE) as f:
            data = json.load(f)
        data.pop(account, None)
        with open(_SECRETS_FILE, "w") as f:
            json.dump(data, f)
    except (FileNotFoundError, ValueError):
        pass


def secret_get(account: str) -> str | None:
    if IS_MAC:
        try:
            out = subprocess.run(
                ["security", "find-generic-password", "-s", _SERVICE, "-a", account, "-w"],
                capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return out.stdout.strip()
        except Exception:
            pass
        return None
    try:
        import keyring
        v = keyring.get_password(_SERVICE, account)
        if v is not None:
            return v
    except Exception:
        pass
    return _file_secret_get(account)


def secret_set(account: str, secret: str) -> bool:
    if IS_MAC:
        try:
            out = subprocess.run(
                ["security", "add-generic-password", "-U",
                 "-s", _SERVICE, "-a", account, "-w", secret],
                capture_output=True, text=True, timeout=5)
            return out.returncode == 0
        except Exception:
            return False
    try:
        import keyring
        keyring.set_password(_SERVICE, account, secret)
        return True
    except Exception:
        return _file_secret_set(account, secret)


def secret_delete(account: str) -> None:
    if IS_MAC:
        try:
            subprocess.run(["security", "delete-generic-password", "-s", _SERVICE, "-a", account],
                           capture_output=True, timeout=5)
        except Exception:
            pass
        return
    try:
        import keyring
        keyring.delete_password(_SERVICE, account)
        return
    except Exception:
        _file_secret_delete(account)
