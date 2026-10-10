"""
Notification helpers — macOS notification + TTS, shared by every background
loop (reminders, automation, autonomy) so there's one place that owns how
Cortex speaks up unprompted.
"""

import subprocess
import logging

logger = logging.getLogger("cortex.notify")


def notify(title: str, message: str, sound: str = "Purr", subtitle: str = None):
    """Send a macOS notification via osascript."""
    safe = message[:200].replace('"', '\\"').replace("'", "\\'")
    full_title = f"Cortex: {title}".replace('"', '\\"') if title else "Cortex"
    subtitle_clause = f' subtitle "{subtitle}"' if subtitle else ""
    script = f'display notification "{safe}" with title "{full_title}"{subtitle_clause} sound name "{sound}"'
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception as e:
        logger.error(f"Notification failed: {e}")


def say(text: str, voice: str = "Tara", rate: int = 175):
    """Speak text aloud via macOS `say`, run through the NLG pipeline for naturalness."""
    try:
        from nlg import to_speech
        text = to_speech(text)
    except Exception:
        pass
    try:
        subprocess.Popen(
            ["say", "-v", voice, "-r", str(rate), text],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        logger.error(f"TTS failed: {e}")
