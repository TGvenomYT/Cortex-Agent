"""
Notification helpers — desktop notification + TTS, shared by every background
loop (reminders, automation, autonomy) so there's one place that owns how
Cortex speaks up unprompted. Cross-platform via osutil (macOS / Linux / Windows).
"""

import logging
import cortex.util.osutil as osutil

logger = logging.getLogger("cortex.notify")


def notify(title: str, message: str, sound: str = "Purr", subtitle: str = None):
    """Send a desktop notification on whatever OS we're running."""
    msg = message if not subtitle else f"{subtitle}\n{message}"
    osutil.notify(title, msg, sound=bool(sound))


def say(text: str, voice: str = "Tara", rate: int = 175):
    """Speak text aloud via the OS TTS, run through the NLG pipeline for naturalness."""
    try:
        from cortex.llm.nlg import to_speech
        text = to_speech(text)
    except Exception:
        pass
    # macOS has the named voices (Tara etc.); other OSes use their default voice.
    osutil.speak(text, voice=voice if osutil.IS_MAC else None, rate=rate)
