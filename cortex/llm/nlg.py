"""
NLG — Text cleanup for TTS.

Kokoro (FastRTC's TTS) handles prosody, pacing, and intonation naturally.
We just strip formatting artifacts and keep clean text.
"""

import re


def to_speech(text: str) -> str:
    """Clean text for TTS. Keep it natural."""
    if not text:
        return ""

    # Code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Markdown
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # URLs
    text = re.sub(r'https?://\S+', '', text)

    # Lists → flowing text
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Table noise
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'─+|═+|—+|---+', '', text)

    # Common abbreviations
    text = re.sub(r'\bk8s\b', 'kubernetes', text, flags=re.IGNORECASE)
    text = re.sub(r'\bOOM\b', 'out of memory', text)
    text = re.sub(r'\b(\d+)%', r'\1 percent', text)

    # Whitespace
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = text.strip().lstrip('.')

    # Trim for voice
    if len(text) > 600:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        trimmed = []
        total = 0
        for s in sentences:
            if total + len(s) > 580:
                break
            trimmed.append(s)
            total += len(s) + 1
        text = ' '.join(trimmed) if trimmed else text[:580].rsplit('.', 1)[0] + '.'

    return text
