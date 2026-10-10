"""
Screen Vision — Captures the screen and uses a local vision model to understand it.
100% local via Ollama (gemma4:e4b). No OpenAI, no API key, ever.

Usage:
  From voice mode: "what's on my screen?" / "explain this" / "what am I looking at?"
  From cx: type "screen" or "look"
  API: POST /vision
"""

import os
import base64
import json
import subprocess
import logging

import httpx

logger = logging.getLogger("cortex.vision")

SCREENSHOT_PATH = "/tmp/cortex-screenshot.png"
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
VISION_MODEL = "gemma4:e4b"  # local, confirmed to support image input


def capture_screen(window_only: bool = False) -> str:
    """
    Capture the screen. Returns path to PNG file.
    window_only=True captures just the frontmost window.
    """
    args = ["screencapture", "-x", "-C", "-t", "png"]
    if window_only:
        args.append("-l")
        # Get frontmost window ID
        wid = _get_front_window_id()
        if wid:
            args.append(str(wid))
        else:
            # Fall back to interactive window capture
            args = ["screencapture", "-x", "-w", "-t", "png"]

    args.append(SCREENSHOT_PATH)

    try:
        subprocess.run(args, capture_output=True, timeout=5)
        if os.path.exists(SCREENSHOT_PATH):
            size = os.path.getsize(SCREENSHOT_PATH)
            logger.info(f"Screenshot captured: {size} bytes")
            return SCREENSHOT_PATH
    except Exception as e:
        logger.error(f"Screenshot failed: {e}")

    return None


def _get_front_window_id() -> int:
    """Get the window ID of the frontmost window via AppleScript."""
    script = '''
    tell application "System Events"
        set frontApp to name of first application process whose frontmost is true
    end tell
    tell application frontApp
        set winId to id of front window
    end tell
    return winId
    '''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def _local_vision_call(image_b64: str, question: str, system_prompt: str, max_tokens: int = 500) -> str:
    """Send an image + question to the local Ollama vision model."""
    payload = {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question, "images": [image_b64]},
        ],
        "stream": False,
    }
    resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def analyze_screen(question: str = None, api_key: str = None) -> str:
    """
    Capture screen + send to local vision model for analysis.
    api_key is accepted and ignored for backward-compat — no key is ever read.
    Returns the explanation text.
    """
    path = capture_screen()
    if not path or not os.path.exists(path):
        return "Couldn't capture the screen."

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    if not question:
        question = "What's on this screen? Describe what you see and what the user is doing."

    system_prompt = (
        "You are Cortex, a developer's AI assistant. The user is showing you their screen. "
        "Describe what you see and help them understand it. Be concise and useful.\n\n"
        "Rules:\n"
        "- Identify the app, website, or tool being shown\n"
        "- Describe the key content/state visible\n"
        "- If the user asked a specific question about it, answer that\n"
        "- If it's a new app/website, explain what it does and how to use the visible UI elements\n"
        "- Keep it to 2-4 sentences for voice mode\n"
        "- Be direct and practical, not descriptive prose"
    )

    try:
        answer = _local_vision_call(image_b64, question, system_prompt)
        logger.info(f"Vision analysis complete (local): {len(answer)} chars")

        try:
            os.remove(path)
        except Exception:
            pass

        return answer

    except Exception as e:
        logger.error(f"Local vision call failed: {e}")
        return f"Vision analysis failed: {str(e)[:100]}"


def analyze_screen_for_voice(question: str = None, api_key: str = None) -> str:
    """Same as analyze_screen but with voice-optimized prompt."""
    path = capture_screen()
    if not path or not os.path.exists(path):
        return "Couldn't capture the screen."

    with open(path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    if not question:
        question = "What am I looking at? Explain briefly."

    system_prompt = (
        "You are Cortex, speaking to Niranjan through a speaker. He's showing you his screen. "
        "Tell him what you see in 1-3 spoken sentences. Talk like a human on a call. "
        "No formatting, no lists, no markdown. Just say what's on screen and what matters. "
        "If it's an app or website he seems unfamiliar with, tell him what it does and what he can do next."
    )

    try:
        answer = _local_vision_call(image_b64, question, system_prompt, max_tokens=200)

        try:
            os.remove(path)
        except Exception:
            pass

        return answer

    except Exception as e:
        logger.error(f"Local vision call failed: {e}")
        return "Couldn't analyze the screen right now."
