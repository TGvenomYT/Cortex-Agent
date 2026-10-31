#!/usr/bin/env python3
"""
Cortex Voice Agent — Full FastRTC pipeline. 100% local. No OpenAI, no API key, ever.

STT: Google Speech Recognition (through daemon /stt)
TTS: macOS `say` (through daemon /tts — fully local, zero network)
VAD: FastRTC's ReplyOnPause (automatic voice detection + turn-taking)

Launches on localhost:7860 — open in browser, click mic, talk.
"""

import json
import os
import sys
import re
import time
import urllib.request
import urllib.error
import subprocess
import numpy as np

from fastrtc import ReplyOnPause, Stream

BASE_URL = "http://localhost:7800"
SESSION_ID = "voice"

print("  Ready. (STT + TTS via daemon)")


# ─── Cortex API ───

def transcribe_via_daemon(audio: tuple) -> str:
    """
    Send audio to daemon's /stt endpoint (Google Speech Recognition).
    audio = (sample_rate, numpy_array)
    Returns transcribed text.
    """
    sample_rate, audio_array = audio

    # Ensure 1D
    audio_array = np.asarray(audio_array).squeeze()
    if audio_array.ndim > 1:
        audio_array = audio_array.reshape(-1)

    # Convert to int16
    if audio_array.dtype in (np.float32, np.float64):
        peak = np.abs(audio_array).max()
        if peak > 0:
            audio_array = (audio_array / peak * 32767).astype(np.int16)
        else:
            audio_array = np.zeros_like(audio_array, dtype=np.int16)
    elif audio_array.dtype != np.int16:
        audio_array = audio_array.astype(np.int16)

    pcm_bytes = audio_array.tobytes()

    try:
        req = urllib.request.Request(
            f"{BASE_URL}/stt", data=pcm_bytes,
            headers={
                "Content-Type": "audio/pcm",
                "X-Sample-Rate": str(sample_rate),
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read())
        text = body.get("text", "").strip()
        if text:
            print(f"  [STT] Got: '{text}'")
        return text
    except Exception as e:
        print(f"  STT error: {e}")
        return ""


def local_tts_chunks(text: str):
    """
    Get TTS audio from the Cortex daemon's /tts endpoint (macOS `say`, fully local).
    Returns (sample_rate, numpy_array) chunks.
    """
    try:
        data = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/tts", data=data,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                print(f"  TTS error: HTTP {resp.status}")
                return
            pcm_data = resp.read()

        if not pcm_data or len(pcm_data) < 100:
            print(f"  TTS error: Empty response ({len(pcm_data)} bytes)")
            return

        audio_array = np.frombuffer(pcm_data, dtype=np.int16)

        # Yield in chunks of 4800 samples (200ms at 24kHz)
        chunk_size = 4800
        for i in range(0, len(audio_array), chunk_size):
            chunk = audio_array[i:i + chunk_size]
            if len(chunk) > 0:
                yield (24000, chunk)

    except Exception as e:
        print(f"  TTS error: {e}")
        return


def send_to_cortex(text: str) -> str:
    """Send query to Cortex daemon."""
    try:
        data = json.dumps({
            "text": text, "source": "siri", "session_id": SESSION_ID
        }).encode()
        req = urllib.request.Request(
            f"{BASE_URL}/ask", data=data,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        return body.get("speak", "No response.")
    except urllib.error.URLError:
        return "Cortex daemon isn't running."
    except Exception:
        return "Something went wrong."


# ─── NLG cleanup ───

def clean_for_speech(text: str) -> str:
    """Strip formatting for TTS."""
    if not text:
        return ""
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'^\s*[-•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'─+|═+|—+|---+', '', text)
    text = re.sub(r'\bk8s\b', 'kubernetes', text, flags=re.IGNORECASE)
    text = re.sub(r'\bOOM\b', 'out of memory', text)
    text = re.sub(r'\b(\d+)%', r'\1 percent', text)
    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.\s*\.', '.', text)
    text = text.strip().lstrip('.')
    if len(text) > 600:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        trimmed, total = [], 0
        for s in sentences:
            if total + len(s) > 580:
                break
            trimmed.append(s)
            total += len(s) + 1
        text = ' '.join(trimmed) if trimmed else text[:580].rsplit('.', 1)[0] + '.'
    return text


# ─── 2FA ───

_pending_2fa = None


def check_2fa(text: str) -> str:
    """Handle 2FA code if spoken."""
    global _pending_2fa
    code_match = re.search(r'\b(\d{4,8})\b', text)
    if not code_match:
        return None
    code = code_match.group(1)
    if _pending_2fa or any(w in text.lower() for w in ["code", "otp", "verification"]):
        site = (_pending_2fa or {}).get("site", "the site")
        _pending_2fa = None
        return send_to_cortex(f"The verification code for {site} is {code}. Continue the login.")
    return None


# ─── Core handler — FastRTC calls this when you stop speaking ───

def cortex_handler(audio: tuple[int, np.ndarray]):
    """
    Called by FastRTC's ReplyOnPause when user finishes a phrase.
    audio = (sample_rate, numpy_array)
    Yields (sample_rate, audio_chunk) tuples back as streamed TTS.
    """
    sample_rate, audio_data = audio

    # FastRTC delivers audio as 2D array (channels, samples) — flatten to 1D
    audio_data = np.asarray(audio_data).squeeze()
    if audio_data.ndim > 1:
        audio_data = audio_data.reshape(-1)

    num_samples = audio_data.shape[0] if audio_data.ndim >= 1 else 0
    duration = num_samples / sample_rate
    print(f"  [DEBUG] Audio: {sample_rate}Hz, {num_samples} samples, {duration:.2f}s, dtype={audio_data.dtype}")

    # Skip very short audio (likely just noise/click)
    if duration < 0.3:
        return

    # Transcribe (pass the flattened audio)
    text = transcribe_via_daemon((sample_rate, audio_data))
    print(f"  [DEBUG] STT returned: '{text}'")
    if not text or not text.strip():
        return

    text = text.strip()
    print(f"  → {text}")

    # Exit
    if text.lower().rstrip('.') in ("goodbye", "exit", "quit", "stop", "bye"):
        for chunk in local_tts_chunks("Later."):
            yield chunk
        return

    # 2FA
    tfa = check_2fa(text)
    if tfa:
        response_text = clean_for_speech(tfa)
        print(f"  ← {response_text[:100]}")
        for chunk in local_tts_chunks(response_text):
            yield chunk
        return

    # Send to Cortex — execute, browse, crawl, whatever
    response = send_to_cortex(text)

    # 2FA detection
    if any(w in response.lower() for w in ["verification code", "2fa", "enter the code", "otp"]):
        global _pending_2fa
        _pending_2fa = {"site": text.split()[0], "asked_at": time.time()}

    response_text = clean_for_speech(response)
    print(f"  ← {response_text[:120]}{'...' if len(response_text) > 120 else ''}")

    # Stream TTS response back (local)
    for chunk in local_tts_chunks(response_text):
        yield chunk


# ─── Entry point ───

def voice_loop():
    """Launch FastRTC voice stream with browser UI."""
    # Host/port/share are overridable via env so the voice UI can be reached
    # from another device (e.g. iPhone over Tailscale). Defaults stay loopback
    # so nothing is exposed unless you explicitly opt in.
    #   CORTEX_VOICE_HOST=0.0.0.0   -> reachable on LAN / Tailscale interfaces
    #   CORTEX_VOICE_PORT=7860
    #   CORTEX_VOICE_SHARE=1        -> Gradio public HTTPS tunnel (no auth!)
    voice_host = os.environ.get("CORTEX_VOICE_HOST", "127.0.0.1")
    voice_port = int(os.environ.get("CORTEX_VOICE_PORT", "7860"))
    voice_share = os.environ.get("CORTEX_VOICE_SHARE", "").lower() in ("1", "true", "yes")

    print("\n  Cortex Voice (FastRTC, 100% local)")
    print(f"  Listening on {voice_host}:{voice_port} — click the mic button ONCE, then just talk.")
    if voice_host != "127.0.0.1":
        print("  NOTE: mobile browsers need HTTPS to grant mic access — front this with `tailscale serve` or a tunnel.\n")
    else:
        print()

    from fastrtc.reply_on_pause import AlgoOptions

    stream = Stream(
        ReplyOnPause(
            cortex_handler,
            algo_options=AlgoOptions(
                audio_chunk_duration=0.6,
                started_talking_threshold=0.2,   # Default — needs 0.2s speech to start
                speech_threshold=0.2,            # Higher = waits longer before deciding you stopped (fewer cutoffs)
            ),
            can_interrupt=False,  # Don't cut off while responding
        ),
        modality="audio",
        mode="send-receive",
    )

    # Auto-open browser on this Mac (always localhost for the local browser)
    subprocess.Popen(
        ["open", f"http://localhost:{voice_port}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    # Launch Gradio UI
    stream.ui.launch(
        server_name=voice_host,
        server_port=voice_port,
        share=voice_share,
    )


if __name__ == "__main__":
    voice_loop()
