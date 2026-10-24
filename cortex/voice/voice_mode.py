#!/usr/bin/env python3
"""
voice_mode.py — Cortex standalone voice mode, MLX-accelerated end to end.

Mic → MLX-Whisper (STT, :8005) → Cortex agent (local, completes the job) →
Kokoro-MLX (TTS, :8005) → speaker. All on-device, all MLX-boosted. Reuses the
MLX voice server POC (voice_server.py) from the livekit Jarvis project.

Run:  python3 voice_mode.py        (or: cx voice)
Say "stop" / "bye" / Ctrl+C to quit.
"""
from __future__ import annotations

import os
import io
import sys
import time
import json
import subprocess

os.chdir(__import__("cortex").ROOT)

import numpy as np
import httpx
import soundfile as sf

VOICE_PORT = int(os.environ.get("CORTEX_VOICE_PORT", "8005"))
VOICE_URL = f"http://127.0.0.1:{VOICE_PORT}"
STT_RATE = 16000              # whisper wants 16 kHz
VOICE = os.environ.get("CORTEX_TTS_VOICE", "bm_daniel")   # warm male; af_heart etc. also work

# Energy-based endpointing (no extra deps): start on speech, stop after silence.
_FRAME = 0.03                 # 30 ms frames
_SILENCE_HANG = 0.9           # stop after this much trailing silence
_MAX_UTT = 15.0               # hard cap per utterance
_START_ENERGY = 0.012         # RMS threshold to call it speech
_MIN_SPEECH = 0.25            # ignore blips shorter than this


# ── MLX voice server lifecycle ─────────────────────────────────────────────
def _server_up() -> bool:
    try:
        httpx.get(VOICE_URL + "/health", timeout=2)
        return True
    except Exception:
        return False


def ensure_server() -> subprocess.Popen | None:
    if _server_up():
        return None
    print("  Starting the MLX voice server (first run loads models — a few seconds)…")
    env = dict(os.environ, CORTEX_VOICE_PORT=str(VOICE_PORT))
    proc = subprocess.Popen([sys.executable, "voice_server.py"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(120):
        if _server_up():
            print("  Voice server ready.")
            return proc
        time.sleep(1)
    print("  Voice server didn't come up in time.")
    return proc


# ── mic capture with energy VAD ────────────────────────────────────────────
def record_utterance() -> np.ndarray | None:
    import sounddevice as sd
    frame_len = int(STT_RATE * _FRAME)
    buf, voiced, silence, speaking = [], [], 0.0, False
    started = time.time()
    with sd.InputStream(samplerate=STT_RATE, channels=1, dtype="float32", blocksize=frame_len) as stream:
        while True:
            block, _ = stream.read(frame_len)
            mono = block[:, 0]
            rms = float(np.sqrt(np.mean(mono * mono)) + 1e-9)
            if not speaking:
                if rms >= _START_ENERGY:
                    speaking = True
                    voiced = [mono]
                # else: keep waiting (idle)
            else:
                voiced.append(mono)
                silence = silence + _FRAME if rms < _START_ENERGY else 0.0
                if silence >= _SILENCE_HANG or (time.time() - started) > _MAX_UTT:
                    break
            if not speaking and (time.time() - started) > 600:
                return None  # 10 min of pure silence → give up
    audio = np.concatenate(voiced) if voiced else np.zeros(0, dtype="float32")
    if len(audio) / STT_RATE < _MIN_SPEECH:
        return None
    return audio


# ── STT / TTS via the MLX server ───────────────────────────────────────────
def transcribe(audio: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, audio, STT_RATE, format="wav")
    buf.seek(0)
    r = httpx.post(VOICE_URL + "/v1/audio/transcriptions",
                   files={"file": ("speech.wav", buf, "audio/wav")},
                   data={"model": "whisper-1", "language": "en"}, timeout=60)
    r.raise_for_status()
    return (r.json().get("text") or "").strip()


def speak(text: str):
    import sounddevice as sd
    r = httpx.post(VOICE_URL + "/v1/audio/speech",
                   json={"model": "kokoro", "input": text, "voice": VOICE, "response_format": "wav"},
                   timeout=120)
    r.raise_for_status()
    data, rate = sf.read(io.BytesIO(r.content), dtype="float32")
    sd.play(data, rate)
    sd.wait()


def _clean_for_speech(text: str) -> str:
    import re
    text = re.sub(r"```[\s\S]*?```", " code block ", text)
    text = re.sub(r"[#*`>_|]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# ── the loop ───────────────────────────────────────────────────────────────
def run_voice_mode():
    from cortex.core.cortex_core import CortexCore
    from cortex.core.context_reader import ContextReader

    proc = ensure_server()
    if not _server_up():
        print("  Could not reach the MLX voice server. Is Ollama + the venv set up?")
        return

    print("\n  🎙  Cortex voice mode (MLX). I'm listening — say \"stop\" to quit.\n")
    core = CortexCore()
    owner = ContextReader.owner_name()
    try:
        speak(f"Hey {owner}, I'm here. What do you need?")
    except Exception as e:
        print(f"  (TTS warmup failed: {e})")

    try:
        while True:
            print("  🎧 listening…", end="\r")
            audio = record_utterance()
            if audio is None:
                continue
            try:
                text = transcribe(audio)
            except Exception as e:
                print(f"  STT error: {e}")
                continue
            if not text:
                continue
            print(f"  you: {text}            ")
            low = text.lower().strip(" .!?")
            if low in ("stop", "bye", "goodbye", "quit", "exit", "that's all", "thats all"):
                speak("Alright, talk soon.")
                break
            print("  🤔 thinking…", end="\r")
            try:
                reply = core.ask(text, session_id="voice", source="siri").get("speak", "")
            except Exception as e:
                reply = f"Sorry, I hit a snag: {e}"
            print(f"  cortex: {reply}\n")
            try:
                speak(_clean_for_speech(reply) or "Done.")
            except Exception as e:
                print(f"  TTS error: {e}")
    except KeyboardInterrupt:
        print("\n  bye.")
    finally:
        if proc is not None:
            proc.terminate()


if __name__ == "__main__":
    run_voice_mode()
