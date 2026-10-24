import os
import numpy as np
import soundfile as sf
import io
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import asyncio
import tempfile

app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "stt": "mlx-whisper", "tts": "kokoro-mlx"}

# ── MLX Whisper (STT) ─────────────────────────────────────────────────────
print("Initializing STT (mlx-whisper)...")
import mlx_whisper

WHISPER_MODEL = "mlx-community/whisper-small-mlx"

# ── MLX Kokoro (TTS) ──────────────────────────────────────────────────────
print("Initializing TTS (kokoro-mlx)...")
from kokoro_mlx import KokoroTTS

kokoro_model = KokoroTTS.from_pretrained()
KOKORO_SAMPLE_RATE = KokoroTTS.SAMPLE_RATE  # 24000

# ── STT vocabulary biasing ────────────────────────────────────────────────
HOTWORDS_FILE = os.path.join(__import__("cortex").ROOT, "hotwords.txt")
_hotwords_cache = {"mtime": 0.0, "value": None}


def get_hotwords():
    try:
        mtime = os.path.getmtime(HOTWORDS_FILE)
        if mtime != _hotwords_cache["mtime"]:
            with open(HOTWORDS_FILE, "r", encoding="utf-8") as f:
                text = f.read().strip()
            _hotwords_cache["value"] = text or None
            _hotwords_cache["mtime"] = mtime
    except OSError:
        return None
    return _hotwords_cache["value"]


@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str = Form("en"),
):
    content = await file.read()

    def _transcribe() -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            tmp.write(content)
            tmp.flush()
            result = mlx_whisper.transcribe(
                tmp.name,
                path_or_hf_repo=WHISPER_MODEL,
                language=language or "en",
                word_timestamps=False,
                condition_on_previous_text=False,
            )
        return result["text"].strip()

    text = await asyncio.to_thread(_transcribe)
    return JSONResponse(content={"text": text})


class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "af_heart"
    response_format: str = "wav"


@app.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    voice = request.voice
    valid_prefixes = ("af_", "am_", "bf_", "bm_", "ef_", "em_")
    if not any(voice.startswith(p) for p in valid_prefixes):
        voice = "bm_daniel"

    print(f"TTS Request: {request.input} with voice {voice}, format {request.response_format}")

    if request.response_format == "pcm":
        async def pcm_stream():
            try:
                for audio_chunk in kokoro_model.generate_stream(
                    text=request.input,
                    voice=voice,
                    speed=1.0,
                ):
                    pcm_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                    yield pcm_bytes
            except Exception as e:
                print(f"Streaming error: {e}")

        return StreamingResponse(pcm_stream(), media_type="audio/pcm")
    else:
        def _generate():
            result = kokoro_model.generate(
                text=request.input,
                voice=voice,
                speed=1.0,
            )
            return result.audio, result.sample_rate

        audio_data, sample_rate = await asyncio.to_thread(_generate)

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format="wav")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("CORTEX_VOICE_PORT", "8005"))
    print(f"Starting MLX Voice Server (mlx-whisper STT + kokoro-mlx TTS) on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port)
