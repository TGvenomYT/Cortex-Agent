"""
rag.py — multi-format file ingestion + retrieval (RAG) for Cortex.

Handles (almost) any file: text/code/csv/json directly, PDFs (pdftotext),
DOCX (stdlib unzip), IMAGES (vision via qwen3-vl — Ollama can pass images to a
non-MLX model), VIDEOS (audio → mlx-whisper transcript + a keyframe caption),
and a best-effort text fallback for anything else. Extracted text is chunked,
embedded (nomic-embed-text, local) and stored in the local DB; at query time
the most relevant chunks are retrieved and fed to the model.
"""
from __future__ import annotations

import os
import re
import base64
import zipfile
import tempfile
import subprocess
import logging

logger = logging.getLogger("cortex.rag")

TEXT_EXT = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml",
            ".log", ".py", ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".sh",
            ".c", ".cpp", ".h", ".java", ".go", ".rs", ".rb", ".sql", ".xml",
            ".ini", ".toml", ".conf", ".env", ".tex", ""}
IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic"}
VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def extract(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXT:
        return _text(path)
    if ext == ".pdf":
        return _pdf(path)
    if ext == ".docx":
        return _docx(path)
    if ext in IMG_EXT:
        return _image(path)
    if ext in VID_EXT:
        return _video(path)
    t = _text(path)
    return t or f"[{os.path.basename(path)}: binary file, {os.path.getsize(path)} bytes — no text extractable]"


def _text(path: str, limit: int = 300_000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read(limit)
    except Exception:
        return ""


def _pdf(path: str) -> str:
    try:
        r = subprocess.run(["pdftotext", "-layout", path, "-"],
                           capture_output=True, text=True, timeout=90)
        return r.stdout.strip() or "[pdf: no extractable text]"
    except Exception as e:
        return f"[pdf error: {e}]"


def _docx(path: str) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
        paras = re.split(r"</w:p>", xml)
        out = []
        for p in paras:
            out.append("".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", p, re.S)))
        return "\n".join(t for t in out if t.strip()) or "[docx: empty]"
    except Exception as e:
        return f"[docx error: {e}]"


def _vision(image_path: str, prompt: str) -> str:
    """Describe/transcribe an image with a vision model (qwen3-vl via Ollama)."""
    import httpx
    try:
        b = base64.b64encode(open(image_path, "rb").read()).decode()
        r = httpx.post("http://localhost:11434/api/chat", json={
            "model": os.environ.get("CORTEX_VISION_MODEL", "qwen3-vl:8b"),
            "messages": [{"role": "user", "content": prompt, "images": [b]}],
            "stream": False}, timeout=180)
        return (r.json().get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        return f"[vision error: {e}]"


def _image(path: str) -> str:
    return "[image: " + os.path.basename(path) + "]\n" + _vision(
        path, "Describe this image in detail, and transcribe any visible text exactly.")


def _video(path: str) -> str:
    out = []
    # 1) audio -> transcript (mlx-whisper)
    try:
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", path, "-ar", "16000", "-ac", "1", wav],
                       capture_output=True, timeout=600)
        import mlx_whisper
        res = mlx_whisper.transcribe(wav, path_or_hf_repo="mlx-community/whisper-small-mlx")
        txt = (res.get("text") or "").strip()
        if txt:
            out.append("[video transcript]\n" + txt)
        os.unlink(wav)
    except Exception as e:
        out.append(f"[video audio error: {e}]")
    # 2) a keyframe caption
    try:
        frame = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-ss", "1", "-i", path, "-frames:v", "1", frame],
                       capture_output=True, timeout=60)
        cap = _vision(frame, "Describe what is shown in this video frame.")
        os.unlink(frame)
        if cap:
            out.append("[video scene] " + cap)
    except Exception:
        pass
    return "\n\n".join(out) or "[video: nothing extractable]"


def chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text) and len(out) < 300:
        out.append(text[i:i + size])
        i += size - overlap
    return out


def ingest(db, path: str, source: str = None) -> tuple[str, int]:
    """Extract -> chunk -> embed -> store. Returns (source_name, n_embedded_chunks).

    Fails LOUDLY (raises) rather than silently storing chunks with no embedding —
    an un-embedded chunk is invisible to retrieval, so a 'successful' upload that
    can never be found is worse than a clear error.
    """
    from cortex.llm.embeddings import embed, to_pgvector
    source = source or os.path.basename(path)
    text = extract(path)
    if not (text or "").strip():
        raise RuntimeError(f"couldn't read any text from {source} (unsupported or empty file?)")
    chunks = chunk(text)
    if not chunks:
        raise RuntimeError(f"no content to index in {source}")
    # Probe the embedder once up front so we fail fast with a helpful message
    # instead of silently storing hundreds of unretrievable chunks.
    probe = embed(chunks[0])
    if probe is None:
        raise RuntimeError(
            "couldn't reach the embedding model — make sure Ollama is running and the "
            "embedding model is installed:  ollama pull nomic-embed-text")
    embedded = 0
    for i, ch in enumerate(chunks):
        vec = probe if i == 0 else embed(ch)
        if vec:
            db.add_document(source, ch, to_pgvector(vec))
            embedded += 1
    if embedded == 0:
        raise RuntimeError(f"embedding failed for every chunk of {source}")
    logger.info(f"ingested {source}: {embedded}/{len(chunks)} chunks embedded")
    return source, embedded


def retrieve(db, query: str, k: int = 6) -> list[dict]:
    from cortex.llm.embeddings import embed
    qv = embed(query)
    if not qv:
        return []
    return db.search_documents(qv, limit=k)
