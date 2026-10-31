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
import shutil
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
    """pdftotext keeps the layout best, but isn't on Windows by default; pypdf is everywhere."""
    if shutil.which("pdftotext"):
        r = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, timeout=90)
        if r.stdout.strip():
            return r.stdout.strip()
    try:
        from pypdf import PdfReader
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages).strip()
    except ImportError:
        raise RuntimeError("reading PDFs needs pdftotext, or: pip install pypdf")


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
    """Describe/transcribe an image with a vision model: locally (qwen3-vl via Ollama), or
    through Ollama Cloud when there's a key and the local engine can't do it."""
    import httpx
    from cortex.llm import llm
    with open(image_path, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    body = {"messages": [{"role": "user", "content": prompt, "images": [b]}], "stream": False}
    try:
        r = httpx.post("http://localhost:11434/api/chat", timeout=180,
                       json={**body, "model": os.environ.get("CORTEX_VISION_MODEL", "qwen3-vl:8b")})
        r.raise_for_status()
        return (r.json().get("message", {}) or {}).get("content", "").strip()
    except Exception as e:
        key = llm.get_key("ollama")
        if not key:
            raise RuntimeError(f"reading images needs a local vision model or an Ollama Cloud key ({e})")
    r = httpx.post(f"{llm.OLLAMA_CLOUD_URL}/api/chat", timeout=180, headers={"Authorization": f"Bearer {key}"},
                   json={**body, "model": os.environ.get("CORTEX_CLOUD_VISION_MODEL", "gemma4:31b")})
    r.raise_for_status()
    return (r.json().get("message", {}) or {}).get("content", "").strip()


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

    With no embedding model (a machine that only uses cloud models, or Ollama stopped)
    the chunks are stored without vectors, and retrieve() finds those by keyword.
    """
    from cortex.llm.embeddings import embed, to_pgvector
    source = source or os.path.basename(path)
    text = extract(path)
    if not (text or "").strip():
        raise RuntimeError(f"couldn't read any text from {source} (unsupported or empty file?)")
    chunks = chunk(text)
    if not chunks:
        raise RuntimeError(f"no content to index in {source}")
    db.clear_document(source)       # uploading a file again replaces it
    embedded = 0
    for ch in chunks:
        vec = embed(ch)
        db.add_document(source, ch, to_pgvector(vec) if vec else None)
        embedded += 1 if vec else 0
    logger.info(f"ingested {source}: {len(chunks)} chunks, {embedded} embedded")
    return source, len(chunks)


def retrieve(db, query: str, k: int = 6) -> list[dict]:
    from cortex.llm.embeddings import embed
    qv = embed(query)
    hits = db.search_documents(qv, limit=k) if qv else []
    return hits + db.search_documents_keyword(query, limit=k)
