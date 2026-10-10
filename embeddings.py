"""
Embeddings — semantic vectors for retrieval-based relevance.
100% local via Ollama's nomic-embed-text. No OpenAI, no API key, ever.

This is the practical substitute for BIG BRAIN's "System 1 fast-weight adaptation":
since we cannot mutate any model's actual weights, relevance is instead learned/
selected via nearest-neighbor retrieval over embedded context, memory, and past
experiences. The model doesn't change; what it SEES does, selected by similarity
to the current situation.

Model: nomic-embed-text (768 dims, local, fast on Apple Silicon).
"""

import os
import logging

import httpx

logger = logging.getLogger("cortex.embeddings")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768


def embed(text: str, api_key: str = None) -> list[float]:
    """
    Embed a single string using the local nomic-embed-text model via Ollama.
    api_key is accepted and ignored for backward-compat with old call sites —
    no key is ever read or used.
    Returns a 768-dim vector, or None on failure.
    """
    if not text or not text.strip():
        return None
    try:
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text[:8000]},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings")
        if not embeddings:
            return None
        return embeddings[0]
    except Exception as e:
        logger.error(f"Local embedding failed: {e}")
        return None


def embed_batch(texts: list[str], api_key: str = None) -> list:
    """
    Embed multiple strings via local Ollama. Returns list of vectors
    (None for empty inputs), same length as input.
    """
    if not texts:
        return []
    non_empty = [(i, t[:8000]) for i, t in enumerate(texts) if t and t.strip()]
    if not non_empty:
        return [None] * len(texts)

    try:
        indices, clean_texts = zip(*non_empty)
        resp = httpx.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": list(clean_texts)},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])

        results = [None] * len(texts)
        for idx, vec in zip(indices, embeddings):
            results[idx] = vec
        return results
    except Exception as e:
        logger.error(f"Local batch embedding failed: {e}")
        return [None] * len(texts)


def to_pgvector(vec: list[float]) -> str:
    """Format a Python float list as a pgvector literal string for SQL params."""
    if vec is None:
        return None
    return "[" + ",".join(str(x) for x in vec) + "]"
