"""
resources.py — what Cortex is costing this machine right now, and the levers to lower it.

The biggest cost by far is the language model sitting in memory (a 12B model is
~8–15 GB of RAM/GPU). Ollama keeps it loaded for a while after every request;
this module shows what's loaded and can unload it on demand. The "how long to keep it"
policy lives in settings (`resources.keep_alive`, applied in cortex/llm/llm.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

logger = logging.getLogger("cortex.resources")

KEEP_ALIVE_CHOICES = [("0", "Free it right after each reply"), ("1m", "1 minute"), ("2m", "2 minutes"),
                      ("5m", "5 minutes (Ollama's default)"), ("30m", "30 minutes"), ("-1", "Keep loaded")]


def ollama_base() -> str:
    try:
        from cortex.util.cortex_bootstrap import load_config
        return ((load_config().get("llm", {}).get("ollama") or {}).get("base_url")
                or "http://127.0.0.1:11434").replace("localhost", "127.0.0.1")
    except Exception:
        return "http://127.0.0.1:11434"


def loaded_models() -> list[dict]:
    """Models Ollama currently holds in memory: [{name, size_gb, expires_in}] (empty if Ollama is down)."""
    try:
        r = httpx.get(ollama_base() + "/api/ps", timeout=2.5)
        out = []
        for m in r.json().get("models", []) or []:
            exp = m.get("expires_at") or ""
            left = None
            try:
                dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
                if dt.year > 1:
                    left = max(0, int((dt - datetime.now(timezone.utc)).total_seconds()))
            except Exception:
                pass
            out.append({"name": m.get("name") or m.get("model"),
                        "size_gb": round((m.get("size_vram") or m.get("size") or 0) / 1e9, 1),
                        "expires_in": left})   # None = stays until unloaded
        return out
    except Exception:
        return []


def unload_models() -> str:
    """Tell Ollama to drop every loaded model right now. Safe — the next request just reloads it."""
    models = loaded_models()
    if not models:
        return "No AI model is loaded in memory right now."
    freed = 0.0
    for m in models:
        try:
            httpx.post(ollama_base() + "/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=15)
            freed += m["size_gb"]
        except Exception as e:
            logger.warning(f"unload {m['name']}: {e}")
    return f"Unloaded {len(models)} model(s) — freed about {freed:.1f} GB."


def snapshot() -> dict:
    import cortex.util.power as power
    import cortex.storage.settings_store as settings_store
    snap = power.snapshot()
    snap["constrained"], snap["constrained_why"] = power.is_constrained(snap)
    snap["models"] = loaded_models()
    snap["settings"] = settings_store.section("resources")
    return snap


def describe() -> str:
    s = snapshot()
    models = ", ".join(f"{m['name']} ({m['size_gb']} GB)" for m in s["models"]) or "none"
    batt = "" if s["battery"] is None else f" ({s['battery']}%)"
    return (f"AI model in memory: {models}. Power: {'plugged in' if s['on_ac'] else 'on battery'}{batt}, "
            f"thermal {s['thermal']}, memory free {s['free_mem_pct']}%.")
