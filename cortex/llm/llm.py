"""
LLM Router — multi-provider, plug-and-play.

Cortex is local-first (Ollama, no key needed) but can also talk to the big cloud
providers when the user pastes an API key: Anthropic (Claude), OpenAI, and Google
(Gemini). Keys live in the OS secret store (never in config/plaintext). All
providers go through ONE normalized interface so the agent loop in cortex_core
doesn't care who's behind the model.

Normalized message format used by the agent loop (and by every backend):
  {"role": "system"|"user", "content": str}
  {"role": "assistant", "content": str, "tool_calls": [
       {"id": str, "function": {"name": str, "arguments": dict}} ]}   # tool_calls optional
  {"role": "tool", "tool_call_id": str, "content": str}

complete_with_tools() returns: {"content": str, "tool_calls": [...], "thinking": str}
where tool_calls are in the SAME normalized shape (arguments as a dict). Each
backend converts this to/from its own wire format.
"""

import os
import json
import time
import uuid
import logging

import httpx

logger = logging.getLogger("cortex.llm")


# ── provider model lists (shown in the GUI when a key is present) ─────────────
# Anthropic IDs per the Claude API reference; others are widely-available chat models.
CLOUD_MODELS = {
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
}
_SECRET_ACCOUNT = {"anthropic": "apikey:anthropic", "openai": "apikey:openai", "gemini": "apikey:gemini"}


def provider_for_model(model: str) -> str:
    """Infer which provider a model string belongs to. Ollama uses name:tag
    (gemma4:12b-mlx, gpt-oss:120b-cloud) — a colon means it's a local/Ollama
    model, even if the name starts with 'gpt'. Cloud model IDs have no colon."""
    m = (model or "").lower()
    if ":" in m:
        return "ollama"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    return "ollama"


def get_key(provider: str) -> str | None:
    acct = _SECRET_ACCOUNT.get(provider)
    if not acct:
        return None
    try:
        import cortex.util.osutil as osutil
        return osutil.secret_get(acct)
    except Exception:
        return None


def set_key(provider: str, key: str) -> bool:
    acct = _SECRET_ACCOUNT.get(provider)
    if not acct:
        return False
    import cortex.util.osutil as osutil
    return osutil.secret_set(acct, key.strip())


def delete_key(provider: str):
    acct = _SECRET_ACCOUNT.get(provider)
    if acct:
        import cortex.util.osutil as osutil
        osutil.secret_delete(acct)


def configured_providers() -> dict:
    """{provider: bool has_key} for the cloud providers."""
    return {p: bool(get_key(p)) for p in CLOUD_MODELS}


# Substrings of provider model ids that are NOT usable via plain chat completions
# (audio/realtime/image/embedding/tts, responses-API-only "-pro"/codex, search previews…).
# Listing them would let the user pick a model that errors out as a 404/400.
_NON_CHAT = ("realtime", "audio", "transcribe", "tts", "image", "imagen", "instruct",
             "embedding", "moderation", "whisper", "dall-e", "codex", "search",
             "computer-use", "deep-research", "davinci", "babbage", "veo", "live",
             "aqa", "robotics", "learnlm")
# OpenAI's "-pro" models (o1-pro, gpt-5-pro) are Responses-API-only, but "-pro" is a
# perfectly good chat model on Gemini (gemini-2.5-pro) — so only exclude it for OpenAI.
_NON_CHAT_OPENAI_ONLY = ("-pro",)


def _chat_only(ids, provider=None):
    extra = _NON_CHAT_OPENAI_ONLY if provider == "openai" else ()
    return [i for i in ids if i and not any(w in i.lower() for w in _NON_CHAT + extra)]


_MODEL_CACHE: dict[str, tuple[float, list]] = {}
_MODEL_CACHE_TTL = 600.0  # 10 min — /models is polled often; don't hammer the provider API


def list_provider_models(provider: str) -> list:
    """Fetch the REAL available models for a provider (its /models endpoint), so the
    selector never shows a deprecated/invalid id. Cached 10 min. Falls back to the
    curated list in CLOUD_MODELS if the call fails (offline, bad key)."""
    key = get_key(provider)
    if not key:
        return []
    hit = _MODEL_CACHE.get(provider)
    if hit and (time.time() - hit[0]) < _MODEL_CACHE_TTL:
        return hit[1]
    try:
        result = None
        if provider == "openai":
            r = httpx.get("https://api.openai.com/v1/models",
                          headers={"Authorization": f"Bearer {key}"}, timeout=10)
            r.raise_for_status()
            ids = [m.get("id", "") for m in r.json().get("data", [])]
            result = sorted(_chat_only((i for i in ids if i.startswith(("gpt-", "o1", "o3", "o4", "chatgpt"))), "openai"))
        elif provider == "gemini":
            r = httpx.get("https://generativelanguage.googleapis.com/v1beta/openai/models",
                          headers={"Authorization": f"Bearer {key}"}, timeout=10)
            r.raise_for_status()
            ids = [m.get("id", "").split("/")[-1] for m in r.json().get("data", [])]
            result = sorted(_chat_only((i for i in ids if i.startswith("gemini")), "gemini"))
        elif provider == "anthropic":
            r = httpx.get("https://api.anthropic.com/v1/models",
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=10)
            r.raise_for_status()
            result = sorted(m.get("id", "") for m in r.json().get("data", []))
        if result:
            _MODEL_CACHE[provider] = (time.time(), result)
            return result
    except Exception as e:
        logger.info(f"live model list for {provider} failed ({e}) — using curated fallback")
    return CLOUD_MODELS.get(provider, [])


class LLMBackend:
    """Abstract base for all LLM backends."""

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        raise NotImplementedError

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        raise NotImplementedError


# ── Ollama (local) ───────────────────────────────────────────────────────────
class OllamaBackend(LLMBackend):
    """Any local Ollama model. No cloud calls, no API keys."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "gemma4:e4b", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        start = time.time()
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        logger.info(f"Ollama [{self.model}] {int((time.time()-start)*1000)}ms (local)")
        return r.json()["message"]["content"]

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": self.model, "messages": messages, "tools": tools, "stream": False}
        start = time.time()
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        msg = r.json().get("message", {})
        logger.info(f"Ollama [{self.model}] {int((time.time()-start)*1000)}ms (local, tools)")
        tcs = []
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            tcs.append({"id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                        "function": {"name": fn.get("name", ""), "arguments": args or {}}})
        return {"content": msg.get("content", ""), "tool_calls": tcs, "thinking": msg.get("thinking", "")}

    def stream(self, messages: list[dict]):
        payload = {"model": self.model, "messages": messages, "stream": True}
        with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    c = json.loads(line).get("message", {}).get("content", "")
                    if c:
                        yield c
                except json.JSONDecodeError:
                    continue


# ── OpenAI-compatible (OpenAI + Gemini's OpenAI endpoint) ────────────────────
class OpenAICompatBackend(LLMBackend):
    """OpenAI Chat Completions format. Also drives Gemini via its OpenAI-compatible
    endpoint, so one implementation covers both."""

    def __init__(self, base_url: str, api_key: str, model: str, label: str = "openai", timeout: int = 90):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.label = label
        self.timeout = timeout

    def _headers(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _wire_messages(self, messages):
        out = []
        for m in messages:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                out.append({
                    "role": "assistant",
                    "content": m.get("content") or "",
                    "tool_calls": [{
                        "id": tc["id"], "type": "function",
                        "function": {"name": tc["function"]["name"],
                                     "arguments": json.dumps(tc["function"].get("arguments") or {})},
                    } for tc in m["tool_calls"]],
                })
            elif role == "tool":
                out.append({"role": "tool", "tool_call_id": m.get("tool_call_id", ""),
                            "content": m.get("content", "")})
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        payload = {"model": self.model, "messages": self._wire_messages(messages)}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        start = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                       json=payload, timeout=self.timeout)
        r.raise_for_status()
        logger.info(f"{self.label} [{self.model}] {int((time.time()-start)*1000)}ms")
        return r.json()["choices"][0]["message"].get("content") or ""

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {"model": self.model, "messages": self._wire_messages(messages), "tools": tools}
        start = time.time()
        r = httpx.post(f"{self.base_url}/chat/completions", headers=self._headers(),
                       json=payload, timeout=self.timeout)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        logger.info(f"{self.label} [{self.model}] {int((time.time()-start)*1000)}ms (tools)")
        tcs = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            tcs.append({"id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                        "function": {"name": fn.get("name", ""), "arguments": args}})
        return {"content": msg.get("content") or "", "tool_calls": tcs, "thinking": ""}


# ── Anthropic (Claude) — native Messages API ─────────────────────────────────
class AnthropicBackend(LLMBackend):
    """Claude via the Anthropic Messages API (POST /v1/messages)."""

    def __init__(self, api_key: str, model: str = "claude-opus-5-5",
                 timeout: int = 120, max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.base_url = "https://api.anthropic.com/v1"

    def _headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                "content-type": "application/json"}

    def _convert(self, messages):
        """Split out the system prompt and convert the normalized history into
        Anthropic's content-block format (tool_use / tool_result), grouping
        consecutive tool results into one user turn."""
        system_parts, conv, pending_tools = [], [], []

        def flush_tools():
            if pending_tools:
                conv.append({"role": "user", "content": list(pending_tools)})
                pending_tools.clear()

        for m in messages:
            role = m.get("role")
            if role == "system":
                if m.get("content"):
                    system_parts.append(m["content"])
                continue
            if role == "tool":
                pending_tools.append({"type": "tool_result",
                                      "tool_use_id": m.get("tool_call_id", ""),
                                      "content": m.get("content", "")})
                continue
            flush_tools()
            if role == "assistant" and m.get("tool_calls"):
                blocks = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": tc["id"],
                                   "name": tc["function"]["name"],
                                   "input": tc["function"].get("arguments") or {}})
                conv.append({"role": "assistant", "content": blocks})
            else:
                conv.append({"role": role, "content": m.get("content", "")})
        flush_tools()
        return "\n\n".join(system_parts), conv

    @staticmethod
    def _tools_to_anthropic(tools):
        out = []
        for t in tools or []:
            fn = t.get("function", t)
            out.append({"name": fn.get("name"), "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}}})
        return out

    def _post(self, payload):
        start = time.time()
        r = httpx.post(f"{self.base_url}/messages", headers=self._headers(),
                       json=payload, timeout=self.timeout)
        r.raise_for_status()
        logger.info(f"anthropic [{self.model}] {int((time.time()-start)*1000)}ms")
        return r.json()

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        system, conv = self._convert(messages)
        if json_mode:
            system = (system + "\n\nRespond with ONLY valid JSON, no prose.").strip()
        data = self._post({"model": self.model, "max_tokens": self.max_tokens,
                           "system": system, "messages": conv})
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        system, conv = self._convert(messages)
        data = self._post({"model": self.model, "max_tokens": self.max_tokens,
                           "system": system, "messages": conv,
                           "tools": self._tools_to_anthropic(tools)})
        content, thinking, tcs = "", "", []
        for b in data.get("content", []):
            bt = b.get("type")
            if bt == "text":
                content += b.get("text", "")
            elif bt == "thinking":
                thinking += b.get("thinking", "") or ""
            elif bt == "tool_use":
                tcs.append({"id": b.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                            "function": {"name": b.get("name", ""), "arguments": b.get("input") or {}}})
        return {"content": content, "tool_calls": tcs, "thinking": thinking}


# ── router ───────────────────────────────────────────────────────────────────
class LLMRouter:
    """Routes to a primary backend, falls back to a secondary on failure."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend = None):
        self.primary = primary
        self.fallback = fallback

    def complete(self, messages, json_mode=False, schema=None):
        try:
            return self.primary.complete(messages, json_mode=json_mode, schema=schema)
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}")
            if self.fallback:
                try:
                    return self.fallback.complete(messages, json_mode=json_mode, schema=schema)
                except Exception:
                    raise e  # report the primary's error, not the fallback's
            raise

    def complete_with_tools(self, messages, tools):
        try:
            return self.primary.complete_with_tools(messages, tools)
        except Exception as e:
            logger.warning(f"Primary LLM failed (tools): {e}")
            if self.fallback:
                try:
                    return self.fallback.complete_with_tools(messages, tools)
                except Exception:
                    raise e  # report the primary's error, not the fallback's
            raise

    def stream(self, messages):
        fn = getattr(self.primary, "stream", None)
        if fn:
            yield from fn(messages)
        else:
            yield self.primary.complete(messages)


def _ollama_base(config: dict) -> str:
    return (config.get("ollama", {}) or {}).get("base_url", "http://localhost:11434")


def backend_for_model(model: str, config: dict, timeout: int = 90) -> LLMBackend:
    """Build the right backend for a model string, pulling the key from the OS
    secret store for cloud providers. Falls back to local Ollama."""
    provider = provider_for_model(model)
    if provider == "anthropic":
        key = get_key("anthropic")
        if key:
            return AnthropicBackend(api_key=key, model=model, timeout=max(timeout, 120))
        logger.warning("Claude model selected but no Anthropic API key set — using local model")
    elif provider == "openai":
        key = get_key("openai")
        if key:
            return OpenAICompatBackend("https://api.openai.com/v1", key, model, "openai", timeout)
        logger.warning("OpenAI model selected but no OpenAI API key set — using local model")
    elif provider == "gemini":
        key = get_key("gemini")
        if key:
            return OpenAICompatBackend("https://generativelanguage.googleapis.com/v1beta/openai",
                                       key, model, "gemini", timeout)
        logger.warning("Gemini model selected but no Google API key set — using local model")
    # default: local Ollama
    return OllamaBackend(base_url=_ollama_base(config), model=model, timeout=timeout)


def create_router(config: dict) -> LLMRouter:
    """Factory: primary from config.yaml (local Ollama by default), local fallback."""
    ollama_cfg = config.get("ollama", {})
    fallback_cfg = config.get("ollama_fallback", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")

    primary_model = ollama_cfg.get("model", "gemma4:e4b")
    primary = backend_for_model(primary_model, config, timeout=ollama_cfg.get("timeout", 90))

    fallback = None
    fallback_model = fallback_cfg.get("model") or config.get("fallback")
    if fallback_model:
        fallback = OllamaBackend(base_url=fallback_cfg.get("base_url", base_url),
                                 model=fallback_model, timeout=fallback_cfg.get("timeout", 90))

    logger.info(f"LLM Router: primary={primary_model}, "
                f"fallback={fallback_model or 'none'}")
    return LLMRouter(primary=primary, fallback=fallback)
