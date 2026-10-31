"""
LLM Router — multi-provider, plug-and-play.

Cortex is local-first (Ollama on this machine, no key needed) but can also talk to the
cloud when the user pastes an API key: Ollama Cloud, Anthropic (Claude), OpenAI, and
Google (Gemini). Keys live in the OS secret store (never in config/plaintext). All
providers go through ONE normalized interface so the agent loop in cortex_core
doesn't care who's behind the model.

Ollama has two routes, and which one a model takes is decided by where it runs:
  local models  -> http://localhost:11434            (no key)
  cloud models  -> https://ollama.com/api, Bearer key (no local Ollama needed at all)
A cloud model picked in the app is named `ollama-cloud/<name>`, where <name> is the id from
ollama.com/api/tags. The daemon's own spelling of the same models (gpt-oss:120b-cloud,
minimax-m3:cloud) also works: with a key it takes the direct route, without one it goes
through the signed-in local daemon.

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
import itertools

import httpx

logger = logging.getLogger("cortex.llm")


# ── provider model lists (shown in the GUI when a key is present) ─────────────
# Anthropic IDs per the Claude API reference; others are widely-available chat models.
CLOUD_MODELS = {
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "gemini": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
    "ollama": [],       # always fetched live; the list needs no key
}
_SECRET_ACCOUNT = {"anthropic": "apikey:anthropic", "openai": "apikey:openai", "gemini": "apikey:gemini",
                   "ollama": "apikey:ollama"}
OLLAMA_CLOUD_URL = "https://ollama.com"
CLOUD_PREFIX = "ollama-cloud/"
# Which Ollama Cloud model to rescue a job with if the local engine can't be reached.
_RESCUE_ORDER = ("gemma4:31b", "gpt-oss:120b", "glm-5.3", "kimi-k3", "minimax-m3")
PROVIDER_LABEL = {"ollama": "Ollama Cloud", "anthropic": "Claude", "openai": "OpenAI", "gemini": "Gemini"}


def provider_for_model(model: str) -> str:
    """Infer which provider a model string belongs to. Ollama uses name:tag
    (gemma4:12b-mlx, gpt-oss:120b-cloud) — a colon means it's a local/Ollama
    model, even if the name starts with 'gpt'. Cloud model IDs have no colon."""
    m = (model or "").lower()
    if m.startswith(CLOUD_PREFIX):
        return "ollama_cloud"
    if m.endswith(("-cloud", ":cloud")):
        return "ollama_cloud" if get_key("ollama") else "ollama"
    if ":" in m:
        return "ollama"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt")):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    return "ollama"


def key_provider(model: str) -> str | None:
    """Which provider's key this model can't run without, or None."""
    p = provider_for_model(model)
    if p == "ollama_cloud":
        return "ollama"
    return p if p in ("anthropic", "openai", "gemini") else None


def cloud_model_name(model: str) -> str:
    """The id ollama.com expects, from any spelling: `ollama-cloud/gpt-oss:120b`, or the
    daemon's `gpt-oss:120b-cloud` and `gemma4:cloud`, the last of which names a family."""
    m = model[len(CLOUD_PREFIX):] if model.startswith(CLOUD_PREFIX) else model
    if m.endswith("-cloud"):
        m = m[:-len("-cloud")]
    elif m.endswith(":cloud"):
        family = m[:-len(":cloud")]
        names = [n[len(CLOUD_PREFIX):] for n in list_provider_models("ollama")]
        if family in names:
            return family
        return next((n for n in names if n.split(":")[0] == family), family)
    return m


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
    if not key and provider != "ollama":     # Ollama Cloud's list is public
        return []
    hit = _MODEL_CACHE.get(provider)
    if hit and (time.time() - hit[0]) < _MODEL_CACHE_TTL:
        return hit[1]
    try:
        result = None
        if provider == "ollama":
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            r = httpx.get(f"{OLLAMA_CLOUD_URL}/api/tags", headers=headers, timeout=6)
            r.raise_for_status()
            result = sorted(CLOUD_PREFIX + m["name"] for m in r.json().get("models", [])
                            if m.get("name") and "completion" in (m.get("capabilities") or ["completion"]))
        elif provider == "openai":
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

    def stream_with_tools(self, messages: list[dict], tools: list[dict]):
        """('delta', text)* then ('done', same dict as complete_with_tools). Backends that
        can't stream just produce the 'done'."""
        yield "done", self.complete_with_tools(messages, tools)


def resolve_keep_alive() -> str:
    """How long Ollama keeps the model resident AFTER a request (its RAM/GPU is the
    single biggest thing Cortex costs you). Background jobs (run_once sets
    CORTEX_BACKGROUND) free it almost immediately; the interactive chat uses the
    user's Settings → Background choice (default 2m, so a conversation stays snappy
    but the model doesn't squat on ~8-15 GB for Ollama's default 5 min)."""
    if os.environ.get("CORTEX_BACKGROUND"):
        return os.environ.get("CORTEX_BG_KEEP_ALIVE", "15s")
    try:
        import cortex.storage.settings_store as settings_store
        v = (settings_store.load().get("resources") or {}).get("keep_alive")
        if v not in (None, ""):
            return str(v)
    except Exception:
        pass
    return "2m"


# ── Ollama (this machine, or Ollama Cloud with a key) ────────────────────────
def _tool_args(args) -> dict:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {}
    return args if isinstance(args, dict) else {}


class OllamaBackend(LLMBackend):
    """One Ollama model. Same API either way: a local daemon, or ollama.com with a Bearer
    key. Local-only knobs (context size, how long the model stays in memory) aren't sent
    to the cloud."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "gemma4:e4b",
                 timeout: int = 60, api_key: str = None, num_ctx: int = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.api_key = api_key
        self.num_ctx = num_ctx
        self._no_tools = False      # this model turned out not to support tool calling

    @property
    def label(self) -> str:
        return "ollama-cloud" if self.api_key else "local"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _httpx_timeout(self):
        # Fail fast if nothing's there, but give a model that's loading time to answer.
        return httpx.Timeout(self.timeout, connect=4.0)

    @staticmethod
    def _wire(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                    {"type": "function", "function": {"name": tc["function"]["name"],
                                                      "arguments": tc["function"].get("arguments") or {}}}
                    for tc in m["tool_calls"]]})
            elif role == "tool":
                msg = {"role": "tool", "content": m.get("content", "")}
                if m.get("tool_name"):
                    msg["tool_name"] = m["tool_name"]
                out.append(msg)
            else:
                out.append({"role": role, "content": m.get("content", "")})
        return out

    def _payload(self, messages: list[dict], stream: bool = False, tools: list = None) -> dict:
        payload = {"model": self.model, "messages": self._wire(messages), "stream": stream}
        if tools and not self._no_tools:
            payload["tools"] = tools
        if not self.api_key:
            payload["keep_alive"] = resolve_keep_alive()
            if self.num_ctx:
                payload["options"] = {"num_ctx": self.num_ctx}
        return payload

    def _post(self, payload: dict) -> dict:
        start = time.time()
        r = httpx.post(f"{self.base_url}/api/chat", json=payload, headers=self._headers(),
                       timeout=self._httpx_timeout())
        if r.status_code == 400 and "tools" in payload and "tool" in r.text.lower():
            self._no_tools = True
            logger.info(f"{self.model} can't call tools; carrying on without them")
            payload = {k: v for k, v in payload.items() if k != "tools"}
            r = httpx.post(f"{self.base_url}/api/chat", json=payload, headers=self._headers(),
                           timeout=self._httpx_timeout())
        r.raise_for_status()
        logger.info(f"Ollama [{self.model}] {int((time.time()-start)*1000)}ms ({self.label})")
        return r.json()

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        payload = self._payload(messages)
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"
        return self._post(payload)["message"]["content"]

    @staticmethod
    def _calls(raw: list) -> list[dict]:
        """Normalize tool calls, joining any that arrived in pieces across stream chunks."""
        merged: dict = {}
        for n, tc in enumerate(raw):
            fn = tc.get("function") or {}
            key = fn.get("index", f"#{n}")
            if key in merged:
                merged[key]["function"]["arguments"].update(_tool_args(fn.get("arguments")))
            else:
                merged[key] = {"id": tc.get("id") or ("call_" + uuid.uuid4().hex[:8]),
                               "function": {"name": fn.get("name", ""), "arguments": _tool_args(fn.get("arguments"))}}
        return list(merged.values())

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        msg = self._post(self._payload(messages, tools=tools)).get("message", {})
        return {"content": msg.get("content", ""), "tool_calls": self._calls(msg.get("tool_calls") or []),
                "thinking": msg.get("thinking", "")}

    def _lines(self, payload: dict):
        """The stream as parsed chunks. Raises with a readable body on an HTTP error."""
        with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, headers=self._headers(),
                          timeout=self._httpx_timeout()) as r:
            if r.status_code >= 400:
                r.read()
                r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    raise RuntimeError(chunk["error"])
                yield chunk

    def _open(self, payload: dict):
        """Start the stream and read its first chunk, so a rejected request surfaces here
        rather than midway through a reply. -> (rest_of_stream, first_chunk)"""
        stream = self._lines(payload)
        return stream, next(stream, None)

    def stream_with_tools(self, messages: list[dict], tools: list[dict]):
        content, thinking, calls = "", "", []
        start = time.time()
        payload = self._payload(messages, stream=True, tools=tools)
        try:
            stream, first = self._open(payload)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 400 or "tools" not in payload or "tool" not in e.response.text.lower():
                raise
            self._no_tools = True
            logger.info(f"{self.model} can't call tools; carrying on without them")
            stream, first = self._open({k: v for k, v in payload.items() if k != "tools"})
        for chunk in itertools.chain([first] if first else [], stream):
            msg = chunk.get("message") or {}
            thinking += msg.get("thinking") or ""
            calls.extend(msg.get("tool_calls") or [])
            piece = msg.get("content") or ""
            if piece:
                content += piece
                yield "delta", piece
        logger.info(f"Ollama [{self.model}] {int((time.time()-start)*1000)}ms ({self.label}, stream)")
        yield "done", {"content": content, "tool_calls": self._calls(calls), "thinking": thinking}

    def stream(self, messages: list[dict]):
        for c in self._lines(self._payload(messages, stream=True)):
            piece = (c.get("message") or {}).get("content", "")
            if piece:
                yield piece


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
    """Tries the chosen model, then each fallback in order. `fallback` is the next model
    (local), `rescue` a cloud model used when nothing local answers."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend = None, rescue: LLMBackend = None):
        self.primary = primary
        self.fallback = fallback
        self.rescue = rescue

    def is_local(self) -> bool:
        return isinstance(self.primary, OllamaBackend) and not self.primary.api_key

    def _chain(self) -> list:
        return [b for b in (self.primary, self.fallback, self.rescue) if b is not None]

    def _first_answer(self, call, what: str):
        first_error = None
        for backend in self._chain():
            try:
                return call(backend)
            except Exception as e:
                logger.warning(f"{what} [{getattr(backend, 'model', '?')}] failed: {e}")
                first_error = first_error or e
        raise first_error  # report the chosen model's error, not the last fallback's

    def complete(self, messages, json_mode=False, schema=None):
        return self._first_answer(lambda b: b.complete(messages, json_mode=json_mode, schema=schema), "LLM")

    def complete_with_tools(self, messages, tools):
        return self._first_answer(lambda b: b.complete_with_tools(messages, tools), "LLM (tools)")

    def stream_with_tools(self, messages, tools):
        first_error = None
        for backend in self._chain():
            started = False
            try:
                for kind, payload in backend.stream_with_tools(messages, tools):
                    started = started or kind == "delta"
                    yield kind, payload
                return
            except Exception as e:
                if started:      # part of a reply is already on screen; don't splice in another model
                    raise
                logger.warning(f"LLM stream [{getattr(backend, 'model', '?')}] failed: {e}")
                first_error = first_error or e
        raise first_error

    def stream(self, messages):
        fn = getattr(self.primary, "stream", None)
        if fn:
            yield from fn(messages)
        else:
            yield self.primary.complete(messages)


def _ollama_cfg(config: dict) -> dict:
    return config.get("ollama", {}) or {}


def _ollama_base(config: dict) -> str:
    return _ollama_cfg(config).get("base_url", "http://localhost:11434")


def backend_for_model(model: str, config: dict, timeout: int = 90) -> LLMBackend:
    """Build the right backend for a model string, pulling the key from the OS
    secret store for cloud providers. Falls back to local Ollama."""
    provider = provider_for_model(model)
    if provider == "ollama_cloud":
        key = get_key("ollama")
        if key:
            return OllamaBackend(OLLAMA_CLOUD_URL, cloud_model_name(model), max(timeout, 120), api_key=key)
        logger.warning("Ollama Cloud model selected but no key set — using the local model")
        model = _ollama_cfg(config).get("model", model)
    elif provider == "anthropic":
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
    return OllamaBackend(base_url=_ollama_base(config), model=model, timeout=timeout,
                         num_ctx=_ollama_cfg(config).get("num_ctx"))


def saved_model() -> str | None:
    """The model picked in the app or with `cx model`, if it can still be used."""
    try:
        import cortex.storage.settings_store as settings_store
        model = (settings_store.section("llm").get("model") or "").strip()
    except Exception:
        return None
    if not model:
        return None
    need = key_provider(model)
    return model if not need or get_key(need) else None


def available_models(config: dict) -> list[dict]:
    """Everything selectable, grouped: this computer, then each cloud provider. Ollama
    Cloud's list needs no key, so its models show even on a machine with no Ollama."""
    groups, local = [], []
    try:
        data = httpx.get(_ollama_base(config) + "/api/tags", timeout=2).json()
        local = sorted(m["name"] for m in data.get("models", []) if "embed" not in m["name"])
    except Exception as e:
        logger.info(f"local Ollama not reachable ({e})")
    keys = configured_providers()
    if keys.get("ollama"):      # the direct route lists these below; don't show them twice
        local = [m for m in local if not m.endswith(("-cloud", ":cloud"))]
    if local:
        groups.append({"label": "On this computer", "models": [{"id": m, "needs_key": False} for m in local]})
    for prov, label in PROVIDER_LABEL.items():
        names = list_provider_models(prov)
        if names:
            groups.append({"label": label, "provider": prov,
                           "models": [{"id": m, "needs_key": not keys.get(prov)} for m in names]})
    return groups


def choose_model(model: str) -> str | None:
    """Remember the model for chat, the terminal and background jobs. -> an error, or None."""
    need = key_provider(model)
    if need and not get_key(need):
        return f"Add your {PROVIDER_LABEL.get(need, need)} key first."
    import cortex.storage.settings_store as settings_store
    settings_store.update_section("llm", {"model": model})
    return None


def rescue_backend(timeout: int = 120) -> LLMBackend | None:
    """A cloud model to finish the job with when the local engine isn't running."""
    if get_key("ollama"):
        names = [n[len(CLOUD_PREFIX):] for n in list_provider_models("ollama")]
        pick = next((m for m in _RESCUE_ORDER if m in names), names[0] if names else _RESCUE_ORDER[0])
        return OllamaBackend(OLLAMA_CLOUD_URL, pick, timeout, api_key=get_key("ollama"))
    if get_key("anthropic"):
        return AnthropicBackend(api_key=get_key("anthropic"), model="claude-sonnet-5-5", timeout=timeout)
    if get_key("openai"):
        return OpenAICompatBackend("https://api.openai.com/v1", get_key("openai"), "gpt-4o", "openai", timeout)
    if get_key("gemini"):
        return OpenAICompatBackend("https://generativelanguage.googleapis.com/v1beta/openai",
                                   get_key("gemini"), "gemini-2.0-flash", "gemini", timeout)
    return None


def create_router(config: dict) -> LLMRouter:
    """Factory: the saved choice, else config.yaml; a local fallback; a cloud rescue."""
    ollama_cfg = _ollama_cfg(config)
    fallback_cfg = config.get("ollama_fallback", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    timeout = ollama_cfg.get("timeout", 90)

    primary_model = saved_model() or ollama_cfg.get("model", "gemma4:e4b")
    primary = backend_for_model(primary_model, config, timeout=timeout)

    fallback = None
    fallback_model = fallback_cfg.get("model") or config.get("fallback")
    if fallback_model:
        fallback = OllamaBackend(base_url=fallback_cfg.get("base_url", base_url), model=fallback_model,
                                 timeout=fallback_cfg.get("timeout", timeout),
                                 num_ctx=ollama_cfg.get("num_ctx"))
    rescue = None
    if isinstance(primary, OllamaBackend) and not primary.api_key:
        rescue = rescue_backend()

    logger.info(f"LLM Router: primary={primary_model}, fallback={fallback_model or 'none'}, "
                f"rescue={getattr(rescue, 'model', 'none')}")
    return LLMRouter(primary=primary, fallback=fallback, rescue=rescue)
