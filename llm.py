"""
LLM Router — Pluggable LLM backends with fallback.
Primary: GPT-4o-mini via OpenAI API
Fallback: Ollama (any model, any host)
"""

import os
import logging
import time

import httpx
from openai import OpenAI

logger = logging.getLogger("cortex.llm")


class LLMBackend:
    """Abstract base for all LLM backends."""

    def complete(self, messages: list[dict], json_mode: bool = False) -> str:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    """GPT-4o-mini via OpenAI API."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini",
                 max_tokens: int = 500, temperature: float = 0.3):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        """Lazy-init OpenAI client (fails at call time, not import time)."""
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("OpenAI API key not set. Set OPENAI_API_KEY env var or add to config.yaml")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def complete(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send messages to OpenAI, return response text."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        start = time.time()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = int((time.time() - start) * 1000)

        text = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None
        logger.info(f"OpenAI [{self.model}] responded in {elapsed}ms, {tokens} tokens")

        return text


class OllamaBackend(LLMBackend):
    """Any Ollama model. Configurable base URL (local or remote)."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "llama3.1", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict], json_mode: bool = False) -> str:
        """Send messages to Ollama /api/chat endpoint."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        start = time.time()
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        elapsed = int((time.time() - start) * 1000)

        data = response.json()
        text = data["message"]["content"]
        logger.info(f"Ollama [{self.model}] responded in {elapsed}ms")

        return text


class LLMRouter:
    """Routes to primary backend, falls back on failure."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend = None):
        self.primary = primary
        self.fallback = fallback

    def complete(self, messages: list[dict], json_mode: bool = False) -> str:
        """Try primary, fall back on any exception."""
        try:
            return self.primary.complete(messages, json_mode=json_mode)
        except Exception as e:
            logger.warning(f"Primary LLM failed: {e}")
            if self.fallback:
                logger.info("Falling back to secondary LLM...")
                try:
                    return self.fallback.complete(messages, json_mode=json_mode)
                except Exception as e2:
                    logger.error(f"Fallback LLM also failed: {e2}")
                    raise
            raise


def create_router(config: dict) -> LLMRouter:
    """Factory: build LLMRouter from config dict."""
    openai_cfg = config.get("openai", {})
    ollama_cfg = config.get("ollama", {})

    backends = {}

    # Build OpenAI backend
    backends["openai"] = OpenAIBackend(
        api_key=openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", ""),
        model=openai_cfg.get("model", "gpt-4o-mini"),
        max_tokens=openai_cfg.get("max_tokens", 500),
        temperature=openai_cfg.get("temperature", 0.3)
    )

    # Build Ollama backend
    backends["ollama"] = OllamaBackend(
        base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
        model=ollama_cfg.get("model", "llama3.1"),
        timeout=ollama_cfg.get("timeout", 30)
    )

    primary_name = config.get("primary", "openai")
    fallback_name = config.get("fallback")  # can be None

    primary = backends.get(primary_name, backends["openai"])
    fallback = backends.get(fallback_name) if fallback_name else None

    logger.info(f"LLM Router: primary={primary_name}, fallback={fallback_name}")
    return LLMRouter(primary=primary, fallback=fallback)
