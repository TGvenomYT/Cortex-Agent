"""
LLM Router — Local-only backend via Ollama.
No OpenAI. No API key is ever read. Everything runs on-device.

Primary: gemma4:e4b (local, JSON mode + vision capable)
Fallback: llama3.1:8b (local, also on this machine)
"""

import os
import json
import logging
import time

import httpx

logger = logging.getLogger("cortex.llm")


class LLMBackend:
    """Abstract base for all LLM backends."""

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        raise NotImplementedError

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Any local Ollama model. No cloud calls, no API keys."""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "gemma4:e4b", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        """
        Send messages to local Ollama /api/chat endpoint.
        schema: a JSON Schema dict (see schemas.py) — when given, Ollama does
        grammar-constrained generation so the output can ONLY be valid JSON
        matching that exact shape. This is materially stronger than
        json_mode=True alone, which only guarantees syntactically valid JSON —
        the model is still free to invent its own keys/shape under json_mode.
        Prefer passing schema whenever the caller parses specific fields.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
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
        logger.info(f"Ollama [{self.model}] responded in {elapsed}ms (local)")

        return text

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """
        Native tool-calling — the model picks a tool (or answers directly)
        using its own tool-use training, instead of authoring a prose JSON
        envelope. Returns {"content": str, "tool_calls": list, "thinking": str}.
        Do not combine with schema= — tool-calling and format-constrained
        generation are separate mechanisms.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "stream": False,
        }

        start = time.time()
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        elapsed = int((time.time() - start) * 1000)

        data = response.json()
        message = data.get("message", {})
        logger.info(f"Ollama [{self.model}] responded in {elapsed}ms (local, tool-calling)")

        return {
            "content": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "thinking": message.get("thinking", ""),
        }

    def stream(self, messages: list[dict]) -> str:
        """
        Stream response from Ollama. Yields text chunks as they arrive.
        Returns the full accumulated text at the end.
        No JSON mode for streaming — we need raw text for progressive display.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }

        start = time.time()
        full_text = ""

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    # Ollama returns "thinking" field for reasoning models, "content" for actual output
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        full_text += content
                        yield content
                except json.JSONDecodeError:
                    continue

        elapsed = int((time.time() - start) * 1000)
        logger.info(f"Ollama [{self.model}] streamed response in {elapsed}ms (local)")
        return full_text


class LLMRouter:
    """Routes to primary local backend, falls back to secondary local backend on failure."""

    def __init__(self, primary: LLMBackend, fallback: LLMBackend = None):
        self.primary = primary
        self.fallback = fallback

    def complete(self, messages: list[dict], json_mode: bool = False, schema: dict = None) -> str:
        """Try primary, fall back on any exception. Both are local — no network egress."""
        try:
            return self.primary.complete(messages, json_mode=json_mode, schema=schema)
        except Exception as e:
            logger.warning(f"Primary local LLM failed: {e}")
            if self.fallback:
                logger.info("Falling back to secondary local LLM...")
                try:
                    return self.fallback.complete(messages, json_mode=json_mode, schema=schema)
                except Exception as e2:
                    logger.error(f"Fallback local LLM also failed: {e2}")
                    raise
            raise

    def complete_with_tools(self, messages: list[dict], tools: list[dict]) -> dict:
        """Try primary, fall back on any exception. Both are local — no network egress."""
        try:
            return self.primary.complete_with_tools(messages, tools)
        except Exception as e:
            logger.warning(f"Primary local LLM failed (tool-calling): {e}")
            if self.fallback:
                logger.info("Falling back to secondary local LLM (tool-calling)...")
                try:
                    return self.fallback.complete_with_tools(messages, tools)
                except Exception as e2:
                    logger.error(f"Fallback local LLM also failed (tool-calling): {e2}")
                    raise
            raise

    def stream(self, messages: list[dict]):
        """Stream response from primary backend. Yields text chunks as they arrive."""
        for chunk in self.primary.stream(messages):
            yield chunk


def create_router(config: dict) -> LLMRouter:
    """Factory: build LLMRouter from config dict. Local Ollama backends only."""
    ollama_cfg = config.get("ollama", {})
    fallback_cfg = config.get("ollama_fallback", {})

    base_url = ollama_cfg.get("base_url", "http://localhost:11434")

    primary = OllamaBackend(
        base_url=base_url,
        model=ollama_cfg.get("model", "gemma4:e4b"),
        timeout=ollama_cfg.get("timeout", 60)
    )

    fallback = None
    fallback_model = fallback_cfg.get("model") or config.get("fallback")
    if fallback_model:
        fallback = OllamaBackend(
            base_url=fallback_cfg.get("base_url", base_url),
            model=fallback_model,
            timeout=fallback_cfg.get("timeout", 60)
        )

    logger.info(f"LLM Router: primary=ollama/{primary.model} (local), "
                f"fallback={'ollama/' + fallback.model if fallback else 'none'} (local)")
    return LLMRouter(primary=primary, fallback=fallback)
