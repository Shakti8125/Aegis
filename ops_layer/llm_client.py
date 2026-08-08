"""LLMClient protocol — Ollama by default, Gemini API via GEMINI_API_KEY for the demo build.

All LLM calls in Aegis go through this adapter so the backing model is
swappable without touching call sites.

Phase 5 — owned by the ops-llm-layer subagent. See PLAN.md section 3.

Architecture
------------
:class:`LLMClient` is a minimal ``Protocol`` — one async method, one sync
wrapper. Every consumer (log_parser, narrator, safety_supervisor) calls
:meth:`complete` with a system prompt and a user message and gets a string
back. Structured output parsing is the consumer's responsibility; the client
is pure text-in/text-out.

Two concrete implementations ship:

* :class:`OllamaClient` — hits ``OLLAMA_HOST`` (default ``http://localhost:11434``).
  No API key, no signup; works with any model pulled locally.
* :class:`GeminiClient` — Google Generative AI SDK via ``GEMINI_API_KEY``.
  Used for the polished demo build where narration quality matters.

:func:`make_client` reads ``LLM_BACKEND`` from the environment and returns
the right one. Call sites never name a concrete class.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)


# ==========================================================================
# Protocol
# ==========================================================================
@runtime_checkable
class LLMClient(Protocol):
    """Minimal text-in/text-out LLM adapter.

    Every consumer passes a system prompt and a user message and gets a
    single string completion back.  No streaming, no tool use — the ops
    layer's prompts are short enough that streaming buys nothing and tool
    use would add coupling we don't need.
    """

    @property
    def model_name(self) -> str:
        """Human-readable identifier for logging (e.g. ``'ollama/llama3.1'``)."""
        ...

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        """Return a single text completion.

        Raises :class:`LLMError` on any transient or permanent failure so
        consumers can fall back to a rule-based default.
        """
        ...


class LLMError(Exception):
    """Raised by any :class:`LLMClient` on failure to produce a completion."""


# ==========================================================================
# Ollama (local, default)
# ==========================================================================
@dataclass
class OllamaClient:
    """Adapter for a locally-running Ollama instance.

    Uses raw ``urllib`` so we have zero additional dependencies — the Ollama
    Python SDK is not in requirements.txt and we don't need it for one POST.
    """

    host: str = ""
    model: str = "llama3.1"
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self.host = self.host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = self.host.rstrip("/")

    @property
    def model_name(self) -> str:
        return f"ollama/{self.model}"

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["message"]["content"]
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc


# ==========================================================================
# Gemini (Google Generative AI API)
# ==========================================================================
@dataclass
class GeminiClient:
    """Adapter for the Google Generative AI REST API.

    Uses raw ``urllib`` to avoid a dependency on the ``google-generativeai``
    SDK. The REST endpoint is stable and the request is one JSON POST.
    """

    api_key: str = ""
    model: str = "gemini-2.0-flash"
    timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise LLMError(
                "GeminiClient requires GEMINI_API_KEY in the environment or "
                "passed explicitly."
            )

    @property
    def model_name(self) -> str:
        return f"gemini/{self.model}"

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"temperature": temperature},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["candidates"][0]["content"]["parts"][0]["text"]
        except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise LLMError(f"Gemini request failed: {exc}") from exc


# ==========================================================================
# Stub client (for testing without an LLM)
# ==========================================================================
@dataclass
class StubClient:
    """Deterministic client that echoes a canned response. For unit tests."""

    response: str = ""
    _calls: list[dict[str, str]] = field(default_factory=list)

    @property
    def model_name(self) -> str:
        return "stub/test"

    def complete(self, system: str, user: str, *, temperature: float = 0.3) -> str:
        self._calls.append({"system": system, "user": user})
        return self.response


# ==========================================================================
# Factory
# ==========================================================================
def make_client(backend: str | None = None, **kwargs: Any) -> LLMClient:
    """Build the right client from ``LLM_BACKEND`` (env) or *backend*.

    ``"ollama"`` (default) → :class:`OllamaClient`
    ``"gemini"``           → :class:`GeminiClient`
    ``"stub"``             → :class:`StubClient`
    """
    backend = (backend or os.environ.get("LLM_BACKEND", "ollama")).lower().strip()
    if backend == "ollama":
        return OllamaClient(**kwargs)
    if backend == "gemini":
        return GeminiClient(**kwargs)
    if backend == "stub":
        return StubClient(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r}. Use 'ollama' or 'gemini'.")
