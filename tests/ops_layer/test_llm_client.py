"""Tests for ops_layer/llm_client.py — protocol, factory, and both concrete clients."""

from __future__ import annotations

import json
import os

import pytest

from ops_layer.llm_client import (
    GeminiClient,
    LLMClient,
    LLMError,
    OllamaClient,
    StubClient,
    make_client,
)


# ==========================================================================
# Protocol compliance
# ==========================================================================
class TestProtocol:
    def test_stub_satisfies_protocol(self):
        client = StubClient(response="hello")
        assert isinstance(client, LLMClient)

    def test_ollama_satisfies_protocol(self):
        client = OllamaClient()
        assert isinstance(client, LLMClient)

    def test_gemini_satisfies_protocol(self):
        # Provide a fake key so __post_init__ doesn't raise
        client = GeminiClient(api_key="fake-key")
        assert isinstance(client, LLMClient)


# ==========================================================================
# StubClient
# ==========================================================================
class TestStubClient:
    def test_returns_configured_response(self):
        client = StubClient(response="test output")
        assert client.complete("sys", "user") == "test output"

    def test_records_calls(self):
        client = StubClient(response="r")
        client.complete("system prompt", "user message")
        client.complete("s2", "u2")
        assert len(client._calls) == 2
        assert client._calls[0] == {"system": "system prompt", "user": "user message"}

    def test_model_name(self):
        client = StubClient()
        assert client.model_name == "stub/test"


# ==========================================================================
# OllamaClient
# ==========================================================================
class TestOllamaClient:
    def test_default_host(self):
        client = OllamaClient()
        assert client.host == "http://localhost:11434"

    def test_env_host(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://custom:1234/")
        client = OllamaClient()
        assert client.host == "http://custom:1234"  # trailing slash stripped

    def test_model_name(self):
        client = OllamaClient(model="mistral")
        assert client.model_name == "ollama/mistral"

    def test_unreachable_raises_llm_error(self):
        client = OllamaClient(host="http://127.0.0.1:1", timeout_s=0.5)
        with pytest.raises(LLMError, match="Ollama request failed"):
            client.complete("sys", "user")


# ==========================================================================
# GeminiClient
# ==========================================================================
class TestGeminiClient:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(LLMError, match="GEMINI_API_KEY"):
            GeminiClient()

    def test_env_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
        client = GeminiClient()
        assert client.api_key == "test-key-123"

    def test_model_name(self):
        client = GeminiClient(api_key="k")
        assert client.model_name == "gemini/gemini-2.0-flash"


# ==========================================================================
# Factory
# ==========================================================================
class TestFactory:
    def test_default_is_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        client = make_client()
        assert isinstance(client, OllamaClient)

    def test_stub_backend(self):
        client = make_client("stub", response="hi")
        assert isinstance(client, StubClient)
        assert client.complete("s", "u") == "hi"

    def test_gemini_backend(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        client = make_client("gemini")
        assert isinstance(client, GeminiClient)

    def test_env_backend(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "stub")
        client = make_client()
        assert isinstance(client, StubClient)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            make_client("nonexistent")
