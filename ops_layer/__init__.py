"""Aegis Phase 5 — LLM Ops Layer.

Three responsibilities, all behind the :class:`LLMClient` protocol so the
backing model is swappable (Ollama locally, Gemini API for the demo):

1. :mod:`log_parser`        — noisy log lines → structured graph-update events
2. :mod:`narrator`          — per-action grounded narration
3. :mod:`safety_supervisor` — policy-based action veto

Public surface:

* :class:`LLMClient` / :func:`make_client` — the swappable LLM adapter
* :class:`LogParser` / :class:`GraphEvent`  — structured log extraction
* :class:`Narrator` / :class:`Narration`    — grounded action narration
* :class:`SafetySupervisor` / :class:`VetoResult` — safety veto engine
"""

from ops_layer.llm_client import (
    GeminiClient,
    LLMClient,
    LLMError,
    OllamaClient,
    StubClient,
    make_client,
)
from ops_layer.log_parser import EventType, GraphEvent, LogParser
from ops_layer.narrator import (
    ActionContext,
    DependencyEdge,
    Narration,
    Narrator,
    ServiceSnapshot,
)
from ops_layer.safety_supervisor import (
    DEFAULT_POLICIES,
    Policy,
    SafetySupervisor,
    VetoDecision,
    VetoResult,
)

__all__ = [
    # llm_client
    "LLMClient",
    "LLMError",
    "OllamaClient",
    "GeminiClient",
    "StubClient",
    "make_client",
    # log_parser
    "LogParser",
    "GraphEvent",
    "EventType",
    # narrator
    "Narrator",
    "Narration",
    "ActionContext",
    "ServiceSnapshot",
    "DependencyEdge",
    # safety_supervisor
    "SafetySupervisor",
    "VetoResult",
    "VetoDecision",
    "Policy",
    "DEFAULT_POLICIES",
]
