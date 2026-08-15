"""Aegis Phase 5 — Expanded LLM Ops Layer.

Public surface:

* :class:`LLMClient` / :func:`make_client` — the swappable LLM adapter
* :class:`LogParser` / :class:`GraphEvent`  — structured log extraction
* :class:`Narrator` / :class:`Narration`    — grounded action narration
* :class:`SafetySupervisor` / :class:`VetoResult` — safety veto engine
* :class:`ReActDiagnosticAgent` / :class:`DiagnosticResult` — ReAct tool-calling agent
* :class:`HybridRAGEngine` / :class:`HybridRAGResponse` — Graph + Log RAG engine
* :class:`GraduatedAutonomyEngine` / :class:`AutonomyLevel` — Autonomy 0-4 decision engine
* :class:`FactGroundedPostMortemGenerator` / :class:`IncidentPostMortem` — zero-hallucination post-mortems
* :class:`AskAegisAssistant` / :class:`AskAegisResponse` — Text-to-Cypher assistant with AST security
"""

from ops_layer.ask_aegis import AskAegisAssistant, AskAegisResponse, validate_cypher_security
from ops_layer.autonomy_engine import (
    ACTION_RISK_SCORES,
    ActionProposal,
    AutonomyDecision,
    AutonomyLevel,
    DecisionOutcome,
    GraduatedAutonomyEngine,
    calculate_policy_entropy,
)
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
from ops_layer.post_mortem import (
    FactGroundedPostMortemGenerator,
    IncidentPostMortem,
    verify_against_facts,
)
from ops_layer.rag_engine import (
    GraphSubgraph,
    HybridRAGEngine,
    HybridRAGResponse,
    LogTrace,
    RAGContext,
)
from ops_layer.react_agent import (
    DiagnosticResult,
    ReActDiagnosticAgent,
    ReActStep,
    ebpf_trace_latency,
    kubectl_get_logs,
    query_neo4j_cypher,
    search_post_mortem_vector_db,
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
    # react_agent
    "ReActDiagnosticAgent",
    "DiagnosticResult",
    "ReActStep",
    "query_neo4j_cypher",
    "kubectl_get_logs",
    "ebpf_trace_latency",
    "search_post_mortem_vector_db",
    # rag_engine
    "HybridRAGEngine",
    "GraphSubgraph",
    "LogTrace",
    "RAGContext",
    "HybridRAGResponse",
    # autonomy_engine
    "GraduatedAutonomyEngine",
    "AutonomyLevel",
    "DecisionOutcome",
    "ActionProposal",
    "AutonomyDecision",
    "calculate_policy_entropy",
    "ACTION_RISK_SCORES",
    # post_mortem
    "FactGroundedPostMortemGenerator",
    "IncidentPostMortem",
    "verify_against_facts",
    # ask_aegis
    "AskAegisAssistant",
    "AskAegisResponse",
    "validate_cypher_security",
]

