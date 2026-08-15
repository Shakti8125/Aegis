"""Unit tests for ops_layer/rag_engine.py."""

from ops_layer.llm_client import StubClient
from ops_layer.rag_engine import (
    GraphSubgraph,
    HybridRAGEngine,
    HybridRAGResponse,
    LogTrace,
    RAGContext,
)


def test_hybrid_rag_retrieval_standalone():
    engine = HybridRAGEngine(llm_client=None)

    subgraph = engine.retrieve_topology_subgraph("svc-03", depth=2)
    assert isinstance(subgraph, GraphSubgraph)
    assert subgraph.center_service == "svc-03"
    assert len(subgraph.nodes) > 0

    logs = engine.retrieve_log_traces(service_id="svc-03", limit=5)
    assert isinstance(logs, list)
    assert len(logs) > 0
    assert isinstance(logs[0], LogTrace)

    context = engine.query_hybrid_rag("Database connection timeout", target_service="svc-03")
    assert isinstance(context, RAGContext)
    assert "--- GRAPH TOPOLOGY SUBGRAPH ---" in context.formatted_prompt
    assert "--- LOG TRACES ---" in context.formatted_prompt


def test_hybrid_rag_synthesis_fallback():
    engine = HybridRAGEngine(llm_client=None)
    response = engine.synthesize_diagnosis("High latency and error rate", target_service="svc-03")

    assert isinstance(response, HybridRAGResponse)
    assert "Hybrid RAG Diagnosis for svc-03" in response.diagnosis
    assert response.model_used == "fallback/template"
    assert len(response.grounded_facts) > 0
    assert "Neo4j Topology Subgraph" in response.sources


def test_hybrid_rag_synthesis_with_stub_llm():
    stub_response = "Grounded RAG Diagnosis: svc-03 is experiencing DB client connection starvation (p99=380ms)."
    stub_client = StubClient(response=stub_response)

    engine = HybridRAGEngine(llm_client=stub_client)
    response = engine.synthesize_diagnosis("Pod crash cascade", target_service="svc-03")

    assert isinstance(response, HybridRAGResponse)
    assert response.diagnosis == stub_response
    assert response.model_used == "stub/test"
    assert len(response.grounded_facts) > 0
