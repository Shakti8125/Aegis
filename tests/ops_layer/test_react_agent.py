"""Unit tests for ops_layer/react_agent.py."""

from ops_layer.llm_client import StubClient
from ops_layer.react_agent import (
    DiagnosticResult,
    ReActDiagnosticAgent,
    ebpf_trace_latency,
    kubectl_get_logs,
    query_neo4j_cypher,
    search_post_mortem_vector_db,
)


def test_tool_executors_standalone():
    neo4j_res = query_neo4j_cypher("MATCH (s:Service) RETURN s")
    assert isinstance(neo4j_res, list)
    assert len(neo4j_res) > 0

    logs = kubectl_get_logs("svc-03", lines=3)
    assert isinstance(logs, list)
    assert len(logs) == 3
    assert "svc-03" in logs[0]

    ebpf = ebpf_trace_latency("svc-01", "svc-03")
    assert isinstance(ebpf, dict)
    assert ebpf["source"] == "svc-01"
    assert ebpf["target"] == "svc-03"
    assert "p99_kernel_latency_ms" in ebpf

    vector_res = search_post_mortem_vector_db("latency spike", k=2)
    assert isinstance(vector_res, list)
    assert len(vector_res) <= 2
    assert "similarity_score" in vector_res[0]


def test_react_agent_fallback_mode():
    agent = ReActDiagnosticAgent(llm_client=None)
    result = agent.diagnose("svc-03", "Elevated p99 latency")

    assert isinstance(result, DiagnosticResult)
    assert result.target_service == "svc-03"
    assert result.model_used == "rule-fallback"
    assert len(result.steps) >= 3
    assert "Fallback Diagnostic Summary" in result.final_answer
    assert len(result.grounded_facts) > 0


def test_react_agent_with_stub_llm():
    stub_response = (
        "Thought: I need to query Neo4j for target service svc-03.\n"
        "Action: query_neo4j_cypher(MATCH (s:Service {id: 'svc-03'}) RETURN s)\n"
    )
    stub_client = StubClient(response=stub_response)

    agent = ReActDiagnosticAgent(llm_client=stub_client, max_steps=2)
    result = agent.diagnose("svc-03", "High memory usage")

    assert isinstance(result, DiagnosticResult)
    assert len(result.steps) > 0


def test_react_agent_final_answer_completion():
    stub_response = (
        "Thought: I have sufficient evidence from logs.\n"
        "Final Answer: Root cause is connection pool exhaustion in svc-03. Restart unhealthiest pod."
    )
    stub_client = StubClient(response=stub_response)

    agent = ReActDiagnosticAgent(llm_client=stub_client, max_steps=3)
    result = agent.diagnose("svc-03", "Connection errors")

    assert isinstance(result, DiagnosticResult)
    assert "Root cause is connection pool exhaustion" in result.final_answer
    assert result.confidence > 0.9
