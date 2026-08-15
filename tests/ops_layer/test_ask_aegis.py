"""Unit tests for ops_layer/ask_aegis.py."""

import pytest
from ops_layer.ask_aegis import AskAegisAssistant, AskAegisResponse, validate_cypher_security
from ops_layer.llm_client import StubClient


def test_validate_cypher_security_safe_queries():
    safe_queries = [
        "MATCH (s:Service) RETURN s.id, s.health",
        "MATCH (s:Service) WHERE s.p99_latency_ms > 300 RETURN s ORDER BY s.health ASC",
        "MATCH (a:Service)-[r:CALLS]->(b:Service) RETURN a, r, b LIMIT 10",
        "EXPLAIN MATCH (s:Service) RETURN s",
    ]
    for q in safe_queries:
        is_safe, err = validate_cypher_security(q)
        assert is_safe is True, f"Expected query to be safe: {q} (Error: {err})"


def test_validate_cypher_security_mutating_queries_blocked():
    unsafe_queries = [
        "MATCH (s:Service {id: 'svc-03'}) SET s.health = 1.0 RETURN s",
        "CREATE (s:Service {id: 'malicious-svc'})",
        "MATCH (s:Service) DELETE s",
        "MATCH (s:Service) DETACH DELETE s",
        "MATCH (s:Service {id: 'svc-01'}) REMOVE s.isolated RETURN s",
        "MERGE (s:Service {id: 'svc-03'})",
        "DROP INDEX ON :Service(id)",
        "CALL apoc.trigger.add('test', 'MATCH (n) RETURN n', 'phase')",
    ]
    for q in unsafe_queries:
        is_safe, err = validate_cypher_security(q)
        assert is_safe is False, f"Expected query to be blocked: {q}"
        assert "Security Violation" in err


def test_ask_aegis_fallback_assistant():
    assistant = AskAegisAssistant(llm_client=None)

    res1 = assistant.query("Which services are unhealthy?")
    assert isinstance(res1, AskAegisResponse)
    assert res1.is_safe is True
    assert "s.health < 0.8" in res1.cypher_query
    assert len(res1.raw_results) > 0

    res2 = assistant.query("Show slow dependency calls")
    assert isinstance(res2, AskAegisResponse)
    assert res2.is_safe is True
    assert "CALLS" in res2.cypher_query


def test_ask_aegis_stub_llm_safe_query():
    stub_cypher = "MATCH (s:Service {id: 'svc-03'}) RETURN s.id, s.p99_latency_ms"
    stub_client = StubClient(response=stub_cypher)

    assistant = AskAegisAssistant(llm_client=stub_client)
    res = assistant.query("Get latency for svc-03")

    assert isinstance(res, AskAegisResponse)
    assert res.is_safe is True
    assert res.cypher_query == stub_cypher


def test_ask_aegis_blocks_llm_generated_mutating_query():
    stub_unsafe_cypher = "MATCH (s:Service {id: 'svc-03'}) DELETE s"
    stub_client = StubClient(response=stub_unsafe_cypher)

    assistant = AskAegisAssistant(llm_client=stub_client)
    # The assistant should catch the unsafe LLM output during translation and fall back to a safe query
    cypher = assistant.translate_to_cypher("Delete svc-03")
    is_safe, err = validate_cypher_security(cypher)
    assert is_safe is True  # Fell back to safe read-only query
