"""Unit tests for ops_layer/post_mortem.py."""

from ops_layer.llm_client import StubClient
from ops_layer.post_mortem import (
    FactGroundedPostMortemGenerator,
    IncidentPostMortem,
    verify_against_facts,
)


def test_verify_against_facts_valid():
    pm = IncidentPostMortem(
        incident_id="INC-1",
        title="Incident on svc-03",
        severity="SEV-2",
        timestamp=1000.0,
        root_cause="Degradation in svc-03 and svc-01",
        remediation_steps=["Restarted pod on svc-03"],
    )
    graph_facts = {
        "services": [{"id": "svc-03"}, {"id": "svc-01"}],
        "target_service": "svc-03",
    }
    is_valid, warnings = verify_against_facts(pm, graph_facts)
    assert is_valid is True
    assert len(warnings) == 0


def test_verify_against_facts_hallucinated():
    pm = IncidentPostMortem(
        incident_id="INC-1",
        title="Incident on svc-99",
        severity="SEV-1",
        timestamp=1000.0,
        root_cause="Cascading failure in svc-99 and svc-999",
    )
    graph_facts = {
        "services": [{"id": "svc-01"}, {"id": "svc-02"}],
        "target_service": "svc-01",
    }
    is_valid, warnings = verify_against_facts(pm, graph_facts)
    assert is_valid is False
    assert len(warnings) > 0
    assert "svc-99" in warnings[0] or "svc-999" in warnings[0]


def test_post_mortem_fallback_generator():
    generator = FactGroundedPostMortemGenerator(llm_client=None)
    graph_facts = {
        "target_service": "svc-03",
        "services": [{"id": "svc-03"}, {"id": "svc-05"}],
        "target_metrics": {"health": 0.35, "p99_latency_ms": 420.0, "error_rate": 0.08},
    }
    incident_data = {"incident_id": "INC-2025-99", "severity": "SEV-2"}

    pm = generator.generate(incident_data, graph_facts)
    assert isinstance(pm, IncidentPostMortem)
    assert pm.incident_id == "INC-2025-99"
    assert pm.verification_status is True
    assert pm.model_used == "fallback/fact-grounded-template"
    assert "svc-03" in pm.root_cause


def test_post_mortem_with_stub_llm_valid_json():
    stub_json = (
        '{\n'
        '  "title": "Post-Mortem: svc-03 Degradation",\n'
        '  "severity": "SEV-2",\n'
        '  "root_cause": "Thread starvation in svc-03 DB client pool.",\n'
        '  "remediation_steps": ["Restarted svc-03 pod", "Scaled up replicas"],\n'
        '  "preventative_actions": ["Review thread pool settings"]\n'
        '}'
    )
    stub_client = StubClient(response=stub_json)
    generator = FactGroundedPostMortemGenerator(llm_client=stub_client)

    graph_facts = {
        "target_service": "svc-03",
        "services": [{"id": "svc-03"}],
    }
    incident_data = {"incident_id": "INC-100"}

    pm = generator.generate(incident_data, graph_facts)
    assert isinstance(pm, IncidentPostMortem)
    assert pm.title == "Post-Mortem: svc-03 Degradation"
    assert pm.verification_status is True
    assert pm.model_used == "stub/test"
