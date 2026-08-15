"""Unit tests for ops_layer/autonomy_engine.py."""

from ops_layer.autonomy_engine import (
    ActionProposal,
    AutonomyDecision,
    AutonomyLevel,
    DecisionOutcome,
    GraduatedAutonomyEngine,
    calculate_policy_entropy,
)
from ops_layer.narrator import ActionContext, ServiceSnapshot
from ops_layer.safety_supervisor import DEFAULT_POLICIES, SafetySupervisor


def test_calculate_policy_entropy():
    # Deterministic policy -> 0 entropy
    assert calculate_policy_entropy([1.0, 0.0, 0.0, 0.0]) == 0.0
    # Uniform 4-choice policy -> log2(4) = 2.0 bits
    assert abs(calculate_policy_entropy([0.25, 0.25, 0.25, 0.25]) - 2.0) < 0.01


def test_autonomy_level_0_manual():
    engine = GraduatedAutonomyEngine(level=AutonomyLevel.LEVEL_0_MANUAL)
    proposal = ActionProposal(agent_id="agent-1", action_name="scale_up", target_service="svc-03", probabilities=[0.9, 0.1])
    decision = engine.evaluate_action(proposal)

    assert decision.outcome == DecisionOutcome.REQUIRES_APPROVAL
    assert "Level 0 (Manual)" in decision.reason
    assert "blocks" in decision.slack_payload


def test_autonomy_level_2_hitl_auto_approve_low_risk():
    engine = GraduatedAutonomyEngine(level=AutonomyLevel.LEVEL_2_HITL, entropy_threshold=1.5)
    proposal = ActionProposal(
        agent_id="agent-1",
        action_name="scale_up",
        target_service="svc-03",
        probabilities=[0.9, 0.1, 0.0],
        risk_score=0.3,
    )
    decision = engine.evaluate_action(proposal)

    assert decision.outcome == DecisionOutcome.APPROVED


def test_autonomy_level_2_hitl_requires_approval_high_entropy():
    engine = GraduatedAutonomyEngine(level=AutonomyLevel.LEVEL_2_HITL, entropy_threshold=1.0)
    # High entropy uniform distribution (~1.58 bits)
    proposal = ActionProposal(
        agent_id="agent-1",
        action_name="scale_up",
        target_service="svc-03",
        probabilities=[0.33, 0.33, 0.34],
        risk_score=0.3,
    )
    decision = engine.evaluate_action(proposal)

    assert decision.outcome == DecisionOutcome.REQUIRES_APPROVAL
    assert "exceeds threshold" in decision.reason


def test_autonomy_level_4_fully_autonomous():
    engine = GraduatedAutonomyEngine(level=AutonomyLevel.LEVEL_4_FULL)
    proposal = ActionProposal(
        agent_id="agent-1",
        action_name="isolate",
        target_service="svc-03",
        probabilities=[0.9, 0.1],
        risk_score=0.9,
    )
    decision = engine.evaluate_action(proposal)

    assert decision.outcome == DecisionOutcome.APPROVED


def test_autonomy_engine_safety_veto_integration():
    supervisor = SafetySupervisor(policies=DEFAULT_POLICIES)
    engine = GraduatedAutonomyEngine(level=AutonomyLevel.LEVEL_4_FULL, safety_supervisor=supervisor)

    # Attempting to isolate a tier-0 gateway service should trigger safety veto
    proposal = ActionProposal(
        agent_id="agent-1",
        action_name="isolate",
        target_service="svc-gateway",
        probabilities=[0.95, 0.05],
    )
    svc = ServiceSnapshot(
        service_id="svc-gateway",
        health=0.4,
        cpu_pct=0.8,
        mem_pct=0.8,
        p99_latency_ms=300.0,
        error_rate=0.05,
        replicas=3,
        ready_replicas=3,
        tier="gateway",
    )
    ctx = ActionContext(tick=10, agent_id="agent-1", action="isolate", target_service=svc)

    decision = engine.evaluate_action(proposal, context=ctx)
    assert decision.outcome == DecisionOutcome.VETOED_BY_SAFETY
    assert "Vetoed by Safety Policy" in decision.reason
