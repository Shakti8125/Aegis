"""Tests for ops_layer/safety_supervisor.py — policy vetoes and LLM evaluation."""

from __future__ import annotations

import pytest

from ops_layer.llm_client import StubClient
from ops_layer.narrator import ActionContext, DependencyEdge, ServiceSnapshot
from ops_layer.safety_supervisor import (
    DEFAULT_POLICIES,
    Policy,
    SafetySupervisor,
    VetoDecision,
    VetoResult,
)


# ==========================================================================
# Fixtures
# ==========================================================================
def _svc(**overrides) -> ServiceSnapshot:
    defaults = dict(
        service_id="svc-03-mid",
        health=0.45,
        cpu_pct=0.82,
        mem_pct=0.60,
        p99_latency_ms=312.5,
        error_rate=0.08,
        replicas=2,
        ready_replicas=1,
        tier="mid",
        isolated=False,
        sla_violating=True,
    )
    defaults.update(overrides)
    return ServiceSnapshot(**defaults)


def _ctx(action=1, **overrides) -> ActionContext:
    defaults = dict(
        tick=42,
        agent_id="service_3",
        action=action,
        target_service=_svc(),
    )
    defaults.update(overrides)
    return ActionContext(**defaults)


# ==========================================================================
# Deploy window policy
# ==========================================================================
class TestDeployWindowPolicy:
    def test_restart_during_deploy_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="restart",
            active_faults=[{"fault_type": "deploy_window"}],
        )
        result = supervisor.check(ctx)
        assert result.vetoed
        assert result.policy_name == "deploy_window"
        assert "deployment window" in result.reason.lower()

    def test_scale_down_during_deploy_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="scale_down",
            active_faults=[{"fault_type": "deploy_window"}],
        )
        result = supervisor.check(ctx)
        assert result.vetoed

    def test_scale_up_during_deploy_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="scale_up",
            active_faults=[{"fault_type": "deploy_window"}],
        )
        result = supervisor.check(ctx)
        assert not result.vetoed

    def test_no_deploy_window_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(action="restart")
        result = supervisor.check(ctx)
        assert not result.vetoed


# ==========================================================================
# Protected service policy
# ==========================================================================
class TestProtectedServicePolicy:
    def test_isolate_tier0_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="isolate",
            target_service=_svc(tier="front"),
        )
        result = supervisor.check(ctx)
        assert result.vetoed
        assert result.policy_name == "protected_service"
        assert "tier-0" in result.reason.lower() or "gateway" in result.reason.lower()

    def test_isolate_gateway_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="isolate",
            target_service=_svc(tier="gateway"),
        )
        result = supervisor.check(ctx)
        assert result.vetoed

    def test_isolate_mid_tier_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="isolate",
            target_service=_svc(tier="mid"),
        )
        result = supervisor.check(ctx)
        assert not result.vetoed

    def test_restart_tier0_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="restart",
            target_service=_svc(tier="front"),
        )
        result = supervisor.check(ctx)
        assert not result.vetoed


# ==========================================================================
# Concurrent action limit policy
# ==========================================================================
class TestConcurrentActionPolicy:
    def test_scale_up_while_isolated_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="scale_up",
            target_service=_svc(isolated=True),
        )
        result = supervisor.check(ctx)
        assert result.vetoed
        assert result.policy_name == "concurrent_action_limit"

    def test_scale_down_while_isolated_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="scale_down",
            target_service=_svc(isolated=True),
        )
        result = supervisor.check(ctx)
        assert result.vetoed

    def test_restart_while_isolated_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="restart",
            target_service=_svc(isolated=True),
        )
        result = supervisor.check(ctx)
        assert not result.vetoed


# ==========================================================================
# Critical health policy
# ==========================================================================
class TestCriticalHealthPolicy:
    def test_scale_up_at_critical_health_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="scale_up",
            target_service=_svc(health=0.10),
        )
        result = supervisor.check(ctx)
        assert result.vetoed
        assert result.policy_name == "critical_health_restart_only"

    def test_restart_at_critical_health_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="restart",
            target_service=_svc(health=0.10),
        )
        result = supervisor.check(ctx)
        assert not result.vetoed

    def test_noop_at_critical_health_allowed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="no-op",
            target_service=_svc(health=0.10),
        )
        result = supervisor.check(ctx)
        assert not result.vetoed

    def test_normal_health_allows_all_actions(self):
        supervisor = SafetySupervisor()
        for action in ("restart", "scale_up", "scale_down", "isolate", "reroute"):
            ctx = _ctx(action=action, target_service=_svc(health=0.50))
            result = supervisor.check(ctx)
            # Only the protected_service policy could veto isolate on tier="mid"
            # but mid tier is allowed, so all should pass
            if action == "isolate":
                assert not result.vetoed
            else:
                assert not result.vetoed


# ==========================================================================
# No-op always allowed
# ==========================================================================
class TestNoOpAlwaysAllowed:
    def test_noop_never_vetoed(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action=0,  # no-op
            target_service=_svc(health=0.01, isolated=True, tier="front"),
            active_faults=[{"fault_type": "deploy_window"}],
        )
        result = supervisor.check(ctx)
        assert not result.vetoed


# ==========================================================================
# Supervisor state tracking
# ==========================================================================
class TestSupervisorState:
    def test_veto_log(self):
        supervisor = SafetySupervisor()
        ctx = _ctx(
            action="restart",
            active_faults=[{"fault_type": "deploy_window"}],
        )
        supervisor.check(ctx)
        assert supervisor.veto_count == 1
        assert supervisor.veto_log[0].policy_name == "deploy_window"

    def test_stats(self):
        supervisor = SafetySupervisor()
        supervisor.check(_ctx(action="restart"))
        supervisor.check(_ctx(action="no-op"))
        assert supervisor.stats["total_checks"] == 2
        assert supervisor.stats["total_vetoes"] == 0

    def test_batch_check(self):
        supervisor = SafetySupervisor()
        contexts = [
            _ctx(action="restart"),
            _ctx(action="isolate", target_service=_svc(tier="front")),
            _ctx(action="scale_up"),
        ]
        results = supervisor.check_batch(contexts)
        assert len(results) == 3
        assert not results[0].vetoed
        assert results[1].vetoed  # isolate front tier
        assert not results[2].vetoed


# ==========================================================================
# VetoResult
# ==========================================================================
class TestVetoResult:
    def test_as_dict(self):
        ctx = _ctx()
        result = VetoResult(
            decision=VetoDecision.VETO,
            policy_name="test_policy",
            reason="test reason",
            context=ctx,
        )
        d = result.as_dict()
        assert d["decision"] == "veto"
        assert d["policy_name"] == "test_policy"
        assert d["tick"] == 42
        assert d["agent_id"] == "service_3"

    def test_vetoed_property(self):
        assert VetoResult(decision=VetoDecision.VETO).vetoed is True
        assert VetoResult(decision=VetoDecision.ALLOW).vetoed is False


# ==========================================================================
# Custom policies
# ==========================================================================
class TestCustomPolicies:
    def test_custom_policy(self):
        def no_reroute(ctx: ActionContext) -> VetoResult | None:
            if ctx.action_name == "reroute":
                return VetoResult(
                    decision=VetoDecision.VETO,
                    policy_name="no_reroute",
                    reason="Rerouting is disabled.",
                    context=ctx,
                )
            return None

        supervisor = SafetySupervisor(
            policies=[Policy(name="no_reroute", description="No rerouting", check=no_reroute)]
        )
        result = supervisor.check(_ctx(action="reroute"))
        assert result.vetoed
        assert result.policy_name == "no_reroute"

    def test_disabled_policy_skipped(self):
        def always_veto(ctx: ActionContext) -> VetoResult | None:
            return VetoResult(
                decision=VetoDecision.VETO,
                policy_name="always",
                reason="always",
                context=ctx,
            )

        supervisor = SafetySupervisor(
            policies=[
                Policy(name="always", description="always veto", check=always_veto, enabled=False)
            ]
        )
        result = supervisor.check(_ctx(action="restart"))
        assert not result.vetoed


# ==========================================================================
# LLM-based evaluation
# ==========================================================================
class TestLLMEvaluation:
    def test_llm_veto(self):
        stub = StubClient(response="DECISION: VETO | Service is in maintenance mode")
        supervisor = SafetySupervisor(
            policies=[],  # skip rule-based
            llm_client=stub,
            llm_policies=["No actions during maintenance windows"],
        )
        result = supervisor.check(_ctx(action="restart"))
        assert result.vetoed
        assert "maintenance" in result.reason.lower()

    def test_llm_allow(self):
        stub = StubClient(response="DECISION: ALLOW")
        supervisor = SafetySupervisor(
            policies=[],
            llm_client=stub,
            llm_policies=["No actions during maintenance"],
        )
        result = supervisor.check(_ctx(action="restart"))
        assert not result.vetoed

    def test_llm_not_called_without_policies(self):
        stub = StubClient(response="DECISION: VETO | should not happen")
        supervisor = SafetySupervisor(
            policies=[],
            llm_client=stub,
            llm_policies=[],  # empty = skip LLM
        )
        result = supervisor.check(_ctx(action="restart"))
        assert not result.vetoed
        assert len(stub._calls) == 0
