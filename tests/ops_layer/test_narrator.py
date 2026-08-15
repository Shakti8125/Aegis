"""Tests for ops_layer/narrator.py — grounded narration and fallback templates."""

from __future__ import annotations

import pytest

from ops_layer.llm_client import StubClient
from ops_layer.narrator import (
    ActionContext,
    DependencyEdge,
    Narration,
    Narrator,
    ServiceSnapshot,
    _build_user_prompt,
    _fallback_narrate,
)


# ==========================================================================
# Fixtures
# ==========================================================================
def _make_snapshot(**overrides) -> ServiceSnapshot:
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


def _make_context(action=1, **overrides) -> ActionContext:
    defaults = dict(
        tick=42,
        agent_id="service_3",
        action=action,
        target_service=_make_snapshot(),
    )
    defaults.update(overrides)
    return ActionContext(**defaults)


# ==========================================================================
# ActionContext
# ==========================================================================
class TestActionContext:
    def test_action_name_from_int(self):
        ctx = _make_context(action=1)
        assert ctx.action_name == "restart"

    def test_action_name_from_string(self):
        ctx = _make_context(action="scale_up")
        assert ctx.action_name == "scale_up"

    def test_action_name_noop(self):
        ctx = _make_context(action=0)
        assert ctx.action_name == "no-op"


# ==========================================================================
# Prompt construction
# ==========================================================================
class TestPromptConstruction:
    def test_prompt_contains_service_id(self):
        ctx = _make_context()
        prompt = _build_user_prompt(ctx)
        assert "svc-03-mid" in prompt

    def test_prompt_contains_metrics(self):
        ctx = _make_context()
        prompt = _build_user_prompt(ctx)
        assert "312.5ms" in prompt
        assert "0.082" in prompt or "0.08" in prompt

    def test_prompt_contains_dependencies(self):
        ctx = _make_context(
            dependencies=[
                DependencyEdge(
                    source_id="svc-03-mid",
                    target_id="svc-07-back",
                    p99_latency_ms=500.0,
                    error_rate=0.15,
                ),
            ]
        )
        prompt = _build_user_prompt(ctx)
        assert "svc-07-back" in prompt
        assert "500.0ms" in prompt

    def test_prompt_contains_veto_info(self):
        ctx = _make_context(was_vetoed=True, veto_reason="Deploy window active")
        prompt = _build_user_prompt(ctx)
        assert "VETOED" in prompt
        assert "Deploy window active" in prompt


# ==========================================================================
# Fallback narration
# ==========================================================================
class TestFallbackNarration:
    def test_restart_narration(self):
        ctx = _make_context(action=1)
        text = _fallback_narrate(ctx)
        assert "Restarting" in text
        assert "svc-03-mid" in text
        assert "312" in text

    def test_scale_up_narration(self):
        ctx = _make_context(action=2)
        text = _fallback_narrate(ctx)
        assert "Scaling up" in text

    def test_scale_down_narration(self):
        ctx = _make_context(action=3)
        text = _fallback_narrate(ctx)
        assert "Scaling down" in text

    def test_isolate_narration(self):
        ctx = _make_context(action=4)
        text = _fallback_narrate(ctx)
        assert "Isolating" in text

    def test_reroute_narration(self):
        ctx = _make_context(
            action=5,
            dependencies=[
                DependencyEdge(
                    source_id="svc-03-mid", target_id="svc-07-back"
                )
            ],
        )
        text = _fallback_narrate(ctx)
        assert "Rerouting" in text
        assert "svc-07-back" in text

    def test_noop_healthy(self):
        ctx = _make_context(
            action=0, target_service=_make_snapshot(health=0.98)
        )
        text = _fallback_narrate(ctx)
        assert "healthy" in text.lower() or "no action" in text.lower()

    def test_veto_narration(self):
        ctx = _make_context(
            action=1, was_vetoed=True, veto_reason="Deploy window active"
        )
        text = _fallback_narrate(ctx)
        assert "vetoed" in text.lower()
        assert "Deploy window active" in text


# ==========================================================================
# Narrator (with LLM)
# ==========================================================================
class TestNarrator:
    def test_llm_narration(self):
        stub = StubClient(response='{"text": "Pod restart triggered due to high latency (312ms).", "cited_facts": ["high latency (312ms)"]}')
        narrator = Narrator(llm_client=stub)
        narration = narrator.narrate(_make_context())
        assert "312" in narration.text
        assert narration.model == "stub/test"
        assert narration.grounded is True

    def test_fallback_when_no_llm(self):
        narrator = Narrator(llm_client=None)
        narration = narrator.narrate(_make_context())
        assert "svc-03-mid" in narration.text
        assert narration.model == "fallback/template"

    def test_narration_as_dict(self):
        narrator = Narrator(llm_client=None)
        narration = narrator.narrate(_make_context())
        d = narration.as_dict()
        assert d["tick"] == 42
        assert d["agent_id"] == "service_3"
        assert d["action"] == "restart"
        assert d["target"] == "svc-03-mid"

    def test_batch_narration(self):
        narrator = Narrator(llm_client=None)
        contexts = [_make_context(action=i) for i in range(6)]
        narrations = narrator.narrate_batch(contexts)
        assert len(narrations) == 6

    def test_stats_tracking(self):
        narrator = Narrator(llm_client=None)
        narrator.narrate(_make_context())
        narrator.narrate(_make_context())
        assert narrator.stats["total_narrations"] == 2
        assert narrator.stats["fallback_narrations"] == 2


# ==========================================================================
# Narration
# ==========================================================================
class TestNarration:
    def test_narration_fields(self):
        ctx = _make_context()
        n = Narration(text="test", context=ctx, model="test/model")
        assert n.text == "test"
        assert n.context is ctx
        assert n.model == "test/model"
        assert n.grounded is True
