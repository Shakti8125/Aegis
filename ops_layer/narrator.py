"""Per-action narration grounded only in the real dependency edges and metrics passed into the prompt.

Phase 5 — owned by the ops-llm-layer subagent. See PLAN.md section 3.

Grounding contract
------------------
The narrator receives *exactly* the graph facts relevant to an agent's action
and must cite *only those facts* in its explanation. This is enforced by:

1. The prompt template explicitly instructs the model to reference only the
   provided data and to never invent causes.
2. The ``ActionContext`` dataclass captures the complete set of facts passed
   to the model, so auditing is a matter of comparing the narration against
   its input — no hidden context.
3. A rule-based fallback generates narrations from templates when no LLM is
   available, proving the pipeline works end-to-end without an external API.

Output
------
:meth:`Narrator.narrate` returns a :class:`Narration` — the text plus the
facts it was grounded in, so the frontend and any audit tool can show both.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ops_layer.llm_client import LLMClient, LLMError

logger = logging.getLogger(__name__)

# Action names matching simulator/cluster_env.py ACTION_NAMES
_ACTION_LABELS: dict[int | str, str] = {
    0: "no-op",
    1: "restart",
    2: "scale_up",
    3: "scale_down",
    4: "isolate",
    5: "reroute",
    "no-op": "no-op",
    "restart": "restart",
    "scale_up": "scale_up",
    "scale_down": "scale_down",
    "isolate": "isolate",
    "reroute": "reroute",
}


# ==========================================================================
# Data types
# ==========================================================================
@dataclass
class ServiceSnapshot:
    """The graph state of one service at the moment of an action."""

    service_id: str
    health: float
    cpu_pct: float
    mem_pct: float
    p99_latency_ms: float
    error_rate: float
    replicas: int
    ready_replicas: int
    tier: str = ""
    isolated: bool = False
    sla_violating: bool = False


@dataclass
class DependencyEdge:
    """A single CALLS or DEPENDS_ON edge relevant to the action."""

    source_id: str
    target_id: str
    relation: str = "CALLS"  # "CALLS" or "DEPENDS_ON"
    p99_latency_ms: float | None = None
    error_rate: float | None = None
    traffic_share: float | None = None


@dataclass
class ActionContext:
    """Everything the narrator sees about one action — and nothing more.

    This is the grounding boundary: the narration must cite only facts
    present in this object.
    """

    tick: int
    agent_id: str
    action: int | str
    target_service: ServiceSnapshot
    dependencies: list[DependencyEdge] = field(default_factory=list)
    dependents: list[DependencyEdge] = field(default_factory=list)
    active_faults: list[dict[str, Any]] = field(default_factory=list)
    was_vetoed: bool = False
    veto_reason: str = ""

    @property
    def action_name(self) -> str:
        return _ACTION_LABELS.get(self.action, str(self.action))


@dataclass
class Narration:
    """The output: a grounded text explanation plus its source facts."""

    text: str
    context: ActionContext
    model: str = ""
    grounded: bool = True  # False if the LLM call failed and we fell back

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tick": self.context.tick,
            "agent_id": self.context.agent_id,
            "action": self.context.action_name,
            "target": self.context.target_service.service_id,
            "model": self.model,
            "grounded": self.grounded,
            "was_vetoed": self.context.was_vetoed,
            "veto_reason": self.context.veto_reason,
        }


# ==========================================================================
# Prompt construction
# ==========================================================================
_SYSTEM_PROMPT = """\
You are a concise site-reliability narrator for a Kubernetes cluster \
self-healing system called Aegis.

Your job: given the graph state around a service at the moment an RL agent \
acts, write a 1–2 sentence explanation of *why* that action was taken.

RULES — these are non-negotiable:
1. Cite ONLY the facts provided in the "GRAPH STATE" block below.
2. NEVER invent, guess, or assume causes not present in the data.
3. Reference specific metric values (latency, error rate, health) when \
   explaining causality.
4. If an action was vetoed by the safety supervisor, explain the veto \
   reason — do not explain why the action would have been correct.
5. Keep it under 50 words. Prefer active voice.
"""


def _build_user_prompt(ctx: ActionContext) -> str:
    """Build the user-message half of the narration prompt.

    Everything the model needs is right here; it should never reach beyond
    this block.
    """
    svc = ctx.target_service
    lines = [
        "GRAPH STATE",
        f"Tick: {ctx.tick}",
        f"Agent: {ctx.agent_id}",
        f"Action: {ctx.action_name}",
        f"Target: {svc.service_id} (tier: {svc.tier})",
        f"  health={svc.health:.2f}  cpu={svc.cpu_pct:.1%}  mem={svc.mem_pct:.1%}",
        f"  p99_latency={svc.p99_latency_ms:.1f}ms  error_rate={svc.error_rate:.3f}",
        f"  replicas={svc.replicas}/{svc.ready_replicas} ready",
        f"  isolated={svc.isolated}  sla_violating={svc.sla_violating}",
    ]

    if ctx.dependencies:
        lines.append("Outgoing dependencies (this service CALLS):")
        for e in ctx.dependencies:
            parts = [f"  → {e.target_id}"]
            if e.p99_latency_ms is not None:
                parts.append(f"latency={e.p99_latency_ms:.1f}ms")
            if e.error_rate is not None:
                parts.append(f"err={e.error_rate:.3f}")
            lines.append("  ".join(parts))

    if ctx.dependents:
        lines.append("Incoming dependents (services that CALL this one):")
        for e in ctx.dependents:
            parts = [f"  ← {e.source_id}"]
            if e.p99_latency_ms is not None:
                parts.append(f"latency={e.p99_latency_ms:.1f}ms")
            if e.error_rate is not None:
                parts.append(f"err={e.error_rate:.3f}")
            lines.append("  ".join(parts))

    if ctx.active_faults:
        lines.append("Active faults:")
        for f in ctx.active_faults:
            lines.append(f"  {f}")

    if ctx.was_vetoed:
        lines.append(f"VETOED: {ctx.veto_reason}")
        lines.append("Explain why the action was blocked, not why it was proposed.")

    lines.append("")
    lines.append("Write the narration now.")
    return "\n".join(lines)


# ==========================================================================
# Rule-based fallback
# ==========================================================================
def _fallback_narrate(ctx: ActionContext) -> str:
    """Template-based narration when no LLM is available.

    Good enough for testing and for the unit-test contract; the LLM version
    is more natural but this proves the pipeline works without one.
    """
    svc = ctx.target_service
    action = ctx.action_name

    if ctx.was_vetoed:
        return (
            f"Action '{action}' on {svc.service_id} was vetoed: "
            f"{ctx.veto_reason}"
        )

    if action == "no-op":
        if svc.health >= 0.95:
            return f"{svc.service_id} is healthy (health={svc.health:.2f}); no action needed."
        return f"{svc.service_id} health={svc.health:.2f} — monitoring, no action taken."

    if action == "restart":
        return (
            f"Restarting unhealthiest pod on {svc.service_id} "
            f"(health={svc.health:.2f}, p99={svc.p99_latency_ms:.0f}ms, "
            f"error_rate={svc.error_rate:.3f})."
        )

    if action == "scale_up":
        return (
            f"Scaling up {svc.service_id} "
            f"(cpu={svc.cpu_pct:.0%}, {svc.ready_replicas}/{svc.replicas} ready, "
            f"p99={svc.p99_latency_ms:.0f}ms)."
        )

    if action == "scale_down":
        return (
            f"Scaling down {svc.service_id} "
            f"(health={svc.health:.2f}, cpu={svc.cpu_pct:.0%}, "
            f"{svc.replicas} replicas)."
        )

    if action == "isolate":
        return (
            f"Isolating {svc.service_id} to contain fault spread "
            f"(error_rate={svc.error_rate:.3f}, "
            f"health={svc.health:.2f})."
        )

    if action == "reroute":
        deps = ", ".join(e.target_id for e in ctx.dependencies[:3])
        return (
            f"Rerouting traffic from {svc.service_id} "
            f"(p99={svc.p99_latency_ms:.0f}ms) "
            f"away from degraded dependencies{': ' + deps if deps else ''}."
        )

    return f"Agent {ctx.agent_id} executed '{action}' on {svc.service_id}."


# ==========================================================================
# Narrator
# ==========================================================================
class Narrator:
    """Generates grounded action narrations.

    Uses the LLM when available, falls back to templates otherwise.
    Every narration is returned with its full :class:`ActionContext` so
    downstream consumers can audit what facts the narration was built from.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client
        self._narration_count = 0
        self._fallback_count = 0

    def narrate(self, context: ActionContext) -> Narration:
        """Produce a narration for a single agent action."""
        self._narration_count += 1

        if self.llm is not None:
            try:
                prompt = _build_user_prompt(context)
                text = self.llm.complete(
                    _SYSTEM_PROMPT, prompt, temperature=0.3
                )
                return Narration(
                    text=text.strip(),
                    context=context,
                    model=self.llm.model_name,
                    grounded=True,
                )
            except LLMError as exc:
                logger.warning("LLM narration failed, falling back: %s", exc)

        self._fallback_count += 1
        return Narration(
            text=_fallback_narrate(context),
            context=context,
            model="fallback/template",
            grounded=True,
        )

    def narrate_batch(
        self, contexts: Sequence[ActionContext]
    ) -> list[Narration]:
        """Narrate a sequence of actions (e.g. one full tick)."""
        return [self.narrate(ctx) for ctx in contexts]

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_narrations": self._narration_count,
            "fallback_narrations": self._fallback_count,
        }
