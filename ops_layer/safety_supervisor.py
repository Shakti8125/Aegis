"""Vetoes RL actions against operating policies the agent cannot know on its own. Every veto is logged with its reason.

Phase 5 — owned by the ops-llm-layer subagent. See PLAN.md section 3.

Why this exists (the strongest differentiator)
----------------------------------------------
RL agents optimize for a mathematical reward function. They cannot know about
human-defined operational policies like "no restarts during a deploy window"
or "never isolate the primary database service". The safety supervisor is an
agentic LLM layer that sits between the RL decision and the execution:

    RL proposes action → Safety Supervisor checks policies → Allow or Veto

This is the "RL decides, an agentic layer can override" story, and it's
genuinely uncommon in portfolio projects.

Two evaluation paths
--------------------
1. **Rule-based evaluation** — a set of :class:`Policy` objects that encode
   hard constraints (deploy windows, protected services, concurrent action
   limits). These are deterministic, fast, and always run.

2. **LLM evaluation** (optional) — for soft or ambiguous policies that are
   hard to encode as rules. The LLM sees the proposed action, the current
   graph state, and a natural-language policy document and decides whether
   to veto. This path is disabled by default and enabled for the demo.

Every veto is logged with its stated reason, the policy that triggered it,
and the full action context — for both audit and for the dashboard's
incident feed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Sequence

from ops_layer.llm_client import LLMClient, LLMError
from ops_layer.narrator import ActionContext, ServiceSnapshot

logger = logging.getLogger(__name__)


# ==========================================================================
# Types
# ==========================================================================
class VetoDecision(str, Enum):
    ALLOW = "allow"
    VETO = "veto"


@dataclass
class VetoResult:
    """The outcome of a safety check on one proposed action."""

    decision: VetoDecision
    policy_name: str = ""
    reason: str = ""
    context: ActionContext | None = None
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = time.time()

    @property
    def vetoed(self) -> bool:
        return self.decision is VetoDecision.VETO

    def as_dict(self) -> dict[str, Any]:
        d = {
            "decision": self.decision.value,
            "policy_name": self.policy_name,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }
        if self.context is not None:
            d["tick"] = self.context.tick
            d["agent_id"] = self.context.agent_id
            d["action"] = self.context.action_name
            d["target"] = self.context.target_service.service_id
        return d


# ==========================================================================
# Policy protocol and built-in policies
# ==========================================================================
@dataclass
class Policy:
    """A single operational constraint.

    Attributes
    ----------
    name : str
        Human-readable policy name for logging.
    description : str
        What this policy enforces, shown in veto reasons.
    check : callable
        ``(ActionContext) -> VetoResult | None``.  Return a ``VetoResult``
        with ``VetoDecision.VETO`` to block, or ``None`` to allow.
    enabled : bool
        Policies can be toggled without removing them.
    """

    name: str
    description: str
    check: Callable[[ActionContext], VetoResult | None]
    enabled: bool = True


def _deploy_window_policy(ctx: ActionContext) -> VetoResult | None:
    """No restarts or scale-downs during a deploy window.

    In the simulator, a deploy window is signaled by an active fault with
    ``fault_type == "deploy_window"`` in the context. In a real cluster this
    would check a deployment controller or a feature flag.

    For demonstration, we also check if any active fault metadata contains
    a ``deploy_active`` flag.
    """
    if ctx.action_name not in ("restart", "scale_down"):
        return None

    for fault in ctx.active_faults:
        fault_type = fault.get("fault_type", "")
        if "deploy" in str(fault_type).lower():
            return VetoResult(
                decision=VetoDecision.VETO,
                policy_name="deploy_window",
                reason=(
                    f"Cannot {ctx.action_name} {ctx.target_service.service_id} "
                    f"during active deployment window."
                ),
                context=ctx,
            )
    return None


def _protected_service_policy(ctx: ActionContext) -> VetoResult | None:
    """Never isolate tier-0 (frontend/gateway) services.

    Isolating the entry point to the cluster cuts all external traffic — a
    worse outcome than whatever fault the agent was trying to contain.
    """
    if ctx.action_name != "isolate":
        return None

    svc = ctx.target_service
    # Tier-0 services are front, gateway, or edge services
    if svc.tier.lower() in ("front", "gateway", "edge", "tier_0"):
        return VetoResult(
            decision=VetoDecision.VETO,
            policy_name="protected_service",
            reason=(
                f"Cannot isolate {svc.service_id}: it is a {svc.tier} "
                f"(tier-0) service. Isolating the gateway cuts all external traffic."
            ),
            context=ctx,
        )
    return None


def _concurrent_action_limit_policy(ctx: ActionContext) -> VetoResult | None:
    """Prevent scaling actions on services that are already isolated.

    An isolated service has its traffic cut — scaling it up while isolated
    wastes resources with no benefit.
    """
    if ctx.action_name in ("scale_up", "scale_down") and ctx.target_service.isolated:
        return VetoResult(
            decision=VetoDecision.VETO,
            policy_name="concurrent_action_limit",
            reason=(
                f"Cannot {ctx.action_name} {ctx.target_service.service_id} "
                f"while it is isolated. Remove isolation first."
            ),
            context=ctx,
        )
    return None


def _critical_health_restart_only(ctx: ActionContext) -> VetoResult | None:
    """When a service is critically unhealthy, only allow restart or no-op.

    Scaling or rerouting a service at health < 0.2 is unlikely to help and
    adds latency; restarting pods is the only meaningful recovery action.
    """
    svc = ctx.target_service
    if svc.health >= 0.2:
        return None
    if ctx.action_name in ("no-op", "restart"):
        return None

    return VetoResult(
        decision=VetoDecision.VETO,
        policy_name="critical_health_restart_only",
        reason=(
            f"Service {svc.service_id} is critically unhealthy "
            f"(health={svc.health:.2f}). Only 'restart' or 'no-op' "
            f"is permitted at this health level."
        ),
        context=ctx,
    )


# Default policy set
DEFAULT_POLICIES: list[Policy] = [
    Policy(
        name="deploy_window",
        description="No restarts or scale-downs during active deployments",
        check=_deploy_window_policy,
    ),
    Policy(
        name="protected_service",
        description="Never isolate tier-0 gateway services",
        check=_protected_service_policy,
    ),
    Policy(
        name="concurrent_action_limit",
        description="No scaling while service is isolated",
        check=_concurrent_action_limit_policy,
    ),
    Policy(
        name="critical_health_restart_only",
        description="Only restart or no-op for critically unhealthy services",
        check=_critical_health_restart_only,
    ),
]


# ==========================================================================
# LLM-based policy evaluation (optional, for soft policies)
# ==========================================================================
_LLM_SUPERVISOR_SYSTEM = """\
You are a safety supervisor for a Kubernetes self-healing system called Aegis.

An RL agent has proposed an action. You must decide whether to ALLOW or VETO \
it based on the operating policies below.

OPERATING POLICIES:
{policies}

Respond with EXACTLY one line in this format:
DECISION: ALLOW
or
DECISION: VETO | <reason>

Do not explain further. Base your decision ONLY on the policies and the \
provided context.
"""


def _build_llm_supervisor_prompt(
    ctx: ActionContext, policy_descriptions: list[str]
) -> tuple[str, str]:
    """Build system and user prompts for LLM-based policy evaluation."""
    policies_text = "\n".join(f"- {p}" for p in policy_descriptions)
    system = _LLM_SUPERVISOR_SYSTEM.format(policies=policies_text)

    svc = ctx.target_service
    user_lines = [
        f"Proposed action: {ctx.action_name}",
        f"Target: {svc.service_id} (tier: {svc.tier})",
        f"Health: {svc.health:.2f}, CPU: {svc.cpu_pct:.1%}, Mem: {svc.mem_pct:.1%}",
        f"P99 latency: {svc.p99_latency_ms:.1f}ms, Error rate: {svc.error_rate:.3f}",
        f"Replicas: {svc.replicas} ({svc.ready_replicas} ready)",
        f"Isolated: {svc.isolated}, SLA violating: {svc.sla_violating}",
    ]
    if ctx.active_faults:
        user_lines.append(f"Active faults: {ctx.active_faults}")

    return system, "\n".join(user_lines)


# ==========================================================================
# Safety Supervisor
# ==========================================================================
class SafetySupervisor:
    """Evaluates proposed RL actions against operational policies.

    Parameters
    ----------
    policies : list[Policy] | None
        Rule-based policies. Defaults to :data:`DEFAULT_POLICIES`.
    llm_client : LLMClient | None
        If provided, also runs LLM-based soft policy evaluation after
        rule-based checks pass.
    llm_policies : list[str] | None
        Natural-language policy descriptions for the LLM evaluator.
    """

    def __init__(
        self,
        policies: list[Policy] | None = None,
        llm_client: LLMClient | None = None,
        llm_policies: list[str] | None = None,
    ) -> None:
        self.policies = policies if policies is not None else list(DEFAULT_POLICIES)
        self.llm = llm_client
        self.llm_policies = llm_policies or []
        self._veto_log: list[VetoResult] = []
        self._check_count = 0

    def check(self, context: ActionContext) -> VetoResult:
        """Evaluate one proposed action. Returns ALLOW or VETO."""
        self._check_count += 1

        # Skip no-ops — they can never be harmful
        if context.action_name == "no-op":
            return VetoResult(
                decision=VetoDecision.ALLOW,
                policy_name="",
                reason="",
                context=context,
            )

        # Rule-based policies first (deterministic, fast)
        for policy in self.policies:
            if not policy.enabled:
                continue
            result = policy.check(context)
            if result is not None and result.vetoed:
                self._log_veto(result)
                return result

        # LLM-based evaluation (optional, for soft policies)
        if self.llm is not None and self.llm_policies:
            result = self._llm_check(context)
            if result is not None and result.vetoed:
                self._log_veto(result)
                return result

        return VetoResult(
            decision=VetoDecision.ALLOW,
            policy_name="",
            reason="",
            context=context,
        )

    def check_batch(
        self, contexts: Sequence[ActionContext]
    ) -> list[VetoResult]:
        """Evaluate a batch of proposed actions (e.g. one full tick)."""
        return [self.check(ctx) for ctx in contexts]

    @property
    def veto_log(self) -> list[VetoResult]:
        """All vetoes issued, in chronological order."""
        return list(self._veto_log)

    @property
    def veto_count(self) -> int:
        return len(self._veto_log)

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_checks": self._check_count,
            "total_vetoes": len(self._veto_log),
        }

    def _log_veto(self, result: VetoResult) -> None:
        self._veto_log.append(result)
        logger.info(
            "VETO [%s] %s on %s: %s",
            result.policy_name,
            result.context.action_name if result.context else "?",
            result.context.target_service.service_id if result.context else "?",
            result.reason,
        )

    def _llm_check(self, context: ActionContext) -> VetoResult | None:
        """Run the LLM-based soft policy evaluation."""
        try:
            system, user = _build_llm_supervisor_prompt(
                context, self.llm_policies
            )
            response = self.llm.complete(system, user, temperature=0.0)
            response = response.strip().upper()

            if "VETO" in response:
                # Extract reason after the pipe
                reason = ""
                if "|" in response:
                    reason = response.split("|", 1)[1].strip()
                return VetoResult(
                    decision=VetoDecision.VETO,
                    policy_name="llm_policy",
                    reason=reason or "Vetoed by LLM safety evaluation",
                    context=context,
                )

            return None  # ALLOW — no veto
        except LLMError as exc:
            logger.warning("LLM safety check failed, defaulting to ALLOW: %s", exc)
            return None
