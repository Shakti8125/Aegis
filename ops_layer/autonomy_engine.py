"""Graduated Autonomy Engine for Aegis.

Supports Autonomy Levels 0 to 4:
- Level 0: Manual Only (All actions require human approval)
- Level 1: Advisory (System suggests actions for human execution)
- Level 2: Human-in-the-Loop (Low risk/low entropy auto-approved, high risk requires approval)
- Level 3: High Autonomy (Auto-executes unless risk > 0.7 or high entropy)
- Level 4: Fully Autonomous (All non-vetoed actions auto-approved)

Evaluates policy confidence entropy H(pi(a|s)), assigns action risk scores,
generates interactive Slack Block Kit payloads, and enforces safety thresholds.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Sequence

from ops_layer.narrator import ActionContext
from ops_layer.safety_supervisor import SafetySupervisor, VetoDecision, VetoResult

# Risk mapping for discrete simulator action types
ACTION_RISK_SCORES: dict[str, float] = {
    "no-op": 0.0,
    "scale_up": 0.3,
    "scale_down": 0.3,
    "restart": 0.5,
    "reroute": 0.7,
    "isolate": 0.9,
}


class AutonomyLevel(IntEnum):
    LEVEL_0_MANUAL = 0
    LEVEL_1_ADVISORY = 1
    LEVEL_2_HITL = 2
    LEVEL_3_HIGH = 3
    LEVEL_4_FULL = 4


class DecisionOutcome(str, Enum):
    APPROVED = "approved"
    REQUIRES_APPROVAL = "requires_approval"
    VETOED_BY_SAFETY = "vetoed_by_safety"


@dataclass
class ActionProposal:
    """Proposed MARL action with uncertainty metrics."""

    agent_id: str
    action_name: str
    target_service: str
    probabilities: list[float] = field(default_factory=list)
    entropy: float = 0.0
    risk_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.risk_score:
            self.risk_score = ACTION_RISK_SCORES.get(self.action_name, 0.5)
        if self.probabilities and self.entropy == 0.0:
            self.entropy = calculate_policy_entropy(self.probabilities)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "action_name": self.action_name,
            "target_service": self.target_service,
            "probabilities": self.probabilities,
            "entropy": round(self.entropy, 4),
            "risk_score": self.risk_score,
        }


@dataclass
class AutonomyDecision:
    """The decision made by the Graduated Autonomy Engine."""

    outcome: DecisionOutcome
    proposal: ActionProposal
    autonomy_level: AutonomyLevel
    reason: str
    slack_payload: dict[str, Any] = field(default_factory=dict)
    ui_payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "proposal": self.proposal.as_dict(),
            "autonomy_level": self.autonomy_level.value,
            "reason": self.reason,
            "slack_payload": self.slack_payload,
            "ui_payload": self.ui_payload,
            "timestamp": self.timestamp,
        }


def calculate_policy_entropy(probabilities: Sequence[float], eps: float = 1e-9) -> float:
    """Calculate policy confidence entropy H(pi(a|s)) in bits.

    H(pi) = - sum_i p_i * log2(p_i + eps)
    High entropy = high uncertainty / low confidence.
    """
    if not probabilities:
        return 0.0
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    norm_probs = [p / total for p in probabilities]
    entropy = -sum(p * math.log2(p + eps) for p in norm_probs if p > 0)
    return max(0.0, float(entropy))


class GraduatedAutonomyEngine:
    """Evaluates action safety, policy entropy, and autonomy rules."""

    def __init__(
        self,
        level: AutonomyLevel = AutonomyLevel.LEVEL_2_HITL,
        entropy_threshold: float = 1.5,
        safety_supervisor: SafetySupervisor | None = None,
    ) -> None:
        self.level = level
        self.entropy_threshold = entropy_threshold
        self.safety_supervisor = safety_supervisor

    def evaluate_action(
        self,
        proposal: ActionProposal,
        context: ActionContext | None = None,
    ) -> AutonomyDecision:
        """Evaluate a proposed action against autonomy level and safety policies."""
        # Step 1: Safety Supervisor Veto Check
        if self.safety_supervisor is not None and context is not None:
            veto_result: VetoResult = self.safety_supervisor.check(context)
            if veto_result.vetoed:
                reason = f"Vetoed by Safety Policy [{veto_result.policy_name}]: {veto_result.reason}"
                decision = AutonomyDecision(
                    outcome=DecisionOutcome.VETOED_BY_SAFETY,
                    proposal=proposal,
                    autonomy_level=self.level,
                    reason=reason,
                )
                decision.slack_payload = self.build_slack_approval_payload(proposal, reason, context, decision.outcome)
                decision.ui_payload = self.build_ui_approval_payload(proposal, reason, context, decision.outcome)
                return decision

        # Step 2: Level 0 (Manual) -> All actions require approval
        if self.level == AutonomyLevel.LEVEL_0_MANUAL:
            reason = "Autonomy Level 0 (Manual): All actions require explicit human approval."
            return self._make_decision(DecisionOutcome.REQUIRES_APPROVAL, proposal, reason, context)

        # Step 3: Level 1 (Advisory) -> System suggests, requires approval
        if self.level == AutonomyLevel.LEVEL_1_ADVISORY:
            reason = "Autonomy Level 1 (Advisory): System proposed action for human review."
            return self._make_decision(DecisionOutcome.REQUIRES_APPROVAL, proposal, reason, context)

        # Step 4: Level 2 (HITL) -> Low risk & low entropy auto-approved
        if self.level == AutonomyLevel.LEVEL_2_HITL:
            if proposal.action_name == "no-op":
                reason = "Level 2 HITL: no-op action auto-approved."
                return self._make_decision(DecisionOutcome.APPROVED, proposal, reason, context)

            if proposal.entropy > self.entropy_threshold:
                reason = (
                    f"Level 2 HITL: Policy entropy H={proposal.entropy:.2f} exceeds threshold "
                    f"{self.entropy_threshold:.2f} (high uncertainty)."
                )
                return self._make_decision(DecisionOutcome.REQUIRES_APPROVAL, proposal, reason, context)

            if proposal.risk_score > 0.3:
                reason = f"Level 2 HITL: Action risk score {proposal.risk_score:.2f} (>0.30) requires approval."
                return self._make_decision(DecisionOutcome.REQUIRES_APPROVAL, proposal, reason, context)

            reason = f"Level 2 HITL: Action risk {proposal.risk_score:.2f} and entropy {proposal.entropy:.2f} within auto-approval bounds."
            return self._make_decision(DecisionOutcome.APPROVED, proposal, reason, context)

        # Step 5: Level 3 (High Autonomy) -> Auto approve up to risk 0.7
        if self.level == AutonomyLevel.LEVEL_3_HIGH:
            if proposal.entropy > self.entropy_threshold:
                reason = f"Level 3 High Autonomy: High entropy H={proposal.entropy:.2f} triggers human review."
                return self._make_decision(DecisionOutcome.REQUIRES_APPROVAL, proposal, reason, context)

            if proposal.risk_score > 0.7:
                reason = f"Level 3 High Autonomy: Critical risk score {proposal.risk_score:.2f} (>0.70) requires approval."
                return self._make_decision(DecisionOutcome.REQUIRES_APPROVAL, proposal, reason, context)

            reason = f"Level 3 High Autonomy: Auto-approved action (risk={proposal.risk_score:.2f})."
            return self._make_decision(DecisionOutcome.APPROVED, proposal, reason, context)

        # Step 6: Level 4 (Fully Autonomous) -> Auto approve all non-vetoed actions
        reason = "Autonomy Level 4 (Fully Autonomous): Action automatically approved."
        return self._make_decision(DecisionOutcome.APPROVED, proposal, reason, context)

    def _make_decision(
        self,
        outcome: DecisionOutcome,
        proposal: ActionProposal,
        reason: str,
        context: ActionContext | None = None,
    ) -> AutonomyDecision:
        decision = AutonomyDecision(
            outcome=outcome,
            proposal=proposal,
            autonomy_level=self.level,
            reason=reason,
        )
        decision.slack_payload = self.build_slack_approval_payload(proposal, reason, context, outcome)
        decision.ui_payload = self.build_ui_approval_payload(proposal, reason, context, outcome)
        return decision

    def build_slack_approval_payload(
        self,
        proposal: ActionProposal,
        reason: str,
        context: ActionContext | None = None,
        outcome: DecisionOutcome = DecisionOutcome.REQUIRES_APPROVAL,
    ) -> dict[str, Any]:
        """Generate Slack Block Kit payload for interactive human approval."""
        status_emoji = "⚠️" if outcome == DecisionOutcome.REQUIRES_APPROVAL else ("❌" if outcome == DecisionOutcome.VETOED_BY_SAFETY else "✅")

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{status_emoji} Aegis Remediation Approval Request: {proposal.action_name.upper()}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Target Service:*\n`{proposal.target_service}`"},
                    {"type": "mrkdwn", "text": f"*Action:* `{proposal.action_name}`"},
                    {"type": "mrkdwn", "text": f"*Risk Score:* `{proposal.risk_score:.2f}`"},
                    {"type": "mrkdwn", "text": f"*Policy Entropy:* `{proposal.entropy:.2f} bits`"},
                    {"type": "mrkdwn", "text": f"*Autonomy Level:* `Level {self.level.value}`"},
                    {"type": "mrkdwn", "text": f"*Outcome:* `{outcome.value.upper()}`"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Reason / Policy Context:*\n>{reason}"},
            },
        ]

        if context is not None and context.target_service:
            svc = context.target_service
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Service Health:* `{svc.health:.2f}`"},
                        {"type": "mrkdwn", "text": f"*P99 Latency:* `{svc.p99_latency_ms:.1f}ms`"},
                        {"type": "mrkdwn", "text": f"*Error Rate:* `{svc.error_rate:.3f}`"},
                        {"type": "mrkdwn", "text": f"*Replicas:* `{svc.ready_replicas}/{svc.replicas}`"},
                    ],
                }
            )

        if outcome == DecisionOutcome.REQUIRES_APPROVAL:
            action_id = f"action_{proposal.agent_id}_{proposal.target_service}_{int(time.time())}"
            blocks.append(
                {
                    "type": "actions",
                    "block_id": action_id,
                    "elements": [
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Approve Action", "emoji": True},
                            "style": "primary",
                            "value": f"approve:{proposal.target_service}:{proposal.action_name}",
                            "action_id": "approve_action",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Deny Action", "emoji": True},
                            "style": "danger",
                            "value": f"deny:{proposal.target_service}:{proposal.action_name}",
                            "action_id": "deny_action",
                        },
                        {
                            "type": "button",
                            "text": {"type": "plain_text", "text": "Override Action", "emoji": True},
                            "value": f"override:{proposal.target_service}",
                            "action_id": "override_action",
                        },
                    ],
                }
            )

        return {"blocks": blocks}

    def build_ui_approval_payload(
        self,
        proposal: ActionProposal,
        reason: str,
        context: ActionContext | None = None,
        outcome: DecisionOutcome = DecisionOutcome.REQUIRES_APPROVAL,
    ) -> dict[str, Any]:
        """Generate WebSocket frame payload for 3D/2D Operations Dashboard."""
        return {
            "type": "hitl_approval_request",
            "outcome": outcome.value,
            "autonomy_level": self.level.value,
            "proposal": proposal.as_dict(),
            "reason": reason,
            "target_service": proposal.target_service,
            "action_name": proposal.action_name,
            "risk_score": proposal.risk_score,
            "entropy": proposal.entropy,
            "requires_user_input": outcome == DecisionOutcome.REQUIRES_APPROVAL,
            "timestamp": time.time(),
        }
