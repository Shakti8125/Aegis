"""Reward terms for cluster self-healing.

Every component (SLA violations, action cost, terminal recovery bonus) is
logged separately and never collapsed into a single scalar - separated logs
are how reward hacking gets caught early.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.

Why this module exists at all
-----------------------------
``simulator/cluster_env.py`` computes a *scalar* reward because PettingZoo's
``step`` signature forces one, and it says in as many words that its weights
are a placeholder Phase 4 owns.  Rather than fork the physics, this module
takes ownership by **neutralising the environment's weights**: every training
and evaluation env is constructed with :func:`RewardConfig.env_weight_overrides`
(all weights = 1.0), so ``infos[agent]["reward_components"]`` comes back as
*unit signals*:

===================  ==========================================================
component (env)      value with unit weights
===================  ==========================================================
``sla_violation``    ``-sla_violation_rate``  (demand-weighted bad-request frac)
``latency``          ``-mean(latency / max_latency_ms)``
``availability``     ``+mean(service_health)``      <- note the sign
``action_cost``      ``-ACTION_COST[a]`` (0 when the action was rejected)
``invalid_action``   ``-1`` if this agent's action was illegal this tick
``terminal``         ``+1`` on "recovered", ``-1`` on "collapsed", else 0
===================  ==========================================================

:class:`RewardShaper` then applies the Phase 4 weighting.  The environment
stays the single source of truth for *what happened*; this file is the single
source of truth for *what it is worth*.

Design rationale for the shaping
--------------------------------
1. **Health becomes a shortfall penalty, not a survival bonus.**  The env emits
   ``availability = +mean_health``.  Paid per tick, that rewards *staying alive*:
   a policy that never recovers keeps collecting ~+1/tick, and terminating early
   (which is exactly what we want) throws that stream away.  Classic reward
   hacking.  We re-baseline to ``-w_availability * (1 - mean_health)``, which is
   the same signal shifted by a constant so it is <= 0 everywhere and equals 0
   only for a perfectly healthy cluster.  Now every extra tick spent unhealthy
   costs something and finishing sooner is strictly better - which is the
   time-to-recovery metric we are actually graded on.

2. **Dense terms are bounded to roughly [-1.25, 0] per tick.**  Keeping the
   dense stream small and non-positive means the sparse ``+w_recovered`` bonus
   is a real, visible event in the return rather than rounding error, and the
   critic never has to span a huge range.

3. **SLA is the heaviest dense term** (``w_sla``), because SLA-violation count
   is one of the two headline metrics in PLAN.md section 3 Phase 4.  Latency is
   kept light: it is highly correlated with SLA and double-counting it would let
   the agent trade real availability for cosmetic latency.

4. **Action cost is charged, invalid actions are charged separately.**  The env
   already zeroes ``action_cost`` for a rejected action, so the two never
   double-charge.  Splitting them tells us *why* a policy is expensive: a rising
   ``action_cost`` curve with flat ``sla_violation`` means it is churning the
   cluster for nothing; a rising ``invalid_action`` curve means it never learned
   the preconditions.  That distinction is invisible in a collapsed scalar.

5. **The collapse penalty is deliberately large** (``w_collapsed`` = 60 vs a
   worst case of ~1.25/tick).  Any terminal state ends the negative dense
   stream, so a small collapse penalty would make "destroy the cluster on tick
   20" a *profitable* exploit - it saves up to ~180 ticks of penalty.  60 is
   larger than the discounted worst-case tail at gamma = 0.99, so collapsing is
   never the cheap way out.  In practice collapse is close to unreachable in
   this simulator (mean health < 0.15 for 10 straight ticks); the ``terminal``
   component is logged separately precisely so we can confirm that empirically
   instead of assuming it.

Accounting
----------
:class:`RewardAccumulator` sums each component over a rollout or an episode and
reports them as a dict keyed by component name.  Nothing in this module ever
returns only a scalar: :meth:`RewardShaper.shape` returns the scalar *and* the
per-component breakdown, and the scalar is exactly the sum of the breakdown
(``tests/marl/test_reward.py`` asserts that identity).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np

# Component names, in a fixed order. Everything downstream (logs, JSONL,
# checkpoints, the Phase 6 dashboard) keys off this tuple.
COMPONENT_NAMES: tuple[str, ...] = (
    "sla_violation",
    "latency",
    "availability",
    "action_cost",
    "invalid_action",
    "terminal",
)

# Passed to ClusterEnv(**overrides) so the env's own weighting is the identity
# and `infos[agent]["reward_components"]` carries raw unit signals.
UNIT_ENV_WEIGHTS: dict[str, float] = {
    "w_sla": 1.0,
    "w_latency": 1.0,
    "w_availability": 1.0,
    "w_action_cost": 1.0,
    "w_invalid": 1.0,
    "w_terminal_bonus": 1.0,
    "w_collapse_penalty": 1.0,
}


@dataclass(frozen=True)
class RewardConfig:
    """Phase 4's reward weights. Frozen; serialised next to every checkpoint."""

    # --- dense, per tick (all produce values <= 0) --------------------------
    w_sla: float = 0.60
    w_latency: float = 0.15
    w_availability: float = 0.35
    w_action_cost: float = 0.30
    w_invalid: float = 0.05

    # --- sparse, terminal ---------------------------------------------------
    w_recovered: float = 5.0
    w_collapsed: float = 60.0

    def env_weight_overrides(self) -> dict[str, float]:
        """Kwargs for ``ClusterEnv`` that neutralise its placeholder weights."""
        return dict(UNIT_ENV_WEIGHTS)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


class RewardShaper:
    """Turns the env's unit reward signals into the Phase 4 shaped reward.

    Stateless and vectorised: every method takes and returns arrays shaped
    ``(...,)`` broadcastable over agents, so the same object serves the
    single-env evaluation harness and the batched training rollout.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    # ------------------------------------------------------------------ raw
    @staticmethod
    def raw_from_infos(
        infos: Mapping[str, Mapping], agents: Sequence[str]
    ) -> dict[str, np.ndarray]:
        """Pull ``reward_components`` out of a PettingZoo ``infos`` dict.

        Returns one ``(n_agents,)`` float32 array per component, in
        :data:`COMPONENT_NAMES` order.  The env broadcasts the four cooperative
        components identically across agents and varies only ``action_cost`` and
        ``invalid_action``; we keep all six as full arrays so callers never have
        to care which is which.
        """
        n = len(agents)
        out = {k: np.empty(n, dtype=np.float32) for k in COMPONENT_NAMES}
        for i, agent in enumerate(agents):
            comps = infos[agent]["reward_components"]
            for k in COMPONENT_NAMES:
                out[k][i] = comps[k]
        return out

    # ---------------------------------------------------------------- shape
    def shape(
        self, raw: Mapping[str, np.ndarray]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Apply the Phase 4 weighting.

        Returns ``(total, components)`` where ``total`` is exactly
        ``sum(components.values())``.  The breakdown is always returned - there
        is no scalar-only entry point in this module by design.
        """
        cfg = self.config
        terminal_raw = np.asarray(raw["terminal"], dtype=np.float32)

        comps: dict[str, np.ndarray] = {
            # Already negative in the env; a positive weight preserves the sign.
            "sla_violation": cfg.w_sla * np.asarray(raw["sla_violation"], np.float32),
            "latency": cfg.w_latency * np.asarray(raw["latency"], np.float32),
            # Re-baselined from "+health bonus" to "-health shortfall" (note 1).
            "availability": -cfg.w_availability
            * (1.0 - np.asarray(raw["availability"], np.float32)),
            "action_cost": cfg.w_action_cost
            * np.asarray(raw["action_cost"], np.float32),
            "invalid_action": cfg.w_invalid
            * np.asarray(raw["invalid_action"], np.float32),
            # +1 -> recovery bonus, -1 -> collapse penalty, 0 -> nothing.
            "terminal": (
                np.maximum(terminal_raw, 0.0) * cfg.w_recovered
                + np.minimum(terminal_raw, 0.0) * cfg.w_collapsed
            ),
        }
        total = np.zeros_like(comps["sla_violation"])
        for value in comps.values():
            total = total + value
        return total.astype(np.float32), comps

    def shape_infos(
        self, infos: Mapping[str, Mapping], agents: Sequence[str]
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Convenience: :meth:`raw_from_infos` then :meth:`shape`."""
        return self.shape(self.raw_from_infos(infos, agents))


class RewardAccumulator:
    """Running per-component sums. Never collapses to a single number.

    ``add`` takes the component dict produced by :meth:`RewardShaper.shape`;
    ``mean_per_step`` / ``sum`` report each component under its own key, plus a
    ``total`` key that is the *sum of the reported components* (kept so logs can
    be sanity-checked, not as a replacement for them).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {k: 0.0 for k in COMPONENT_NAMES}
        self._samples = 0

    def add(self, components: Mapping[str, np.ndarray], count: int | None = None) -> None:
        for k in COMPONENT_NAMES:
            self._sums[k] += float(np.sum(components[k]))
        if count is None:
            count = int(np.size(components[COMPONENT_NAMES[0]]))
        self._samples += count

    @property
    def samples(self) -> int:
        return self._samples

    def sums(self) -> dict[str, float]:
        out = dict(self._sums)
        out["total"] = float(sum(self._sums.values()))
        return out

    def mean_per_sample(self) -> dict[str, float]:
        n = max(self._samples, 1)
        out = {k: v / n for k, v in self._sums.items()}
        out["total"] = float(sum(out.values()))
        return out
