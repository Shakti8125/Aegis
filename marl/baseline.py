"""Threshold-triggered rule-based controller. The bar MAPPO has to beat.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.

PLAN.md: "Build ``marl/baseline.py`` - a simple threshold-triggered rule-based
controller - before or alongside the MAPPO policy; without it you have no way to
prove the RL is actually doing something."  Which means this file only earns its
keep if it is a *real* opponent.  Three deliberate choices to that end:

1. **Same information.**  The controller reads the *same fixed-length
   observation vector* the decentralized actor gets
   (``env.observation_space(agent)``), plus the same ``infos`` feedback.  It
   never peeks at ``env`` internals, the fault schedule, or the global state.
   Any win MAPPO posts is therefore a win on policy, not on privileged input.

2. **Same interface.**  It satisfies ``marl.evaluation.Controller``, so both are
   scored by one harness on identical seeds.

3. **Tunable, not hand-crippled.**  Every threshold, cooldown and enable flag is
   a field on :class:`BaselineConfig`, and :func:`tune_baseline` sweeps a small
   grid on a *tuning* seed set that is disjoint from the evaluation seeds.  The
   number MAPPO is compared against is the best baseline that search finds, not
   the first set of constants that happened to work.

Rule set, evaluated per service in priority order
-------------------------------------------------
============  ==========================================================
action        fires when
============  ==========================================================
restart       a live replica is down (crashed or unready) and the restart
              cooldown has elapsed.  Cooldown >= ``restart_latency + 1``
              so the controller does not re-restart a pod it is already
              restarting - the single most common way a naive rule-based
              controller burns action cost for nothing.
scale_up      utilisation over ``scale_up_util`` (or latency over
              ``scale_up_latency``), no scale-up already in flight, room
              under ``max_replicas``.
reroute       the worst downstream dependency's error rate is over
              ``reroute_downstream_error``.  Self-disables for a service
              whose ``reroute`` came back invalid (fewer than two CALLS
              edges) - it learns that from ``infos``, exactly the signal
              the RL agent gets through its ``invalid_action`` penalty.
isolate       this service's own error rate is over ``isolate_error`` AND
              its health is under ``isolate_health``.  Threshold is high
              on purpose: ``isolate`` forces error to 1.0 for its
              duration and blocks the env's "recovered" terminal while
              the timer runs, so it is a last resort, not a reflex.
scale_down    utilisation under ``scale_down_util`` and health over
              ``scale_down_health``, sustained for ``scale_down_patience``
              consecutive ticks, with replicas to spare.
no-op         otherwise.
============  ==========================================================
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from simulator.cluster_env import (
    ACTION_ISOLATE,
    ACTION_NOOP,
    ACTION_REROUTE,
    ACTION_RESTART,
    ACTION_SCALE_DOWN,
    ACTION_SCALE_UP,
    ClusterEnv,
)


# --------------------------------------------------------------------------
# Observation layout (mirrors the OBS LAYOUT block in simulator/cluster_env.py)
# --------------------------------------------------------------------------
IDX_HEALTH = 0
IDX_POD_CPU = 1
IDX_POD_MEM = 2
IDX_LATENCY = 3
IDX_ERROR = 4
IDX_UTIL_HALF = 5          # utilisation / 2
IDX_REPLICAS = 6           # replicas / max_replicas
IDX_READY = 7              # ready pods / max_replicas
IDX_ISOLATE_TIMER = 8
IDX_RESTARTS = 9
IDX_DEMAND = 10
IDX_SLA_FLAG = 11
IDX_SCALE_PENDING = 12
IDX_DOWN_FRAC = 13         # down pods / max_replicas

# Neighbourhood block offsets, relative to b = 14 + n_tiers.
OFF_DOWN_MEAN_LAT = 0
OFF_DOWN_MEAN_ERR = 1
OFF_DOWN_MEAN_HEALTH = 2
OFF_DOWN_MAX_LAT = 3
OFF_DOWN_MAX_ERR = 4
OFF_DOWN_MIN_HEALTH = 5
OFF_UP_MEAN_LAT = 6
OFF_UP_MEAN_ERR = 7
OFF_UP_MEAN_HEALTH = 8
OFF_UP_MAX_LAT = 9
OFF_UP_MAX_ERR = 10
OFF_UP_MIN_HEALTH = 11
OFF_NODE_MEAN_CPU = 12
OFF_NODE_MEAN_MEM = 13
OFF_NODE_MAX_CPU = 14
OFF_NODE_MAX_MEM = 15


@dataclass(frozen=True)
class BaselineConfig:
    """Every knob of the rule-based controller. Small on purpose - and tunable."""

    # --- restart -----------------------------------------------------------
    enable_restart: bool = True
    restart_down_frac: float = 0.01   # any down replica at all
    restart_cooldown: int = 5         # >= restart_latency + 1

    # --- scale_up ----------------------------------------------------------
    # Four independent triggers, any of which fires. The health / node-pressure
    # pair matters more than the utilisation pair in this simulator: a node
    # CPU/memory spike raises every co-located pod's resource use without moving
    # utilisation at all, and the only lever that answers it is putting more
    # replicas on other nodes. A baseline without these two rules is a strawman
    # on the node_spike scenario. Set a trigger out of range (health 0.0,
    # pressure > 1.0) to disable it.
    enable_scale_up: bool = True
    scale_up_util: float = 0.85       # actual utilisation, not obs[5]
    scale_up_latency: float = 0.12    # latency / max_latency_ms (~145 ms)
    scale_up_health: float = 0.92     # own service health under this -> add a replica
    scale_up_node_pressure: float = 0.75  # max cpu/mem of a hosting node
    scale_up_cooldown: int = 4        # >= scale_latency + 1

    # --- reroute -----------------------------------------------------------
    enable_reroute: bool = True
    reroute_downstream_error: float = 0.10
    reroute_cooldown: int = 8

    # --- isolate -----------------------------------------------------------
    enable_isolate: bool = True
    isolate_error: float = 0.90
    isolate_health: float = 0.25
    isolate_cooldown: int = 20

    # --- scale_down --------------------------------------------------------
    enable_scale_down: bool = True
    scale_down_util: float = 0.25
    scale_down_health: float = 0.97
    scale_down_patience: int = 12
    scale_down_cooldown: int = 15

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RuleBasedController:
    """Threshold-triggered healing rules over the per-agent observation vector."""

    def __init__(self, config: BaselineConfig | None = None, name: str = "baseline") -> None:
        self.config = config or BaselineConfig()
        self.name = name
        self._agents: list[str] = []
        self._n_tiers = 3
        self._b = 17

    # ---------------------------------------------------------------- setup
    def reset(self, env: ClusterEnv) -> None:
        self._agents = list(env.possible_agents)
        n = len(self._agents)
        self._n_tiers = int(env.cfg.n_tiers)
        self._b = 14 + self._n_tiers
        self._min_replica_frac = float(env.cfg.min_replicas) / float(env.cfg.max_replicas)

        self._cool_restart = np.zeros(n, dtype=np.int32)
        self._cool_scale_up = np.zeros(n, dtype=np.int32)
        self._cool_scale_down = np.zeros(n, dtype=np.int32)
        self._cool_reroute = np.zeros(n, dtype=np.int32)
        self._cool_isolate = np.zeros(n, dtype=np.int32)
        self._quiet_streak = np.zeros(n, dtype=np.int32)
        # Set when a reroute is rejected: that service has < 2 CALLS edges, so
        # rerouting is structurally impossible for it. Learned from `infos`,
        # which is the same feedback the RL agent gets via invalid_action.
        self._reroute_impossible = np.zeros(n, dtype=bool)
        self._last_action = np.zeros(n, dtype=np.int64)

    # ----------------------------------------------------------------- act
    def act(
        self,
        observations: Mapping[str, np.ndarray],
        infos: Mapping[str, Mapping],
        env: ClusterEnv,
    ) -> dict[str, int]:
        cfg = self.config
        agents = self._agents or list(env.agents)
        b = self._b

        self._absorb_feedback(infos, agents)
        for cooldown in (
            self._cool_restart,
            self._cool_scale_up,
            self._cool_scale_down,
            self._cool_reroute,
            self._cool_isolate,
        ):
            np.maximum(cooldown - 1, 0, out=cooldown)

        actions: dict[str, int] = {}
        for i, agent in enumerate(agents):
            o = observations[agent]
            util = 2.0 * float(o[IDX_UTIL_HALF])
            health = float(o[IDX_HEALTH])
            error = float(o[IDX_ERROR])
            latency = float(o[IDX_LATENCY])
            replica_frac = float(o[IDX_REPLICAS])
            action = ACTION_NOOP

            # 1) a replica is down -> bring it back.
            if (
                cfg.enable_restart
                and float(o[IDX_DOWN_FRAC]) > cfg.restart_down_frac
                and self._cool_restart[i] == 0
            ):
                action = ACTION_RESTART
                self._cool_restart[i] = cfg.restart_cooldown

            # 2) saturated, degraded, or sitting on a hot node -> add capacity.
            elif (
                cfg.enable_scale_up
                and (
                    util >= cfg.scale_up_util
                    or latency >= cfg.scale_up_latency
                    or health <= cfg.scale_up_health
                    or max(
                        float(o[b + OFF_NODE_MAX_CPU]), float(o[b + OFF_NODE_MAX_MEM])
                    )
                    >= cfg.scale_up_node_pressure
                )
                and float(o[IDX_SCALE_PENDING]) < 0.5
                and replica_frac < 1.0
                and self._cool_scale_up[i] == 0
            ):
                action = ACTION_SCALE_UP
                self._cool_scale_up[i] = cfg.scale_up_cooldown

            # 3) a dependency is failing -> shift traffic off it.
            elif (
                cfg.enable_reroute
                and not self._reroute_impossible[i]
                and float(o[b + OFF_DOWN_MAX_ERR]) >= cfg.reroute_downstream_error
                and self._cool_reroute[i] == 0
            ):
                action = ACTION_REROUTE
                self._cool_reroute[i] = cfg.reroute_cooldown

            # 4) this service is a black hole -> stop the bleeding. Last resort.
            elif (
                cfg.enable_isolate
                and error >= cfg.isolate_error
                and health <= cfg.isolate_health
                and float(o[IDX_ISOLATE_TIMER]) <= 0.0
                and self._cool_isolate[i] == 0
            ):
                action = ACTION_ISOLATE
                self._cool_isolate[i] = cfg.isolate_cooldown

            # 5) sustained idle capacity -> give it back.
            elif (
                cfg.enable_scale_down
                and util <= cfg.scale_down_util
                and health >= cfg.scale_down_health
                and replica_frac > self._min_replica_frac
                and self._quiet_streak[i] >= cfg.scale_down_patience
                and self._cool_scale_down[i] == 0
            ):
                action = ACTION_SCALE_DOWN
                self._cool_scale_down[i] = cfg.scale_down_cooldown
                self._quiet_streak[i] = 0

            if util <= cfg.scale_down_util and health >= cfg.scale_down_health:
                self._quiet_streak[i] += 1
            else:
                self._quiet_streak[i] = 0

            self._last_action[i] = action
            actions[agent] = int(action)
        return actions

    # ------------------------------------------------------------ feedback
    def _absorb_feedback(
        self, infos: Mapping[str, Mapping], agents: Sequence[str]
    ) -> None:
        """React to the env's rejection signal, the same one the RL agent sees."""
        for i, agent in enumerate(agents):
            info = infos.get(agent)
            if not info or not info.get("invalid_action_now"):
                continue
            last = int(self._last_action[i])
            if last == ACTION_REROUTE:
                self._reroute_impossible[i] = True
            elif last == ACTION_SCALE_UP:
                # Node is full or already at max: back off for a while.
                self._cool_scale_up[i] = max(
                    int(self._cool_scale_up[i]), self.config.scale_up_cooldown * 3
                )
            elif last == ACTION_RESTART:
                self._cool_restart[i] = max(
                    int(self._cool_restart[i]), self.config.restart_cooldown
                )


# --------------------------------------------------------------------------
# Threshold tuning - so the opponent is the best rule set, not the first one
# --------------------------------------------------------------------------
DEFAULT_TUNING_GRID: dict[str, Sequence[Any]] = {
    # 0.0 disables the health trigger, > 1.0 disables the node-pressure trigger,
    # so "no rule" is inside the search space and the sweep can turn a rule off
    # if it hurts.
    "scale_up_health": (0.0, 0.92, 0.99),
    "scale_up_node_pressure": (0.60, 0.75, 1.10),
    "reroute_downstream_error": (0.05, 0.15, 0.40),
    "scale_up_util": (0.70, 0.85),
}


def tune_baseline(
    env_factory: Callable[[], ClusterEnv],
    seeds: Sequence[int],
    scenario: str = "mixed",
    grid: Mapping[str, Sequence[Any]] | None = None,
    base: BaselineConfig | None = None,
    verbose: bool = False,
) -> tuple[BaselineConfig, list[dict[str, Any]]]:
    """Small grid search over thresholds. Returns the best config and the trace.

    Scored on the two metrics PLAN.md grades Phase 4 on, each normalised by the
    *base* config's value so neither unit dominates the other:

        score = mean_ttr / base_ttr + mean_sla_service_ticks / base_sla

    Lower is better; the base config scores 2.0 by construction, so a candidate
    only wins if it is a genuine joint improvement.  ``seeds`` must be
    **disjoint** from the evaluation seeds or the comparison is rigged in the
    baseline's favour (``TrainConfig.tune_seed_base`` vs ``eval_seed_base``).
    """
    from marl.evaluation import evaluate  # local import: avoids a cycle

    grid = dict(grid or DEFAULT_TUNING_GRID)
    base = base or BaselineConfig()
    keys = list(grid)
    trace: list[dict[str, Any]] = []

    base_report, _ = evaluate(env_factory, RuleBasedController(base), seeds, scenario)
    ttr_ref = max(base_report.mean_ttr, 1e-6)
    sla_ref = max(base_report.mean_sla_service_ticks, 1e-6)

    best = 2.0  # the base config's score, by construction
    best_cfg = base

    for values in itertools.product(*(grid[k] for k in keys)):
        candidate = replace(base, **dict(zip(keys, values)))
        report, _ = evaluate(
            env_factory, RuleBasedController(candidate), seeds, scenario
        )
        score = report.mean_ttr / ttr_ref + report.mean_sla_service_ticks / sla_ref
        trace.append(
            {
                "config": dict(zip(keys, values)),
                "score": float(score),
                "mean_sla_service_ticks": report.mean_sla_service_ticks,
                "mean_ttr": report.mean_ttr,
                "recovery_rate": report.recovery_rate,
            }
        )
        if verbose:
            print(
                f"  baseline {dict(zip(keys, values))} -> "
                f"score={score:.3f} sla={report.mean_sla_service_ticks:.1f} "
                f"ttr={report.mean_ttr:.1f}"
            )
        if score < best:
            best, best_cfg = score, candidate

    return best_cfg, trace
