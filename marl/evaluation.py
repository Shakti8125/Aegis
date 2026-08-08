"""One evaluation harness, used identically for MAPPO and the rule-based baseline.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.

PLAN.md's definition of done for Phase 4 is "MAPPO measurably beats the baseline
on time-to-recovery and SLA-violation count, on multiple fault scenarios".  For
that claim to mean anything, both controllers have to be measured by the *same
code* on the *same episodes*, so everything here is controller-agnostic:

* a :class:`Controller` protocol both implementations satisfy;
* a fixed list of evaluation seeds per scenario.  ``ClusterEnv.reset(seed=s)``
  always reproduces episode 0 of seed ``s``, and the fault schedule for the whole
  episode is drawn at reset *before* any action is taken, so every controller
  faces a byte-identical fault sequence.  The dynamics-noise stream consumes a
  fixed number of draws per tick regardless of the actions taken, so it stays
  aligned too.

Metric definitions (both computed from public env arrays only)
--------------------------------------------------------------
**SLA-violation count** - ``sla_service_ticks``: summed over ticks, the number
of services whose p99 latency is over ``sla_latency_ms`` or whose error rate is
over ``sla_error_rate``.  Service-ticks rather than a rate, because "how much
SLA pain did this episode contain" is the operator-facing number.  The
demand-weighted request fraction is reported alongside as
``sla_request_ticks`` (the integral of ``env.sla_violation_rate``) so a policy
that protects small services while dropping the busiest one cannot hide.

**Time-to-recovery** - ``ttr``: ticks from the first fault firing until the last
tick at which the cluster was unhealthy, where unhealthy means
``min(service_health) < recovery_health`` or ``sla_violation_rate > 0``.  In
words: how long until the cluster is well *and stays well*.  Episodes that end
still unhealthy are reported with ``ttr_censored = True`` and their TTR clamped
to the episode length, and ``recovery_rate`` is reported next to mean TTR so a
censored run can never be mistaken for a fast one.

Ticks before the first fault are excluded from TTR (a controller cannot be
credited or blamed for a period with nothing wrong) but any damage it does there
still shows up in ``sla_service_ticks`` and in ``self_inflicted_ticks``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from simulator.cluster_env import ACTION_NAMES, N_ACTIONS, ClusterEnv
from marl.reward import COMPONENT_NAMES, RewardAccumulator, RewardShaper
from marl.vec_env import sla_violating


class Controller(Protocol):
    """Anything that can drive a ``ClusterEnv``. MAPPO and the baseline both do."""

    name: str

    def reset(self, env: ClusterEnv) -> None:
        """Called once per episode, after ``env.reset``."""

    def act(
        self,
        observations: Mapping[str, np.ndarray],
        infos: Mapping[str, Mapping],
        env: ClusterEnv,
    ) -> dict[str, int]:
        """Return one action per agent in ``env.agents``."""


# --------------------------------------------------------------------------
# Per-episode result
# --------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    seed: int
    length: int
    terminal_reason: str
    recovered: bool
    collapsed: bool
    first_fault_tick: int
    n_faults: int
    ttr: float
    ttr_censored: bool
    sla_service_ticks: int
    sla_request_ticks: float
    self_inflicted_ticks: int
    mean_health: float
    min_health: float
    action_counts: list[int]
    reward_components: dict[str, float] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return float(sum(self.reward_components.values()))


@dataclass
class ScenarioReport:
    """Aggregate over the evaluation seeds of one scenario, for one controller."""

    scenario: str
    controller: str
    episodes: int
    mean_ttr: float
    median_ttr: float
    ttr_censored_frac: float
    mean_sla_service_ticks: float
    mean_sla_request_ticks: float
    recovery_rate: float
    collapse_rate: float
    mean_episode_length: float
    mean_total_reward: float
    mean_reward_components: dict[str, float]
    action_share: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# The harness
# --------------------------------------------------------------------------
def run_episode(
    env: ClusterEnv,
    controller: Controller,
    seed: int,
    shaper: RewardShaper | None = None,
) -> EpisodeResult:
    """Drive one full episode through the *public* PettingZoo API."""
    shaper = shaper or RewardShaper()
    cfg = env.cfg
    observations, infos = env.reset(seed=int(seed))
    controller.reset(env)
    agents = list(env.agents)

    acc = RewardAccumulator()
    action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
    unhealthy_ticks: list[int] = []
    sla_service_ticks = 0
    sla_request_ticks = 0.0
    health_trace: list[float] = []

    while env.agents:
        actions = controller.act(observations, infos, env)
        for value in actions.values():
            if 0 <= int(value) < N_ACTIONS:
                action_counts[int(value)] += 1
        observations, _, _, _, infos = env.step(actions)

        _, components = shaper.shape_infos(infos, agents)
        acc.add(components)

        violating = sla_violating(env)
        sla_service_ticks += int(violating.sum())
        sla_request_ticks += float(env.sla_violation_rate)
        min_health = float(env.svc_health.min())
        health_trace.append(float(env.svc_health.mean()))
        if min_health < cfg.recovery_health or env.sla_violation_rate > 1e-6:
            unhealthy_ticks.append(int(env.t))

    length = int(env.t)
    fired = list(env.fired_faults)
    first_fault_tick = int(fired[0].tick) if fired else 0

    post_fault = [t for t in unhealthy_ticks if t >= first_fault_tick]
    self_inflicted = len([t for t in unhealthy_ticks if t < first_fault_tick])
    if not fired:
        ttr, censored = 0.0, False
    elif not post_fault:
        ttr, censored = 0.0, False
    else:
        last_bad = max(post_fault)
        censored = last_bad >= length
        ttr = float(min(last_bad + 1, length) - first_fault_tick)

    return EpisodeResult(
        seed=int(seed),
        length=length,
        terminal_reason=str(env.terminal_reason),
        recovered=env.terminal_reason == "recovered",
        collapsed=env.terminal_reason == "collapsed",
        first_fault_tick=first_fault_tick,
        n_faults=len(fired),
        ttr=ttr,
        ttr_censored=bool(censored),
        sla_service_ticks=sla_service_ticks,
        sla_request_ticks=sla_request_ticks,
        self_inflicted_ticks=self_inflicted,
        mean_health=float(np.mean(health_trace)) if health_trace else 1.0,
        min_health=float(np.min(health_trace)) if health_trace else 1.0,
        action_counts=action_counts.tolist(),
        reward_components={k: v for k, v in acc.sums().items() if k in COMPONENT_NAMES},
    )


def evaluate(
    env_factory: Callable[[], ClusterEnv],
    controller: Controller,
    seeds: Sequence[int],
    scenario: str,
    shaper: RewardShaper | None = None,
) -> tuple[ScenarioReport, list[EpisodeResult]]:
    """Run ``controller`` over ``seeds`` and aggregate."""
    env = env_factory()
    shaper = shaper or RewardShaper()
    results = [run_episode(env, controller, s, shaper) for s in seeds]
    env.close()
    return summarise(results, scenario, getattr(controller, "name", "controller")), results


def summarise(
    results: Sequence[EpisodeResult], scenario: str, controller: str
) -> ScenarioReport:
    n = max(len(results), 1)
    ttrs = [r.ttr for r in results]
    total_actions = max(sum(sum(r.action_counts) for r in results), 1)
    comps = {
        k: float(np.mean([r.reward_components.get(k, 0.0) for r in results]))
        for k in COMPONENT_NAMES
    }
    return ScenarioReport(
        scenario=scenario,
        controller=controller,
        episodes=len(results),
        mean_ttr=float(np.mean(ttrs)) if ttrs else 0.0,
        median_ttr=float(np.median(ttrs)) if ttrs else 0.0,
        ttr_censored_frac=float(np.mean([r.ttr_censored for r in results])) if results else 0.0,
        mean_sla_service_ticks=float(np.mean([r.sla_service_ticks for r in results])) if results else 0.0,
        mean_sla_request_ticks=float(np.mean([r.sla_request_ticks for r in results])) if results else 0.0,
        recovery_rate=float(np.mean([r.recovered for r in results])) if results else 0.0,
        collapse_rate=float(np.mean([r.collapsed for r in results])) if results else 0.0,
        mean_episode_length=float(np.mean([r.length for r in results])) if results else 0.0,
        mean_total_reward=float(np.mean([r.total_reward for r in results])) if results else 0.0,
        mean_reward_components=comps,
        action_share={
            ACTION_NAMES[a]: sum(r.action_counts[a] for r in results) / total_actions
            for a in range(N_ACTIONS)
        },
    )


# --------------------------------------------------------------------------
# Controller adapters
# --------------------------------------------------------------------------
class PolicyController:
    """Wraps a trained :class:`marl.mappo.MAPPO` as a :class:`Controller`."""

    def __init__(self, policy, name: str = "mappo", deterministic: bool = True) -> None:
        self.policy = policy
        self.name = name
        self.deterministic = deterministic
        self._agents: list[str] = []

    def reset(self, env: ClusterEnv) -> None:
        self._agents = list(env.possible_agents)
        self.policy.eval()

    def act(self, observations, infos, env) -> dict[str, int]:
        obs = np.stack([observations[a] for a in self._agents]).astype(np.float32)
        actions = self.policy.act_single(
            obs, env.state(), deterministic=self.deterministic
        )
        return {a: int(actions[i]) for i, a in enumerate(self._agents)}


class NoOpController:
    """Do-nothing floor. Not the opponent - the reference point under it.

    Reported alongside MAPPO and the baseline so "MAPPO beat the baseline" can be
    read against "and both beat doing nothing", which is the check that catches a
    baseline that is secretly harmful.
    """

    name = "no-op"

    def reset(self, env: ClusterEnv) -> None:
        return None

    def act(self, observations, infos, env) -> dict[str, int]:
        return {a: 0 for a in env.agents}


class RandomController:
    """Uniform random actions - the noise floor, for context in the table."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, env: ClusterEnv) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, observations, infos, env) -> dict[str, int]:
        return {a: int(self._rng.integers(0, N_ACTIONS)) for a in env.agents}


# --------------------------------------------------------------------------
# Comparison table
# --------------------------------------------------------------------------
def _fmt(value: float, width: int = 9, places: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a".rjust(width)
    return f"{value:>{width}.{places}f}"


def format_comparison(reports: Sequence[ScenarioReport], headline: str = "") -> str:
    """Human-readable MAPPO-vs-baseline table, grouped by scenario."""
    lines: list[str] = []
    if headline:
        lines += ["=" * 96, headline, "=" * 96]
    by_scenario: dict[str, list[ScenarioReport]] = {}
    for report in reports:
        by_scenario.setdefault(report.scenario, []).append(report)

    header = (
        f"{'controller':<12}{'TTR':>9}{'TTR med':>9}{'censored':>10}"
        f"{'SLA svc-ticks':>15}{'SLA req-ticks':>15}{'recov%':>9}{'ep len':>9}{'reward':>10}"
    )
    for scenario, group in by_scenario.items():
        lines.append("")
        lines.append(f"-- scenario: {scenario} " + "-" * max(0, 80 - len(scenario)))
        lines.append(header)
        for report in group:
            lines.append(
                f"{report.controller:<12}"
                + _fmt(report.mean_ttr, 9, 1)
                + _fmt(report.median_ttr, 9, 1)
                + _fmt(100.0 * report.ttr_censored_frac, 10, 1)
                + _fmt(report.mean_sla_service_ticks, 15, 1)
                + _fmt(report.mean_sla_request_ticks, 15, 2)
                + _fmt(100.0 * report.recovery_rate, 9, 1)
                + _fmt(report.mean_episode_length, 9, 1)
                + _fmt(report.mean_total_reward, 10, 2)
            )
    return "\n".join(lines)


def format_reward_components(reports: Sequence[ScenarioReport]) -> str:
    """Per-component reward table - never a single collapsed scalar (CLAUDE.md)."""
    lines = [
        "",
        "-- reward components (episode sums, mean over eval episodes) " + "-" * 34,
        f"{'scenario':<20}{'controller':<12}"
        + "".join(f"{k:>16}" for k in COMPONENT_NAMES)
        + f"{'total':>12}",
    ]
    for report in reports:
        comps = report.mean_reward_components
        lines.append(
            f"{report.scenario:<20}{report.controller:<12}"
            + "".join(_fmt(comps.get(k, 0.0), 16, 2) for k in COMPONENT_NAMES)
            + _fmt(sum(comps.values()), 12, 2)
        )
    return "\n".join(lines)


def beats(
    challenger: ScenarioReport, incumbent: ScenarioReport
) -> dict[str, Any]:
    """Did ``challenger`` beat ``incumbent`` on the two PLAN.md metrics?"""
    ttr_delta = incumbent.mean_ttr - challenger.mean_ttr
    sla_delta = incumbent.mean_sla_service_ticks - challenger.mean_sla_service_ticks
    return {
        "scenario": challenger.scenario,
        "ttr_delta": float(ttr_delta),
        "ttr_pct": float(
            100.0 * ttr_delta / incumbent.mean_ttr if incumbent.mean_ttr else 0.0
        ),
        "sla_delta": float(sla_delta),
        "sla_pct": float(
            100.0 * sla_delta / incumbent.mean_sla_service_ticks
            if incumbent.mean_sla_service_ticks
            else 0.0
        ),
        "beats_ttr": bool(ttr_delta > 0),
        "beats_sla": bool(sla_delta > 0),
        "beats_both": bool(ttr_delta > 0 and sla_delta > 0),
    }
