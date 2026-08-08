"""Fault scenarios and a batched wrapper around ``simulator.ClusterEnv``.

Phase 4 - owned by the rl-trainer subagent. See PLAN.md section 3.

Two things live here, both "how do we get environments":

* :data:`SCENARIOS` - named ``ClusterConfig`` overrides that isolate one fault
  family each, plus ``mixed``.  PLAN.md section 3 Phase 4 requires the baseline
  comparison to hold up on *more than one* fault scenario, so the scenario set
  is a first-class object rather than a flag buried in ``train.py``.
* :class:`VecClusterEnv` - N independent ``ClusterEnv`` instances stepped in
  lockstep, with auto-reset.  ``simulator/benchmark.py`` measured the PettingZoo
  dict overhead at roughly zero (~-4%, i.e. inside the noise), so this wrapper
  deliberately uses the *public* ``reset``/``step`` API and never reaches into
  env internals.  Batching exists to amortise the torch forward pass over
  ``n_envs * n_agents`` rows, not to dodge the API.

Auto-reset contract
-------------------
When an env finishes, ``step`` returns the observation/state of the **new**
episode in ``obs``/``state``, and the true final state of the finished episode
in ``final_state`` (valid only where ``done``).  GAE needs that final state to
bootstrap ``V(s_T)`` on a *truncation*; ``mappo.compute_gae`` treats a
``termination`` as a hard 0 instead.  Mixing the two up is the single easiest
way to silently poison a PPO run, which is why the final state is threaded
through explicitly rather than reconstructed.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from simulator.cluster_env import ClusterConfig, ClusterEnv
from simulator.fault_injection import FaultType
from marl.reward import COMPONENT_NAMES, RewardConfig

ALL_FAULTS: tuple[FaultType, ...] = (
    FaultType.POD_CRASH,
    FaultType.NODE_CPU_SPIKE,
    FaultType.NODE_MEM_SPIKE,
    FaultType.NETWORK_PARTITION,
    FaultType.CASCADING_LATENCY,
)

# Named fault scenarios. Each value is a dict of ClusterConfig overrides.
SCENARIOS: dict[str, dict[str, Any]] = {
    "pod_crash": {
        "enabled_faults": (FaultType.POD_CRASH,),
        "n_faults_range": (2, 4),
    },
    "node_spike": {
        "enabled_faults": (FaultType.NODE_CPU_SPIKE, FaultType.NODE_MEM_SPIKE),
        "n_faults_range": (1, 3),
    },
    "partition": {
        "enabled_faults": (FaultType.NETWORK_PARTITION,),
        "n_faults_range": (1, 3),
    },
    "cascading_latency": {
        "enabled_faults": (FaultType.CASCADING_LATENCY,),
        "n_faults_range": (1, 3),
    },
    "mixed": {
        "enabled_faults": ALL_FAULTS,
        "n_faults_range": (1, 4),
    },
}

DEFAULT_EVAL_SCENARIOS: tuple[str, ...] = (
    "pod_crash",
    "node_spike",
    "partition",
    "cascading_latency",
    "mixed",
)


def scenario_overrides(
    scenario: str,
    reward_config: RewardConfig | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """``ClusterConfig`` kwargs for ``scenario``, with Phase 4 reward weights.

    The unit reward weights are folded in here so that *every* env in this
    project - training, evaluation and baseline alike - reports raw component
    signals and ``marl/reward.py`` stays the only place weighting happens.
    """
    if scenario not in SCENARIOS:
        raise KeyError(
            f"unknown scenario {scenario!r}; known: {sorted(SCENARIOS)}"
        )
    cfg = dict(SCENARIOS[scenario])
    cfg.update((reward_config or RewardConfig()).env_weight_overrides())
    cfg.update(extra)
    return cfg


def make_scenario_env(
    scenario: str,
    reward_config: RewardConfig | None = None,
    **extra: Any,
) -> ClusterEnv:
    """Build one ``ClusterEnv`` configured for ``scenario``."""
    return ClusterEnv(**scenario_overrides(scenario, reward_config, **extra))


class VecClusterEnv:
    """``n_envs`` independent ``ClusterEnv``s stepped together, with auto-reset.

    Shapes (``E`` = n_envs, ``N`` = n_agents):

    * ``obs``          ``(E, N, obs_dim)`` float32
    * ``state``        ``(E, state_dim)`` float32
    * ``raw``          dict of ``(E, N)`` float32, one per reward component
    * ``terminated``   ``(E,)`` bool - a real terminal (recovered / collapsed)
    * ``truncated``    ``(E,)`` bool - ``max_cycles`` hit; bootstrap V(s), do
      **not** treat as terminal
    * ``final_state``  ``(E, state_dim)`` float32, meaningful only where done
    """

    def __init__(
        self,
        env_fn: Callable[[], ClusterEnv],
        n_envs: int,
        seeds: Sequence[int] | None = None,
    ) -> None:
        if n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.envs: list[ClusterEnv] = [env_fn() for _ in range(n_envs)]
        self.n_envs = n_envs
        self.agents: list[str] = list(self.envs[0].possible_agents)
        self.n_agents = len(self.agents)
        self.obs_dim = int(self.envs[0].obs_dim)
        self.state_dim = int(self.envs[0].state_dim)
        self.seeds = list(seeds) if seeds is not None else list(range(n_envs))
        if len(self.seeds) != n_envs:
            raise ValueError("len(seeds) must equal n_envs")

        self._obs = np.zeros((n_envs, self.n_agents, self.obs_dim), dtype=np.float32)
        self._state = np.zeros((n_envs, self.state_dim), dtype=np.float32)
        self._final_state = np.zeros((n_envs, self.state_dim), dtype=np.float32)
        self._raw = {
            k: np.zeros((n_envs, self.n_agents), dtype=np.float32)
            for k in COMPONENT_NAMES
        }
        self._terminated = np.zeros(n_envs, dtype=bool)
        self._truncated = np.zeros(n_envs, dtype=bool)
        # Per-episode bookkeeping the trainer surfaces alongside reward curves.
        self._ep_len = np.zeros(n_envs, dtype=np.int64)
        self._ep_sla_service_ticks = np.zeros(n_envs, dtype=np.int64)
        self.episode_stats: list[dict[str, float]] = []

    # ------------------------------------------------------------------ api
    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        for i, env in enumerate(self.envs):
            obs, _ = env.reset(seed=int(self.seeds[i]))
            self._write_obs(i, obs)
            self._state[i] = env.state()
            self._ep_len[i] = 0
            self._ep_sla_service_ticks[i] = 0
        self.episode_stats.clear()
        return self._obs, self._state

    def step(self, actions: np.ndarray) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[str, np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        agents = self.agents
        self._final_state.fill(0.0)
        for i, env in enumerate(self.envs):
            act = {a: int(actions[i, j]) for j, a in enumerate(agents)}
            obs, _, terms, truncs, infos = env.step(act)

            for j, a in enumerate(agents):
                comps = infos[a]["reward_components"]
                for k in COMPONENT_NAMES:
                    self._raw[k][i, j] = comps[k]

            term = bool(terms[agents[0]])
            trunc = bool(truncs[agents[0]])
            self._terminated[i] = term
            self._truncated[i] = trunc
            self._ep_len[i] += 1
            self._ep_sla_service_ticks[i] += int(sla_violating(env).sum())

            if term or trunc:
                self._final_state[i] = env.state()
                self.episode_stats.append(
                    {
                        "length": float(self._ep_len[i]),
                        "sla_service_ticks": float(self._ep_sla_service_ticks[i]),
                        "recovered": float(env.terminal_reason == "recovered"),
                        "collapsed": float(env.terminal_reason == "collapsed"),
                        "truncated": float(env.terminal_reason == "truncated"),
                        "final_mean_health": float(env.svc_health.mean()),
                    }
                )
                self._ep_len[i] = 0
                self._ep_sla_service_ticks[i] = 0
                obs, _ = env.reset()
            self._write_obs(i, obs)
            self._state[i] = env.state()

        return (
            self._obs,
            self._state,
            self._raw,
            self._terminated,
            self._truncated,
            self._final_state,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    # -------------------------------------------------------------- helpers
    def _write_obs(self, i: int, obs: dict[str, np.ndarray]) -> None:
        row = self._obs[i]
        for j, a in enumerate(self.agents):
            row[j] = obs[a]

    def drain_episode_stats(self) -> list[dict[str, float]]:
        out = list(self.episode_stats)
        self.episode_stats.clear()
        return out


def sla_violating(env: ClusterEnv) -> np.ndarray:
    """Per-service SLA-violation mask, using only public env arrays.

    Mirrors ``ClusterEnv._compute_health_and_sla`` exactly (latency over the SLA
    *or* error rate over the SLA).  Recomputed rather than read off the env's
    private ``_sla_violating`` so ``marl/`` never depends on simulator
    internals.
    """
    cfg = env.cfg
    return (env.svc_latency > cfg.sla_latency_ms) | (env.svc_error > cfg.sla_error_rate)
