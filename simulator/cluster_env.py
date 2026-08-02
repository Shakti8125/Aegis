"""Simulated cluster as a PettingZoo ParallelEnv (reset/step/observation_space/action_space).

Phase 1 - owned by the sim-engineer subagent. See PLAN.md section 3.

One agent per Service
---------------------
Four of the six actions PLAN.md section 3 Phase 4 specifies are service-level
(``scale_up``/``scale_down`` are replica-count ops, ``reroute`` shifts traffic
between dependency edges, ``isolate`` drops a service's traffic); ``restart``
maps to "restart this service's unhealthiest pod".  So the agent set is the
service set, and **``env.agents`` is constant for the whole episode** - no
agent death, no ragged batches, no index remapping for Phase 4.  It is emptied
only at episode end, because the PettingZoo API requires that.

Implementation shape
--------------------
Struct-of-arrays throughout.  Pods live in a fixed ``(n_services, max_replicas)``
grid with an ``alive`` mask, so scaling flips bits instead of reallocating and
observation shapes never move.  Service topology is a dense
``n_services x n_services`` float32 matrix, which turns cascade propagation into
a matmul instead of a graph traversal.  At the planned 12-30 services dense
beats sparse on both speed and readability; past roughly 50 services the dense
adjacency is the thing to revisit.

Two observation surfaces
------------------------
* :meth:`observation_space` - a fixed-length ``Box`` vector per agent, so Phase 1
  stands alone and Phase 4 can run a no-GNN ablation.
* :meth:`graph_snapshot` - nodes/edges in the exact PLAN.md section 3 Phase 2
  schema shape, so Phase 2 ingestion and Phase 3's encoder can wrap the env
  without it changing.

Reward
------
PettingZoo's ``step`` forces a scalar, so the env returns the scalar *and* the
breakdown in ``infos[agent]["reward_components"]`` - the CLAUDE.md
non-negotiable.  **The weights below are a deliberate placeholder.  Reward
shaping is Phase 4's job (marl/reward.py); tuning it here would be guessing.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv

from simulator.fault_injection import (
    FaultEvent,
    FaultType,
    apply_fault,
    build_fault_schedule,
)
from simulator.topology_generator import Topology, generate_topology, tier_names

# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------
ACTION_NOOP = 0
ACTION_RESTART = 1
ACTION_SCALE_UP = 2
ACTION_SCALE_DOWN = 3
ACTION_ISOLATE = 4
ACTION_REROUTE = 5
N_ACTIONS = 6

ACTION_NAMES: tuple[str, ...] = (
    "no-op",
    "restart",
    "scale_up",
    "scale_down",
    "isolate",
    "reroute",
)

# Operational cost of each action, charged only when the action actually fires.
ACTION_COST = np.array([0.0, 0.15, 0.25, 0.10, 0.50, 0.20], dtype=np.float32)

# Which cached "is anything running" flag each fault type switches on. The
# flags let the per-tick timer pass skip whole categories without an .any() scan.
_FAULT_FLAG: dict[FaultType, str] = {
    FaultType.POD_CRASH: "_down_on",
    FaultType.NODE_CPU_SPIKE: "_node_pressure_on",
    FaultType.NODE_MEM_SPIKE: "_node_pressure_on",
    FaultType.NETWORK_PARTITION: "_edge_fault_on",
    FaultType.CASCADING_LATENCY: "_svc_lat_fault_on",
}

DEFAULT_ENABLED_FAULTS: tuple[FaultType, ...] = (
    FaultType.POD_CRASH,
    FaultType.NODE_CPU_SPIKE,
    FaultType.NODE_MEM_SPIKE,
    FaultType.NETWORK_PARTITION,
    FaultType.CASCADING_LATENCY,
)


@dataclass(frozen=True)
class ClusterConfig:
    """Every knob of the simulated cluster. Frozen so a config can't drift mid-run."""

    # ---- topology ----------------------------------------------------------
    n_services: int = 12
    n_nodes: int = 6
    n_tiers: int = 3
    max_replicas: int = 4
    init_replicas: int = 2
    min_replicas: int = 1
    max_deps_per_service: int = 3
    call_edge_prob: float = 0.85
    allow_skip_tier_deps: bool = False
    node_pod_capacity: int | None = None
    target_utilization: float = 0.45
    external_load_range: tuple[float, float] = (20.0, 60.0)
    internal_load_range: tuple[float, float] = (0.0, 4.0)
    base_latency_ms_range: tuple[float, float] = (5.0, 25.0)
    fixed_topology: bool = True

    # ---- episode -----------------------------------------------------------
    max_cycles: int = 200

    # ---- action latencies --------------------------------------------------
    restart_latency: int = 3
    scale_latency: int = 2
    isolate_duration: int = 5
    reroute_fraction: float = 0.5

    # ---- dynamics ----------------------------------------------------------
    pod_base_cpu: float = 0.08
    pod_cpu_per_util: float = 0.80
    pod_base_mem: float = 0.15
    pod_mem_per_util: float = 0.55
    health_knee: float = 0.80
    noise_std: float = 0.01
    latency_alpha: float = 2.5
    cascade_decay: float = 0.80
    error_decay: float = 0.70
    error_slope: float = 1.50
    timeout_latency_ms: float = 1000.0
    max_latency_ms: float = 1200.0
    sla_latency_ms: float = 250.0
    sla_error_rate: float = 0.05
    isolate_loss_factor: float = 0.30

    # ---- faults ------------------------------------------------------------
    enabled_faults: tuple[FaultType, ...] = DEFAULT_ENABLED_FAULTS
    n_faults_range: tuple[int, int] = (1, 4)
    fault_start_tick: int = 5
    fault_window_frac: float = 0.6
    pod_crash_duration_range: tuple[int, int] = (20, 60)
    node_spike_duration_range: tuple[int, int] = (15, 40)
    node_spike_magnitude_range: tuple[float, float] = (0.30, 0.60)
    partition_duration_range: tuple[int, int] = (10, 30)
    partition_magnitude_range: tuple[float, float] = (0.40, 0.90)
    cascade_duration_range: tuple[int, int] = (10, 30)
    cascade_magnitude_ms_range: tuple[float, float] = (150.0, 500.0)

    # ---- termination -------------------------------------------------------
    recovery_health: float = 0.95
    collapse_health: float = 0.15
    collapse_patience: int = 10

    # ---- reward weights (PLACEHOLDER - Phase 4 owns these) -----------------
    w_sla: float = 1.00
    w_latency: float = 0.30
    w_availability: float = 0.50
    w_action_cost: float = 1.00
    w_invalid: float = 0.05
    w_terminal_bonus: float = 10.0
    w_collapse_penalty: float = 10.0


class ClusterEnv(ParallelEnv):
    """A seedable, CPU-fast cluster-healing environment (PettingZoo ParallelEnv)."""

    metadata: dict[str, Any] = {
        "render_modes": ["human", "ansi"],
        "name": "aegis_cluster_v0",
        "is_parallelizable": True,
    }

    # ---------------------------------------------------------------- setup
    def __init__(
        self,
        config: ClusterConfig | None = None,
        render_mode: str | None = None,
        **overrides: Any,
    ) -> None:
        cfg = config or ClusterConfig()
        if overrides:
            cfg = ClusterConfig(**{**cfg.__dict__, **overrides})
        self.cfg = cfg
        self.render_mode = render_mode

        self.n_services = int(cfg.n_services)
        self.n_nodes = int(cfg.n_nodes)
        self.max_replicas = int(cfg.max_replicas)
        # Mutable on purpose: pettingzoo's parallel_api_test overrides it.
        self.max_cycles = int(cfg.max_cycles)

        self.possible_agents: list[str] = [
            f"service_{i}" for i in range(self.n_services)
        ]
        self.agents: list[str] = list(self.possible_agents)
        self.agent_name_to_index = {n: i for i, n in enumerate(self.possible_agents)}

        # 35 fixed slots + a tier one-hot. See the OBS LAYOUT block below.
        self.obs_dim = 35 + int(cfg.n_tiers)
        self.state_dim = 10 * self.n_services + 3 * self.n_nodes + 5

        obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32
        )
        act_space = spaces.Discrete(N_ACTIONS)
        # Identity matters: pettingzoo asserts space objects are the same object.
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}
        self.state_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )

        self._master_seed: int | None = None
        self._reset_count: int = 0
        self._topology_cache_key: tuple | None = None
        self.topology: Topology | None = None
        self.fault_schedule: list[FaultEvent] = []
        self.active_faults: list[FaultEvent] = []
        self._active_expiry: list[int] = []
        self.t = 0

        self._action_buf = np.zeros(self.n_services, dtype=np.int64)
        self._rng_dyn = np.random.default_rng(0)

    # ------------------------------------------------------------ API spaces
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ------------------------------------------------------------ seeding
    def _seed_streams(self) -> tuple[np.random.Generator, np.random.Generator, tuple]:
        """Split the master seed into topology / fault / dynamics streams.

        ``SeedSequence(seed).spawn(2)`` separates topology from episode
        randomness, exactly as the approved plan specifies; the episode child is
        then split again into fault-schedule and dynamics-noise streams so that
        adding noise never perturbs the fault draw.

        Reset index ``k`` selects deterministic grandchildren, so ``reset(seed=s)``
        always reproduces episode 0 while a bare ``reset()`` advances to the next
        episode reproducibly.
        """
        base = np.random.SeedSequence(self._master_seed)
        topo_root, episode_root = base.spawn(2)
        k = self._reset_count

        if self.cfg.fixed_topology:
            topo_ss = topo_root
            topo_key = (self._master_seed, "fixed")
        else:
            topo_ss = topo_root.spawn(k + 1)[-1]
            topo_key = (self._master_seed, k)

        fault_ss, dyn_ss = episode_root.spawn(2 * (k + 1))[-2:]
        self._rng_dyn = np.random.Generator(np.random.PCG64(dyn_ss))
        return (
            np.random.Generator(np.random.PCG64(topo_ss)),
            np.random.Generator(np.random.PCG64(fault_ss)),
            topo_key,
        )

    # ------------------------------------------------------------ reset
    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._master_seed = int(seed)
            self._reset_count = 0
        elif self._master_seed is None:
            self._master_seed = int(np.random.SeedSequence().generate_state(1)[0])
            self._reset_count = 0
        else:
            self._reset_count += 1

        rng_topo, rng_fault, topo_key = self._seed_streams()

        if self.topology is None or topo_key != self._topology_cache_key:
            self.topology = generate_topology(self.cfg, rng_topo)
            self._topology_cache_key = topo_key
            self._bind_topology(self.topology)

        self.agents = list(self.possible_agents)
        self.t = 0
        self._init_runtime_state()

        self.fault_schedule = build_fault_schedule(
            self.topology, self.cfg, self.max_cycles, rng_fault
        )
        self._next_fault = 0
        self.active_faults = []
        self._active_expiry = []
        self.fired_faults: list[FaultEvent] = []

        self._settle()
        self._terminated = False
        self._truncated = False
        self.terminal_reason: str | None = None
        self._collapse_streak = 0
        self._invalid_counts = np.zeros(self.n_services, dtype=np.int32)
        self._invalid_now = np.zeros(self.n_services, dtype=bool)
        self._last_action = np.zeros(self.n_services, dtype=np.int64)
        self._rewards = np.zeros(self.n_services, dtype=np.float32)
        self._reward_components = {
            k: np.zeros(self.n_services, dtype=np.float32)
            for k in (
                "sla_violation",
                "latency",
                "availability",
                "action_cost",
                "invalid_action",
                "terminal",
            )
        }
        self._reward_shared = {
            k: 0.0 for k in ("sla_violation", "latency", "availability", "terminal")
        }
        self._build_obs()

        observations = {n: self._obs[i] for i, n in enumerate(self.possible_agents)}
        infos = {
            n: {
                "reward_components": {k: 0.0 for k in self._reward_components},
                "invalid_action": 0,
                "invalid_action_now": False,
                "action": 0,
            }
            for n in self.possible_agents
        }
        return observations, infos

    def _bind_topology(self, topo: Topology) -> None:
        """Cache the read-only views the hot loop touches."""
        self._calls_mask = topo.calls_mask
        self._calls_bool = topo.calls_mask > 0
        self._calls_deg = np.maximum(topo.calls_mask.sum(1), 1.0).astype(np.float32)
        self._called_by = topo.calls_mask.T.copy()
        self._called_by_bool = self._called_by > 0
        self._called_by_deg = np.maximum(self._called_by.sum(1), 1.0).astype(np.float32)
        self._svc_node_mask = topo.service_node_mask
        self._svc_node_deg = np.maximum(
            topo.service_node_mask.sum(1), 1.0
        ).astype(np.float32)
        self._pod_node_flat = topo.pod_node.reshape(-1)
        self._pod_node_2d = topo.pod_node
        self._inv_pod_capacity = (1.0 / topo.pod_capacity).astype(np.float32)
        # One-hot pod -> node, so rolling pod metrics up to nodes is a matmul
        # instead of a boolean-masked bincount (which allocates twice per call).
        n_pods = topo.n_pods
        self._pod_to_node = np.zeros((n_pods, self.n_nodes), dtype=np.float32)
        self._pod_to_node[np.arange(n_pods), self._pod_node_flat] = 1.0
        self._tier_onehot = np.eye(self.cfg.n_tiers, dtype=np.float32)[topo.service_tier]
        self._rows_all = np.arange(self.n_services)
        self._can_reroute = topo.calls_mask.sum(1) >= 2
        self._inv_calls_deg = (1.0 / self._calls_deg).astype(np.float32)
        self._inv_called_by_deg = (1.0 / self._called_by_deg).astype(np.float32)
        self._inv_node_deg = (1.0 / self._svc_node_deg).astype(np.float32)
        self._inv_n_services = np.float32(1.0 / self.n_services)
        self._demand_scale = float(max(1.0, topo.steady_demand.max() * 2.0))
        self._tier_names = tier_names(self.cfg.n_tiers)
        self._calls_pairs = topo.calls_pairs()

    def _init_runtime_state(self) -> None:
        topo = self.topology
        s_n, r_n, n_n = self.n_services, self.max_replicas, self.n_nodes

        self.pod_alive = topo.initial_alive.copy()
        self.pod_up = topo.initial_alive.copy()
        self.pod_health = topo.initial_alive.astype(np.float32)
        self.pod_cpu = np.zeros((s_n, r_n), dtype=np.float32)
        self.pod_mem = np.zeros((s_n, r_n), dtype=np.float32)
        self.pod_load_cpu = np.zeros((s_n, r_n), dtype=np.float32)
        self.pod_down_timer = np.zeros((s_n, r_n), dtype=np.int32)
        self.pod_pending_timer = np.zeros((s_n, r_n), dtype=np.int32)
        self.pod_restarts = np.zeros((s_n, r_n), dtype=np.int32)
        self.pod_crashed = np.zeros((s_n, r_n), dtype=bool)

        self.node_cpu = np.zeros(n_n, dtype=np.float32)
        self.node_mem = np.zeros(n_n, dtype=np.float32)
        self.node_cpu_pressure = np.zeros(n_n, dtype=np.float32)
        self.node_mem_pressure = np.zeros(n_n, dtype=np.float32)
        self.node_fault_timer = np.zeros(n_n, dtype=np.int32)
        self.node_mem_fault_timer = np.zeros(n_n, dtype=np.int32)

        self.call_weight = topo.base_call_weight.copy()
        self.edge_fault_error = np.zeros((s_n, s_n), dtype=np.float32)
        self.edge_fault_timer = np.zeros((s_n, s_n), dtype=np.int32)
        self.svc_latency_fault = np.zeros(s_n, dtype=np.float32)
        self.svc_latency_fault_timer = np.zeros(s_n, dtype=np.int32)

        self.svc_demand = topo.steady_demand.copy()
        self.svc_capacity = np.zeros(s_n, dtype=np.float32)
        self.svc_latency = topo.base_latency_ms.copy()
        self.svc_error = np.zeros(s_n, dtype=np.float32)
        self.svc_health = np.ones(s_n, dtype=np.float32)
        self.svc_util = np.zeros(s_n, dtype=np.float32)
        self.svc_replicas = self.pod_alive.sum(1).astype(np.int32)
        self.svc_ready = self.pod_up.sum(1).astype(np.int32)
        self.svc_isolate_timer = np.zeros(s_n, dtype=np.int32)
        self._sla_violating = np.zeros(s_n, dtype=bool)
        self.sla_violation_rate = 0.0
        self._lat_norm = np.zeros(s_n, dtype=np.float32)
        self._mean_health = 1.0
        self._min_health = 1.0
        self._frac_sla = 0.0
        self._mean_lat_norm = 0.0
        self._node_pressure_on = False
        self._edge_fault_on = False
        self._svc_lat_fault_on = False
        self._pending_on = False
        self._down_on = False
        self._isolate_on = False

    def _settle(self) -> None:
        """Run the physics fault-free until it reaches its equilibrium."""
        for _ in range(self.cfg.n_tiers + 2):
            self._physics(noise=False)
        self._compute_health_and_sla()

    # ------------------------------------------------------------ step
    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict, dict, dict, dict, dict]:
        arr = self._actions_to_array(actions)
        self._advance(arr)
        return self._package()

    def _actions_to_array(self, actions: dict[str, int]) -> np.ndarray:
        buf = self._action_buf
        buf.fill(ACTION_NOOP)
        idx = self.agent_name_to_index
        for name, a in actions.items():
            i = idx.get(name)
            if i is not None:
                buf[i] = a
        return buf

    def _advance(self, action_arr: np.ndarray) -> None:
        """Full tick: physics + reward + observations, without any dict building.

        ``step`` is exactly this plus :meth:`_package`.  ``benchmark.py`` times
        the two separately to quantify what the PettingZoo dict API costs.
        """
        self.t += 1
        self._tick_timers()
        self._fire_faults()
        self._apply_actions(action_arr)
        self._physics(noise=True)
        self._compute_health_and_sla()
        self._check_termination()
        self._compute_reward()
        self._build_obs()

    def _package(self) -> tuple[dict, dict, dict, dict, dict]:
        names = self.possible_agents
        obs_m, rw = self._obs, self._rewards
        shared = self._reward_shared
        cost = self._reward_components["action_cost"]
        inval = self._reward_components["invalid_action"]
        term, trunc = self._terminated, self._truncated

        observations, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        for i, n in enumerate(names):
            observations[n] = obs_m[i]
            rewards[n] = float(rw[i])
            terminations[n] = term
            truncations[n] = trunc
            infos[n] = {
                "reward_components": {
                    **shared,
                    "action_cost": float(cost[i]),
                    "invalid_action": float(inval[i]),
                },
                "invalid_action": int(self._invalid_counts[i]),
                "invalid_action_now": bool(self._invalid_now[i]),
                "action": int(self._last_action[i]),
            }
        if term or trunc:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    # ------------------------------------------------------------ tick parts
    def _tick_timers(self) -> None:
        """Age every timer by one tick.

        Each block is gated on a cached "is anything running" flag rather than
        an ``.any()`` scan. ``.any()`` costs ~1 us here - more than the work it
        guards - and the fault-free path is the common one, so the flags are set
        when a fault fires or an action starts and cleared when the last timer
        in that category drains.
        """
        if self._pending_on:
            pending = self.pod_pending_timer > 0
            np.subtract(self.pod_pending_timer, pending, out=self.pod_pending_timer)
            landed = pending & (self.pod_pending_timer == 0)
            self.pod_alive |= landed
            self.pod_up |= landed
            self.pod_health[landed] = 1.0
            self._pending_on = bool(self.pod_pending_timer.any())

        # Pods finishing a restart / crash outage come back healthy.
        if self._down_on:
            down = self.pod_down_timer > 0
            np.subtract(self.pod_down_timer, down, out=self.pod_down_timer)
            back = down & (self.pod_down_timer == 0) & self.pod_alive
            self.pod_up |= back
            self.pod_health[back] = 1.0
            self.pod_crashed[back] = False
            self._down_on = bool(self.pod_down_timer.any())

        if self._isolate_on:
            np.maximum(self.svc_isolate_timer - 1, 0, out=self.svc_isolate_timer)
            self._isolate_on = bool(self.svc_isolate_timer.any())

        if self._node_pressure_on:
            np.maximum(self.node_fault_timer - 1, 0, out=self.node_fault_timer)
            self.node_cpu_pressure[self.node_fault_timer == 0] = 0.0
            np.maximum(self.node_mem_fault_timer - 1, 0, out=self.node_mem_fault_timer)
            self.node_mem_pressure[self.node_mem_fault_timer == 0] = 0.0
            self._node_pressure_on = bool(
                self.node_fault_timer.any() or self.node_mem_fault_timer.any()
            )
        if self._edge_fault_on:
            np.maximum(self.edge_fault_timer - 1, 0, out=self.edge_fault_timer)
            self.edge_fault_error[self.edge_fault_timer == 0] = 0.0
            self._edge_fault_on = bool(self.edge_fault_timer.any())
        if self._svc_lat_fault_on:
            np.maximum(
                self.svc_latency_fault_timer - 1, 0, out=self.svc_latency_fault_timer
            )
            self.svc_latency_fault[self.svc_latency_fault_timer == 0] = 0.0
            self._svc_lat_fault_on = bool(self.svc_latency_fault_timer.any())

    def _fire_faults(self) -> None:
        sched = self.fault_schedule
        i = self._next_fault
        n = len(sched)
        while i < n and sched[i].tick <= self.t:
            ev = sched[i]
            apply_fault(self, ev)
            setattr(self, _FAULT_FLAG[ev.fault_type], True)
            self.fired_faults.append(ev)
            self.active_faults.append(ev)
            self._active_expiry.append(self.t + int(ev.duration))
            i += 1
        self._next_fault = i
        if self.active_faults:
            self._prune_active_faults()

    def _prune_active_faults(self) -> None:
        keep_e, keep_x = [], []
        for ev, exp in zip(self.active_faults, self._active_expiry):
            if ev.fault_type is FaultType.POD_CRASH:
                s, r = ev.target
                if not self.pod_crashed[s, r]:
                    continue
            elif exp <= self.t:
                continue
            keep_e.append(ev)
            keep_x.append(exp)
        self.active_faults = keep_e
        self._active_expiry = keep_x

    def _apply_actions(self, a: np.ndarray) -> None:
        """Vectorised over services; each action kind costs nothing if unused.

        ``np.bincount`` over the action array tells us in one call which of the
        six blocks have any work to do. Under a trained policy most steps are
        mostly no-ops, so skipping the unused blocks is the difference between
        six masked passes over the pod grid and one.
        """
        cfg = self.cfg
        rows_all = self._rows_all

        # Viewed as unsigned, a negative action wraps to a huge value, so one
        # comparison catches both out-of-range directions.
        invalid = a.view(np.uint64) >= np.uint64(N_ACTIONS)
        if invalid.any():
            a = np.where(invalid, ACTION_NOOP, a)
        self._last_action[:] = a
        counts = np.bincount(a, minlength=N_ACTIONS)

        alive = self.pod_alive

        # --- restart (1): unhealthiest pod that is not already restarting ----
        if counts[ACTION_RESTART]:
            want = a == ACTION_RESTART
            restarting = (self.pod_down_timer > 0) & ~self.pod_crashed
            elig = alive & ~restarting
            has_target = elig.any(1)
            do = want & has_target
            if do.any():
                score = np.where(elig, self.pod_health, np.inf)
                col = score.argmin(1)
                rows = np.flatnonzero(do)
                cols = col[rows]
                self.pod_up[rows, cols] = False
                self.pod_health[rows, cols] = 0.0
                self.pod_cpu[rows, cols] = 0.0
                self.pod_mem[rows, cols] = 0.0
                self.pod_crashed[rows, cols] = False
                self.pod_down_timer[rows, cols] = cfg.restart_latency
                self.pod_restarts[rows, cols] += 1
                self._down_on = True
            invalid = invalid | (want & ~has_target)

        # --- scale_up (2): free slot, under max_replicas, node has room -------
        if counts[ACTION_SCALE_UP]:
            want = a == ACTION_SCALE_UP
            free = ~alive & (self.pod_pending_timer == 0)
            has_free = free.any(1)
            slot = free.argmax(1)
            pending_count = (self.pod_pending_timer > 0).sum(1)
            under_max = (alive.sum(1) + pending_count) < cfg.max_replicas
            node_counts = np.bincount(
                self._pod_node_flat[alive.reshape(-1)], minlength=self.n_nodes
            )
            node_room = node_counts[self._pod_node_2d[rows_all, slot]] < (
                self.topology.node_pod_capacity
            )
            ok = has_free & under_max & node_room
            do = want & ok
            if do.any():
                rows = np.flatnonzero(do)
                self.pod_pending_timer[rows, slot[rows]] = cfg.scale_latency
                self._pending_on = True
            invalid = invalid | (want & ~ok)

        # --- scale_down (3): drop the unhealthiest alive pod ------------------
        if counts[ACTION_SCALE_DOWN]:
            want = a == ACTION_SCALE_DOWN
            ok = alive.sum(1) > cfg.min_replicas
            do = want & ok
            if do.any():
                score = np.where(alive, self.pod_health, np.inf)
                col = score.argmin(1)
                rows = np.flatnonzero(do)
                cols = col[rows]
                self.pod_alive[rows, cols] = False
                self.pod_up[rows, cols] = False
                self.pod_health[rows, cols] = 0.0
                self.pod_cpu[rows, cols] = 0.0
                self.pod_mem[rows, cols] = 0.0
                self.pod_crashed[rows, cols] = False
                self.pod_down_timer[rows, cols] = 0
            invalid = invalid | (want & ~ok)

        # --- isolate (4): always legal ----------------------------------------
        if counts[ACTION_ISOLATE]:
            self.svc_isolate_timer[a == ACTION_ISOLATE] = cfg.isolate_duration
            self._isolate_on = True

        # --- reroute (5): shift traffic off the worst CALLS edge ---------------
        if counts[ACTION_REROUTE]:
            want = a == ACTION_REROUTE
            multi = self._can_reroute
            do = want & multi
            rows = np.flatnonzero(do)
            if rows.size:
                badness = self._calls_mask[rows] * (
                    self.edge_fault_error[rows]
                    + self.svc_error[None, :]
                    + self.svc_latency[None, :] / cfg.max_latency_ms
                )
                worst = badness.argmax(1)
                k = np.arange(rows.size)
                w = self.call_weight[rows]
                moved = w[k, worst] * cfg.reroute_fraction
                w[k, worst] -= moved
                others = self._calls_mask[rows].copy()
                others[k, worst] = 0.0
                denom = np.maximum(others.sum(1, keepdims=True), 1e-6)
                w += others * (moved[:, None] / denom)
                self.call_weight[rows] = w
            invalid = invalid | (want & ~multi)

        self._invalid_now = invalid
        self._invalid_counts += invalid
        self._action_cost_now = ACTION_COST[a] * ~invalid

    def _physics(self, noise: bool) -> None:
        self._update_resources(noise)
        self._aggregate_services()
        self._propagate()

    def _update_resources(self, noise: bool) -> None:
        """Pod cpu/mem/health from service demand, node pressure and noise.

        Kept in float32 end to end and written in place where possible: at this
        cluster size numpy's per-call dispatch, not the arithmetic, is the cost.
        """
        cfg = self.cfg
        ready = self.pod_alive & self.pod_up
        readyf = ready.astype(np.float32)
        n_ready = np.maximum(ready.sum(1), 1).astype(np.float32)
        util = (self.svc_demand / n_ready)[:, None] * self._inv_pod_capacity

        load_cpu = util * np.float32(cfg.pod_cpu_per_util) + np.float32(cfg.pod_base_cpu)
        load_mem = util * np.float32(cfg.pod_mem_per_util) + np.float32(cfg.pod_base_mem)
        if noise and cfg.noise_std > 0.0:
            shape = load_cpu.shape
            load_cpu += self._rng_dyn.normal(0.0, cfg.noise_std, size=shape).astype(
                np.float32
            )
            load_mem += self._rng_dyn.normal(0.0, cfg.noise_std, size=shape).astype(
                np.float32
            )
        np.clip(load_cpu, 0.0, 1.0, out=load_cpu)
        np.clip(load_mem, 0.0, 1.0, out=load_mem)

        if self._node_pressure_on:
            cpu = load_cpu + self.node_cpu_pressure[self._pod_node_2d]
            mem = load_mem + self.node_mem_pressure[self._pod_node_2d]
            np.clip(cpu, 0.0, 1.0, out=cpu)
            np.clip(mem, 0.0, 1.0, out=mem)
        else:
            cpu, mem = load_cpu, load_mem

        knee = np.float32(cfg.health_knee)
        inv_span = np.float32(1.0 / max(1.0 - cfg.health_knee, 1e-6))
        over = np.maximum(cpu, mem)
        over -= knee
        np.maximum(over, 0.0, out=over)
        over *= inv_span
        health = np.subtract(1.0, over, out=over)
        np.clip(health, 0.0, 1.0, out=health)

        # Dead / not-ready slots report nothing at all.
        self.pod_cpu = cpu * readyf
        self.pod_mem = mem * readyf
        self.pod_health = health * readyf
        self.pod_load_cpu = load_cpu * readyf if cpu is not load_cpu else self.pod_cpu
        self._pod_load_mem = load_mem * readyf if mem is not load_mem else self.pod_mem

        # Node metrics: pod load rolled up (a matmul against the static one-hot
        # pod->node map), plus the node's own fault pressure. Slots that are not
        # ready already carry zero load, so no masking is needed here.
        inv_cap = np.float32(1.0 / float(self.topology.node_pod_capacity))
        node_cpu = (self.pod_load_cpu.reshape(-1) @ self._pod_to_node) * inv_cap
        node_mem = (self._pod_load_mem.reshape(-1) @ self._pod_to_node) * inv_cap
        if self._node_pressure_on:
            node_cpu += self.node_cpu_pressure
            node_mem += self.node_mem_pressure
        np.clip(node_cpu, 0.0, 1.0, out=node_cpu)
        np.clip(node_mem, 0.0, 1.0, out=node_mem)
        self.node_cpu = node_cpu
        self.node_mem = node_mem

    def _aggregate_services(self) -> None:
        ready = self.pod_alive & self.pod_up
        self.svc_replicas = self.pod_alive.sum(1).astype(np.int32)
        self.svc_ready = ready.sum(1).astype(np.int32)
        self.svc_capacity = (
            ready * self.topology.pod_capacity * self.pod_health
        ).sum(1).astype(np.float32)

    def _propagate(self) -> None:
        """One hop of demand / latency / error per tick, so cascades take time."""
        cfg = self.cfg
        any_isolated = bool(self.svc_isolate_timer.any())
        if any_isolated:
            isolated = self.svc_isolate_timer > 0
            live = (~isolated).astype(np.float32)
            # An isolated service neither issues nor accepts calls.
            w = self.call_weight * live[:, None] * live[None, :]
        else:
            isolated = None
            w = self.call_weight

        prev_demand = self.svc_demand
        prev_latency = self.svc_latency
        prev_error = self.svc_error

        demand = self.topology.base_load + w.T @ prev_demand
        np.maximum(demand, 0.0, out=demand)

        cap = np.maximum(self.svc_capacity, 1e-6)
        util = demand / cap
        dead = self.svc_capacity <= 1e-6

        local_lat = self.topology.base_latency_ms * (
            1.0 + cfg.latency_alpha * np.square(np.minimum(util, 4.0))
        )
        if self._svc_lat_fault_on:
            local_lat += self.svc_latency_fault
        local_lat = np.where(dead, cfg.timeout_latency_ms, local_lat)
        latency = local_lat + cfg.cascade_decay * (w @ prev_latency)
        np.minimum(latency, cfg.max_latency_ms, out=latency)

        error = (util - 1.0) * cfg.error_slope
        np.clip(error, 0.0, 1.0, out=error)
        np.copyto(error, 1.0, where=dead)
        if self._edge_fault_on:
            error += (w * self.edge_fault_error).sum(1)
        if any_isolated:
            # Traffic aimed at an isolated dependency degrades gracefully rather
            # than failing outright - this is what makes `isolate` a mitigation.
            lost = (self.call_weight * isolated[None, :]).sum(1)
            error += cfg.isolate_loss_factor * lost
            np.copyto(error, 1.0, where=isolated)
        error += cfg.error_decay * (w @ prev_error)
        np.clip(error, 0.0, 1.0, out=error)

        self.svc_demand = demand.astype(np.float32, copy=False)
        self.svc_util = util.astype(np.float32, copy=False)
        self.svc_latency = latency.astype(np.float32, copy=False)
        self.svc_error = error.astype(np.float32, copy=False)

    def _compute_health_and_sla(self) -> None:
        cfg = self.cfg
        alive_c = np.maximum(self.svc_replicas, 1).astype(np.float32)
        h_pods = np.where(
            self.svc_replicas > 0,
            (self.pod_health * self.pod_alive).sum(1) / alive_c,
            0.0,
        )
        # Health hits its latency floor at 3x the SLA, not at max_latency_ms.
        # Spanning all the way to max_latency_ms made a service 65% over SLA
        # still score ~0.95 health, which is too forgiving to be a useful
        # learning signal for Phase 4.
        lat_span = max(2.0 * cfg.sla_latency_ms, 1e-6)
        h_lat = np.clip(
            1.0 - np.maximum(self.svc_latency - cfg.sla_latency_ms, 0.0) / lat_span,
            0.0,
            1.0,
        )
        h_err = 1.0 - self.svc_error
        self.svc_health = np.clip(
            0.4 * h_pods + 0.3 * h_lat + 0.3 * h_err, 0.0, 1.0
        ).astype(np.float32)

        slow = self.svc_latency > cfg.sla_latency_ms
        self._sla_violating = slow | (self.svc_error > cfg.sla_error_rate)
        bad = self.svc_demand * (
            self.svc_error + slow * (1.0 - self.svc_error)
        )
        total = float(self.svc_demand.sum())
        self.sla_violation_rate = float(bad.sum() / total) if total > 1e-6 else 0.0

        # Aggregates that _check_termination, _compute_reward, _build_obs and
        # state() all want. Computing them once here saves ~6 tiny reductions
        # per step, which at this array size is real time.
        self._lat_norm = np.minimum(
            self.svc_latency * np.float32(1.0 / cfg.max_latency_ms), 1.0
        )
        self._mean_health = float(self.svc_health.mean())
        self._min_health = float(self.svc_health.min())
        self._frac_sla = float(self._sla_violating.mean())
        self._mean_lat_norm = float(self._lat_norm.mean())

    def _faults_active(self) -> bool:
        return bool(
            self.pod_crashed.any()
            or self.node_fault_timer.any()
            or self.node_mem_fault_timer.any()
            or self.edge_fault_timer.any()
            or self.svc_latency_fault_timer.any()
        )

    def _check_termination(self) -> None:
        cfg = self.cfg
        mean_health = self._mean_health
        if mean_health < cfg.collapse_health:
            self._collapse_streak += 1
        else:
            self._collapse_streak = 0

        recovered = (
            # Something must actually have broken - otherwise a fault-free
            # episode would collect the terminal bonus on tick 1.
            len(self.fired_faults) > 0
            and self._next_fault >= len(self.fault_schedule)
            and not self._faults_active()
            and not self.svc_isolate_timer.any()
            and bool((self.svc_health >= cfg.recovery_health).all())
            and self.sla_violation_rate <= 1e-6
        )
        collapsed = self._collapse_streak >= cfg.collapse_patience

        self._terminated = bool(recovered or collapsed)
        self._truncated = bool(not self._terminated and self.t >= self.max_cycles)
        if recovered:
            self.terminal_reason = "recovered"
        elif collapsed:
            self.terminal_reason = "collapsed"
        elif self._truncated:
            self.terminal_reason = "truncated"
        else:
            self.terminal_reason = None

    def _compute_reward(self) -> None:
        """PLACEHOLDER weighting. Phase 4 (marl/reward.py) owns the real shaping.

        Components are exposed raw in ``infos[agent]["reward_components"]`` so a
        reward-hacking policy is visible in the logs rather than hidden inside a
        collapsed scalar (CLAUDE.md non-negotiable).
        """
        cfg = self.cfg
        sla = -cfg.w_sla * self.sla_violation_rate
        lat = -cfg.w_latency * self._mean_lat_norm
        avail = cfg.w_availability * self._mean_health

        term = 0.0
        if self.terminal_reason == "recovered":
            term = cfg.w_terminal_bonus
        elif self.terminal_reason == "collapsed":
            term = -cfg.w_collapse_penalty

        # Cooperative components are identical across agents, so they are held
        # as plain floats and only the two per-agent terms stay as arrays.
        self._reward_shared = {
            "sla_violation": sla,
            "latency": lat,
            "availability": avail,
            "terminal": term,
        }
        c = self._reward_components
        c["sla_violation"].fill(sla)
        c["latency"].fill(lat)
        c["availability"].fill(avail)
        c["terminal"].fill(term)
        cost = np.multiply(
            self._action_cost_now, -cfg.w_action_cost, out=c["action_cost"]
        )
        inval = np.multiply(self._invalid_now, -cfg.w_invalid, out=c["invalid_action"])

        self._rewards = (cost + inval) + np.float32(sla + lat + avail + term)

    # ------------------------------------------------------------ observations
    # OBS LAYOUT (obs_dim = 35 + n_tiers, all values clipped to [0, 1]):
    #   0  service health                 7  ready pods / max_replicas
    #   1  mean pod cpu                   8  isolate timer / isolate_duration
    #   2  mean pod mem                   9  restart count (capped at 10)
    #   3  latency / max_latency_ms      10  demand / demand_scale
    #   4  error rate                    11  SLA violating flag
    #   5  utilisation / 2               12  scale-up in flight flag
    #   6  replicas / max_replicas       13  down pods / max_replicas
    #   14 .. 14+n_tiers-1  tier one-hot
    #   b = 14 + n_tiers
    #   b+0..b+2   downstream (CALLS out) mean: latency, error, health
    #   b+3..b+4   downstream max: latency, error
    #   b+5        downstream min health
    #   b+6..b+11  upstream (CALLS in): same six, same order
    #   b+12..b+13 node context mean: cpu, mem   (over nodes hosting this service)
    #   b+14..b+15 node context max: cpu, mem
    #   b+16..b+20 global: mean health, min health, frac SLA-violating,
    #              mean latency, episode progress
    #
    # The neighbourhood blocks are grouped mean-then-max on purpose: it lets
    # three aggregates be written with one matmul and one slice assignment
    # instead of six of each.
    def _build_obs(self) -> None:
        cfg = self.cfg
        s_n, r_n = self.n_services, self.max_replicas
        nt = cfg.n_tiers
        o = np.empty((s_n, self.obs_dim), dtype=np.float32)

        lat_n = self._lat_norm
        err = self.svc_error
        health = self.svc_health
        alive = self.pod_alive
        inv_alive = 1.0 / np.maximum(self.svc_replicas, 1).astype(np.float32)
        inv_r = np.float32(1.0 / r_n)

        o[:, 0] = health
        o[:, 1] = (self.pod_cpu * alive).sum(1) * inv_alive
        o[:, 2] = (self.pod_mem * alive).sum(1) * inv_alive
        o[:, 3] = lat_n
        o[:, 4] = err
        o[:, 5] = 0.5 * np.minimum(self.svc_util, 2.0)
        o[:, 6] = self.svc_replicas * inv_r
        o[:, 7] = self.svc_ready * inv_r
        o[:, 8] = self.svc_isolate_timer * np.float32(1.0 / max(cfg.isolate_duration, 1))
        o[:, 9] = np.minimum(self.pod_restarts.sum(1) * 0.1, 1.0)
        o[:, 10] = np.minimum(self.svc_demand * np.float32(1.0 / self._demand_scale), 1.0)
        o[:, 11] = self._sla_violating
        o[:, 12] = (self.pod_pending_timer > 0).any(1)
        o[:, 13] = ((self.pod_down_timer > 0) & alive).sum(1) * inv_r
        o[:, 14 : 14 + nt] = self._tier_onehot

        # (S, 3) of [latency, error, health] - one matmul covers all three means.
        feats = np.stack((lat_n, err, health), axis=1)
        b = 14 + nt
        m, mb, mdeg = self._calls_mask, self._calls_bool, self._calls_deg
        o[:, b + 0 : b + 3] = (m @ feats) / mdeg[:, None]
        o[:, b + 3 : b + 5] = (m[:, :, None] * feats[None, :, :2]).max(1)
        o[:, b + 5] = np.where(mb, health[None, :], 1.0).min(1)

        u, ub, udeg = self._called_by, self._called_by_bool, self._called_by_deg
        o[:, b + 6 : b + 9] = (u @ feats) / udeg[:, None]
        o[:, b + 9 : b + 11] = (u[:, :, None] * feats[None, :, :2]).max(1)
        o[:, b + 11] = np.where(ub, health[None, :], 1.0).min(1)

        nfeats = np.stack((self.node_cpu, self.node_mem), axis=1)
        nm, ndeg = self._svc_node_mask, self._svc_node_deg
        o[:, b + 12 : b + 14] = (nm @ nfeats) / ndeg[:, None]
        o[:, b + 14 : b + 16] = (nm[:, :, None] * nfeats[None, :, :]).max(1)

        o[:, b + 16 : b + 21] = (
            self._mean_health,
            self._min_health,
            self._frac_sla,
            self._mean_lat_norm,
            min(self.t / max(self.max_cycles, 1), 1.0),
        )

        np.clip(o, 0.0, 1.0, out=o)
        self._obs = o

    def state(self) -> np.ndarray:
        """Global state for the centralized critic (CTDE). Shape ``(state_dim,)``."""
        cfg = self.cfg
        s_n, r_n = self.n_services, self.max_replicas
        alive = self.pod_alive
        alive_c = np.maximum(self.svc_replicas, 1).astype(np.float32)
        per_svc = np.empty((s_n, 10), dtype=np.float32)
        per_svc[:, 0] = self.svc_health
        per_svc[:, 1] = (self.pod_cpu * alive).sum(1) / alive_c
        per_svc[:, 2] = (self.pod_mem * alive).sum(1) / alive_c
        per_svc[:, 3] = self.svc_latency / cfg.max_latency_ms
        per_svc[:, 4] = self.svc_error
        per_svc[:, 5] = 0.5 * np.minimum(self.svc_util, 2.0)
        per_svc[:, 6] = self.svc_replicas / r_n
        per_svc[:, 7] = self.svc_ready / r_n
        per_svc[:, 8] = self.svc_isolate_timer / max(cfg.isolate_duration, 1)
        per_svc[:, 9] = self._sla_violating

        alive_flat = alive.reshape(-1)
        counts = np.bincount(
            self._pod_node_flat[alive_flat], minlength=self.n_nodes
        ).astype(np.float32)
        per_node = np.empty((self.n_nodes, 3), dtype=np.float32)
        per_node[:, 0] = self.node_cpu
        per_node[:, 1] = self.node_mem
        per_node[:, 2] = counts / float(self.topology.node_pod_capacity)

        glob = np.array(
            [
                self._mean_health,
                self._frac_sla,
                self._mean_lat_norm,
                min(self.t / max(self.max_cycles, 1), 1.0),
                len(self.active_faults) / max(len(self.fault_schedule), 1),
            ],
            dtype=np.float32,
        )
        out = np.concatenate([per_svc.reshape(-1), per_node.reshape(-1), glob])
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------ graph view
    def graph_snapshot(self) -> dict[str, Any]:
        """Nodes and relationships in the PLAN.md section 3 Phase 2 schema shape.

        Not on the hot path - Phase 2 ingestion calls this, training does not.

        Emits labels ``Service`` / ``Pod`` / ``Node`` and relationship types
        ``DEPENDS_ON`` / ``INSTANCE_OF`` / ``RUNS_ON`` / ``CALLS``.  Every node
        carries ``health``, ``cpu_pct``, ``mem_pct`` and ``restart_count``;
        ``CALLS`` carries ``p99_latency_ms`` and ``error_rate``.
        """
        topo = self.topology
        r_n = self.max_replicas
        alive = self.pod_alive
        alive_c = np.maximum(self.svc_replicas, 1).astype(np.float32)
        svc_cpu = (self.pod_cpu * alive).sum(1) / alive_c
        svc_mem = (self.pod_mem * alive).sum(1) / alive_c
        svc_restarts = self.pod_restarts.sum(1)

        services = [
            {
                "id": topo.service_names[i],
                "name": topo.service_names[i],
                "tier": self._tier_names[int(topo.service_tier[i])],
                "health": float(self.svc_health[i]),
                "cpu_pct": float(svc_cpu[i] * 100.0),
                "mem_pct": float(svc_mem[i] * 100.0),
                "restart_count": int(svc_restarts[i]),
                "replicas": int(self.svc_replicas[i]),
                "ready_replicas": int(self.svc_ready[i]),
                "p99_latency_ms": float(self.svc_latency[i]),
                "error_rate": float(self.svc_error[i]),
                "isolated": bool(self.svc_isolate_timer[i] > 0),
                "sla_violating": bool(self._sla_violating[i]),
            }
            for i in range(self.n_services)
        ]

        pods, instance_of, runs_on = [], [], []
        for s in range(self.n_services):
            for r in range(r_n):
                if not alive[s, r]:
                    continue
                pid = topo.pod_names[s * r_n + r]
                if self.pod_crashed[s, r]:
                    status = "CrashLoopBackOff"
                elif self.pod_down_timer[s, r] > 0:
                    status = "Restarting"
                elif self.pod_up[s, r]:
                    status = "Running"
                else:
                    status = "NotReady"
                pods.append(
                    {
                        "id": pid,
                        "name": pid,
                        "health": float(self.pod_health[s, r]),
                        "cpu_pct": float(self.pod_cpu[s, r] * 100.0),
                        "mem_pct": float(self.pod_mem[s, r] * 100.0),
                        "restart_count": int(self.pod_restarts[s, r]),
                        "status": status,
                    }
                )
                instance_of.append({"source": pid, "target": topo.service_names[s]})
                runs_on.append(
                    {"source": pid, "target": topo.node_names[int(topo.pod_node[s, r])]}
                )

        alive_flat = alive.reshape(-1)
        counts = np.bincount(
            self._pod_node_flat[alive_flat], minlength=self.n_nodes
        )
        nodes = [
            {
                "id": topo.node_names[n],
                "name": topo.node_names[n],
                "health": float(max(0.0, 1.0 - max(self.node_cpu[n], self.node_mem[n]))),
                "cpu_pct": float(self.node_cpu[n] * 100.0),
                "mem_pct": float(self.node_mem[n] * 100.0),
                "restart_count": 0,
                "pod_count": int(counts[n]),
                "pod_capacity": int(topo.node_pod_capacity),
            }
            for n in range(self.n_nodes)
        ]

        depends_on = [
            {"source": topo.service_names[i], "target": topo.service_names[j]}
            for i, j in topo.depends_on_pairs()
        ]
        calls = [
            {
                "source": topo.service_names[i],
                "target": topo.service_names[j],
                "p99_latency_ms": float(self.svc_latency[j]),
                "error_rate": float(
                    min(1.0, self.svc_error[j] + self.edge_fault_error[i, j])
                ),
                "traffic_share": float(self.call_weight[i, j]),
            }
            for i, j in self._calls_pairs
        ]

        return {
            "tick": int(self.t),
            "nodes": {"Service": services, "Pod": pods, "Node": nodes},
            "relationships": {
                "DEPENDS_ON": depends_on,
                "INSTANCE_OF": instance_of,
                "RUNS_ON": runs_on,
                "CALLS": calls,
            },
            "active_faults": [
                {
                    "type": ev.fault_type.name,
                    "target": list(ev.target),
                    "tick": int(ev.tick),
                    "duration": int(ev.duration),
                    "magnitude": float(ev.magnitude),
                }
                for ev in self.active_faults
            ],
        }

    # ------------------------------------------------------------ misc
    def render(self) -> str | None:
        lines = [
            f"tick={self.t}  mean_health={self.svc_health.mean():.3f}  "
            f"sla_violation_rate={self.sla_violation_rate:.3f}  "
            f"active_faults={len(self.active_faults)}"
        ]
        for i in range(self.n_services):
            lines.append(
                f"  {self.topology.service_names[i]:>18}  "
                f"h={self.svc_health[i]:.2f} lat={self.svc_latency[i]:7.1f}ms "
                f"err={self.svc_error[i]:.2f} "
                f"pods={self.svc_ready[i]}/{self.svc_replicas[i]}"
            )
        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
            return None
        return text

    def close(self) -> None:
        return None


def make_env(**kwargs: Any) -> ClusterEnv:
    """Factory used by tests, benchmark.py and (later) marl/train.py."""
    return ClusterEnv(**kwargs)
