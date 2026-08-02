"""Seeded, reproducible fault injection: pod crash, CPU/memory spike, network partition, cascading latency.

Phase 1 - owned by the sim-engineer subagent. See PLAN.md section 3.

Reproducibility model
---------------------
The *entire* episode's fault schedule is drawn from the seeded RNG up front,
at :meth:`ClusterEnv.reset`, into a tick-sorted list of :class:`FaultEvent`.
The step loop only fires events whose tick has arrived.  Consequences:

* "same seed -> identical trajectory" is exact, not statistical.
* Tests can assert directly on ``env.fault_schedule``.
* Phase 5 gets a ground-truth causal log to check narrations against.

Adding a fault type means adding an entry to :class:`FaultType`, a builder in
``_BUILDERS`` (target selection + duration/magnitude draw) and an applier in
``_APPLIERS`` (the state mutation).  Nothing else in the env changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from simulator.cluster_env import ClusterConfig, ClusterEnv
    from simulator.topology_generator import Topology


class FaultType(IntEnum):
    """Fault kinds. Values are stable - they are used as sort keys."""

    POD_CRASH = 0
    NODE_CPU_SPIKE = 1
    NODE_MEM_SPIKE = 2
    NETWORK_PARTITION = 3
    CASCADING_LATENCY = 4


@dataclass(frozen=True)
class FaultEvent:
    """One scheduled fault.

    ``target`` is a tuple of integer ids whose meaning depends on
    ``fault_type``:

    ===================  ==========================================  ==================
    fault_type           target                                      magnitude
    ===================  ==========================================  ==================
    POD_CRASH            ``(service_idx, replica_slot)``             unused (1.0)
    NODE_CPU_SPIKE       ``(node_idx,)``                             extra cpu fraction
    NODE_MEM_SPIKE       ``(node_idx,)``                             extra mem fraction
    NETWORK_PARTITION    ``(caller_idx, callee_idx)`` on a CALLS edge added error rate
    CASCADING_LATENCY    ``(service_idx,)`` in the data tier         injected ms
    ===================  ==========================================  ==================
    """

    tick: int
    fault_type: FaultType
    target: tuple[int, ...]
    duration: int
    magnitude: float

    def sort_key(self) -> tuple:
        return (self.tick, int(self.fault_type), self.target, self.duration)

    def describe(self, topo: "Topology") -> str:
        name = self.fault_type.name
        if self.fault_type is FaultType.POD_CRASH:
            s, r = self.target
            what = topo.pod_names[s * topo.max_replicas + r]
        elif self.fault_type in (FaultType.NODE_CPU_SPIKE, FaultType.NODE_MEM_SPIKE):
            what = topo.node_names[self.target[0]]
        elif self.fault_type is FaultType.NETWORK_PARTITION:
            i, j = self.target
            what = f"{topo.service_names[i]}->{topo.service_names[j]}"
        else:
            what = topo.service_names[self.target[0]]
        return f"t={self.tick} {name} {what} (dur={self.duration}, mag={self.magnitude:.3g})"


# --------------------------------------------------------------------------
# Schedule construction
# --------------------------------------------------------------------------

def _draw_int_range(rng: np.random.Generator, span: Sequence[int]) -> int:
    lo, hi = int(span[0]), int(span[1])
    return int(rng.integers(lo, hi + 1))


def _draw_float_range(rng: np.random.Generator, span: Sequence[float]) -> float:
    lo, hi = float(span[0]), float(span[1])
    return float(rng.uniform(lo, hi))


def _pod_crash_event(
    tick: int, topo: "Topology", cfg: "ClusterConfig", rng: np.random.Generator
) -> FaultEvent | None:
    alive = np.argwhere(topo.initial_alive)
    if alive.shape[0] == 0:
        return None
    idx = int(rng.integers(0, alive.shape[0]))
    s, r = int(alive[idx, 0]), int(alive[idx, 1])
    return FaultEvent(
        tick=tick,
        fault_type=FaultType.POD_CRASH,
        target=(s, r),
        duration=_draw_int_range(rng, cfg.pod_crash_duration_range),
        magnitude=1.0,
    )


def _node_cpu_spike_event(
    tick: int, topo: "Topology", cfg: "ClusterConfig", rng: np.random.Generator
) -> FaultEvent | None:
    return FaultEvent(
        tick=tick,
        fault_type=FaultType.NODE_CPU_SPIKE,
        target=(int(rng.integers(0, topo.n_nodes)),),
        duration=_draw_int_range(rng, cfg.node_spike_duration_range),
        magnitude=_draw_float_range(rng, cfg.node_spike_magnitude_range),
    )


def _node_mem_spike_event(
    tick: int, topo: "Topology", cfg: "ClusterConfig", rng: np.random.Generator
) -> FaultEvent | None:
    return FaultEvent(
        tick=tick,
        fault_type=FaultType.NODE_MEM_SPIKE,
        target=(int(rng.integers(0, topo.n_nodes)),),
        duration=_draw_int_range(rng, cfg.node_spike_duration_range),
        magnitude=_draw_float_range(rng, cfg.node_spike_magnitude_range),
    )


def _network_partition_event(
    tick: int, topo: "Topology", cfg: "ClusterConfig", rng: np.random.Generator
) -> FaultEvent | None:
    edges = np.argwhere(topo.calls_mask > 0)
    if edges.shape[0] == 0:
        return None
    idx = int(rng.integers(0, edges.shape[0]))
    i, j = int(edges[idx, 0]), int(edges[idx, 1])
    return FaultEvent(
        tick=tick,
        fault_type=FaultType.NETWORK_PARTITION,
        target=(i, j),
        duration=_draw_int_range(rng, cfg.partition_duration_range),
        magnitude=_draw_float_range(rng, cfg.partition_magnitude_range),
    )


def _cascading_latency_event(
    tick: int, topo: "Topology", cfg: "ClusterConfig", rng: np.random.Generator
) -> FaultEvent | None:
    """Injected at the deepest tier, so it has somewhere to cascade *to*."""
    deepest = np.flatnonzero(topo.service_tier == topo.n_tiers - 1)
    if deepest.size == 0:
        deepest = np.arange(topo.n_services)
    s = int(deepest[int(rng.integers(0, deepest.size))])
    return FaultEvent(
        tick=tick,
        fault_type=FaultType.CASCADING_LATENCY,
        target=(s,),
        duration=_draw_int_range(rng, cfg.cascade_duration_range),
        magnitude=_draw_float_range(rng, cfg.cascade_magnitude_ms_range),
    )


_BUILDERS = {
    FaultType.POD_CRASH: _pod_crash_event,
    FaultType.NODE_CPU_SPIKE: _node_cpu_spike_event,
    FaultType.NODE_MEM_SPIKE: _node_mem_spike_event,
    FaultType.NETWORK_PARTITION: _network_partition_event,
    FaultType.CASCADING_LATENCY: _cascading_latency_event,
}


def build_fault_schedule(
    topo: "Topology",
    cfg: "ClusterConfig",
    max_cycles: int,
    rng: np.random.Generator,
) -> list[FaultEvent]:
    """Draw the whole episode's faults up front, sorted by tick.

    Faults are confined to the first ``cfg.fault_window_frac`` of the episode so
    that "cluster recovered" termination stays reachable.
    """
    enabled = [f for f in cfg.enabled_faults if f in _BUILDERS]
    if not enabled:
        return []

    n_faults = _draw_int_range(rng, cfg.n_faults_range)
    if n_faults <= 0:
        return []

    first = max(1, int(cfg.fault_start_tick))
    last = max(first, int(max_cycles * float(cfg.fault_window_frac)))

    events: list[FaultEvent] = []
    for _ in range(n_faults):
        tick = int(rng.integers(first, last + 1))
        ftype = enabled[int(rng.integers(0, len(enabled)))]
        ev = _BUILDERS[ftype](tick, topo, cfg, rng)
        if ev is not None:
            events.append(ev)

    events.sort(key=FaultEvent.sort_key)
    return events


# --------------------------------------------------------------------------
# Appliers
# --------------------------------------------------------------------------
# These mutate ClusterEnv's runtime arrays in place. The env is passed
# duck-typed so this module never imports cluster_env at runtime.
# Overlapping faults compose by taking the max of magnitude and remaining
# duration, so two spikes on one node never cancel each other out.


def _apply_pod_crash(env: "ClusterEnv", ev: FaultEvent) -> None:
    s, r = ev.target
    if not env.pod_alive[s, r]:
        # Slot was scaled down before the crash landed - nothing to crash.
        return
    env.pod_up[s, r] = False
    env.pod_health[s, r] = 0.0
    env.pod_cpu[s, r] = 0.0
    env.pod_mem[s, r] = 0.0
    env.pod_crashed[s, r] = True
    env.pod_down_timer[s, r] = int(ev.duration)


def _apply_node_cpu_spike(env: "ClusterEnv", ev: FaultEvent) -> None:
    n = ev.target[0]
    env.node_cpu_pressure[n] = max(float(env.node_cpu_pressure[n]), float(ev.magnitude))
    env.node_fault_timer[n] = max(int(env.node_fault_timer[n]), int(ev.duration))


def _apply_node_mem_spike(env: "ClusterEnv", ev: FaultEvent) -> None:
    n = ev.target[0]
    env.node_mem_pressure[n] = max(float(env.node_mem_pressure[n]), float(ev.magnitude))
    env.node_mem_fault_timer[n] = max(int(env.node_mem_fault_timer[n]), int(ev.duration))


def _apply_network_partition(env: "ClusterEnv", ev: FaultEvent) -> None:
    i, j = ev.target
    env.edge_fault_error[i, j] = max(
        float(env.edge_fault_error[i, j]), float(ev.magnitude)
    )
    env.edge_fault_timer[i, j] = max(int(env.edge_fault_timer[i, j]), int(ev.duration))


def _apply_cascading_latency(env: "ClusterEnv", ev: FaultEvent) -> None:
    s = ev.target[0]
    env.svc_latency_fault[s] = max(
        float(env.svc_latency_fault[s]), float(ev.magnitude)
    )
    env.svc_latency_fault_timer[s] = max(
        int(env.svc_latency_fault_timer[s]), int(ev.duration)
    )


_APPLIERS = {
    FaultType.POD_CRASH: _apply_pod_crash,
    FaultType.NODE_CPU_SPIKE: _apply_node_cpu_spike,
    FaultType.NODE_MEM_SPIKE: _apply_node_mem_spike,
    FaultType.NETWORK_PARTITION: _apply_network_partition,
    FaultType.CASCADING_LATENCY: _apply_cascading_latency,
}


def apply_fault(env: "ClusterEnv", ev: FaultEvent) -> None:
    """Dispatch one fault event onto the environment's runtime arrays."""
    applier = _APPLIERS.get(ev.fault_type)
    if applier is None:  # pragma: no cover - unreachable while enum is complete
        raise ValueError(f"no applier registered for {ev.fault_type!r}")
    applier(env, ev)
