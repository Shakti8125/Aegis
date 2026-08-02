"""Generates service/pod/node topologies for the simulated cluster.

Phase 1 - owned by the sim-engineer subagent. See PLAN.md section 3.

Design notes
------------
Everything is *struct-of-arrays*: there are no ``Pod`` or ``Service`` objects,
only parallel numpy arrays.  This keeps the hot step loop free of Python-level
iteration.

Services are assigned **tiers** (``edge -> mid -> data``).  ``DEPENDS_ON``
edges point only from a lower tier to the next higher tier, which makes the
dependency graph a DAG *by construction* - there is no cycle-rejection loop -
and gives cascade propagation a well-defined direction (data tier upward to
the edge tier, one hop per tick).

``CALLS`` is a subset of ``DEPENDS_ON``: a declared dependency that currently
carries no live traffic has no ``CALLS`` edge.  This mirrors PLAN.md section 3
Phase 2 exactly.

Pods live in a **fixed-capacity** ``(n_services, max_replicas)`` grid.  Pod
``(s, r)`` always belongs to service ``s`` and always lands on the same node,
so ``scale_up`` / ``scale_down`` only flip bits in an ``alive`` mask and never
reallocate.  Replica slots of one service are spread over distinct nodes
(whenever ``max_replicas <= n_nodes``) so that a node fault stays
distinguishable from a total service outage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from simulator.cluster_env import ClusterConfig


def tier_names(n_tiers: int) -> tuple[str, ...]:
    """Human-readable tier labels; the canonical 3-tier case is edge/mid/data."""
    if n_tiers == 3:
        return ("edge", "mid", "data")
    if n_tiers == 1:
        return ("edge",)
    if n_tiers == 2:
        return ("edge", "data")
    middles = tuple(f"mid{i}" for i in range(1, n_tiers - 1))
    return ("edge",) + middles + ("data",)


@dataclass(frozen=True)
class Topology:
    """Static cluster structure. Nothing here changes during an episode.

    Shapes: ``S = n_services``, ``N = n_nodes``, ``R = max_replicas``,
    ``P = S * R`` (the fixed pod-slot pool).
    """

    n_services: int
    n_nodes: int
    max_replicas: int
    n_tiers: int

    service_names: tuple[str, ...]
    node_names: tuple[str, ...]
    pod_names: tuple[str, ...]  # flat, length P, index = s * R + r

    service_tier: np.ndarray  # (S,)   int32, 0 = edge ... n_tiers-1 = data
    depends_on: np.ndarray  # (S,S) float32 0/1, row = dependent, col = dependency
    calls_mask: np.ndarray  # (S,S) float32 0/1, subset of depends_on
    base_call_weight: np.ndarray  # (S,S) float32, rows sum to 1 over CALLS edges
    base_load: np.ndarray  # (S,)   float32 external requests/tick
    base_latency_ms: np.ndarray  # (S,)   float32 unloaded service latency

    pod_node: np.ndarray  # (S,R) int32   node hosting slot (s, r)
    pod_capacity: np.ndarray  # (S,R) float32 requests/tick one pod can serve
    initial_alive: np.ndarray  # (S,R) bool    slots alive at reset
    service_node_mask: np.ndarray  # (S,N) float32 0/1, service has a slot on node
    node_pod_capacity: int
    steady_demand: np.ndarray  # (S,)   float32 fault-free equilibrium demand

    # ---------------------------------------------------------------- helpers
    @property
    def n_pods(self) -> int:
        return self.n_services * self.max_replicas

    @property
    def pod_service_flat(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_services, dtype=np.int32), self.max_replicas)

    @property
    def pod_node_flat(self) -> np.ndarray:
        return self.pod_node.reshape(-1)

    def depends_on_pairs(self) -> list[tuple[int, int]]:
        src, dst = np.nonzero(self.depends_on)
        return list(zip(src.tolist(), dst.tolist()))

    def calls_pairs(self) -> list[tuple[int, int]]:
        src, dst = np.nonzero(self.calls_mask)
        return list(zip(src.tolist(), dst.tolist()))


def _assign_tiers(n_services: int, n_tiers: int) -> np.ndarray:
    """Contiguous, as-equal-as-possible tier blocks; service index order = tier order."""
    if n_services < n_tiers:
        raise ValueError(f"need at least n_tiers={n_tiers} services, got {n_services}")
    sizes = [n_services // n_tiers] * n_tiers
    for i in range(n_services % n_tiers):
        sizes[i] += 1
    tiers = np.empty(n_services, dtype=np.int32)
    start = 0
    for t, size in enumerate(sizes):
        tiers[start : start + size] = t
        start += size
    return tiers


def _build_dependencies(
    tiers: np.ndarray,
    n_tiers: int,
    max_deps: int,
    allow_skip_tier: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    """DAG by construction: edges only go from tier t to tier t+1 (or t+2 if allowed)."""
    n = tiers.shape[0]
    dep = np.zeros((n, n), dtype=np.float32)
    by_tier = [np.flatnonzero(tiers == t) for t in range(n_tiers)]

    for t in range(n_tiers - 1):
        pool = by_tier[t + 1]
        if allow_skip_tier and t + 2 < n_tiers:
            pool = np.concatenate([pool, by_tier[t + 2]])
        if pool.size == 0:
            continue
        for i in by_tier[t]:
            k = int(rng.integers(1, min(max_deps, pool.size) + 1))
            targets = rng.choice(pool, size=k, replace=False)
            dep[i, targets] = 1.0

    # Guarantee reachability: every non-edge service is depended on by someone.
    for t in range(1, n_tiers):
        for j in by_tier[t]:
            if dep[:, j].sum() == 0.0:
                parents = by_tier[t - 1]
                if parents.size:
                    dep[rng.choice(parents), j] = 1.0
    return dep


def _build_calls(
    dep: np.ndarray, call_prob: float, rng: np.random.Generator
) -> np.ndarray:
    """CALLS is the live-traffic subset of DEPENDS_ON; keep >= 1 per service with deps."""
    calls = (dep > 0) & (rng.random(dep.shape) < call_prob)
    calls = calls.astype(np.float32)
    for i in range(dep.shape[0]):
        deps_i = np.flatnonzero(dep[i] > 0)
        if deps_i.size and calls[i].sum() == 0.0:
            calls[i, rng.choice(deps_i)] = 1.0
    return calls


def _steady_state_demand(
    base_load: np.ndarray, weight: np.ndarray, n_tiers: int
) -> np.ndarray:
    """Fixed point of ``d = base + Wt @ d``; exact after n_tiers sweeps on a DAG."""
    d = base_load.astype(np.float64).copy()
    wt = weight.astype(np.float64).T
    for _ in range(n_tiers + 2):
        d = base_load + wt @ d
    return d.astype(np.float32)


def generate_topology(cfg: "ClusterConfig", rng: np.random.Generator) -> Topology:
    """Build a :class:`Topology` deterministically from ``rng``.

    ``rng`` is consumed in a fixed order, so the same seeded generator always
    yields byte-identical arrays.
    """
    s_n = int(cfg.n_services)
    n_n = int(cfg.n_nodes)
    r_n = int(cfg.max_replicas)
    n_tiers = int(cfg.n_tiers)

    tiers = _assign_tiers(s_n, n_tiers)
    dep = _build_dependencies(
        tiers, n_tiers, int(cfg.max_deps_per_service), bool(cfg.allow_skip_tier_deps), rng
    )
    calls = _build_calls(dep, float(cfg.call_edge_prob), rng)

    # Traffic split: uniform over a service's CALLS edges.
    deg = calls.sum(axis=1, keepdims=True)
    weight = np.divide(calls, deg, out=np.zeros_like(calls), where=deg > 0)

    # External load only enters at the edge tier; deeper tiers get a trickle.
    lo_e, hi_e = cfg.external_load_range
    lo_i, hi_i = cfg.internal_load_range
    base_load = np.where(
        tiers == 0,
        rng.uniform(lo_e, hi_e, size=s_n),
        rng.uniform(lo_i, hi_i, size=s_n),
    ).astype(np.float32)

    lo_l, hi_l = cfg.base_latency_ms_range
    base_latency = rng.uniform(lo_l, hi_l, size=s_n).astype(np.float32)
    # Data-tier services are intrinsically slower (disk/IO bound).
    base_latency = base_latency * (1.0 + 0.25 * tiers).astype(np.float32)

    # Replica slot -> node. Slots of one service walk consecutive nodes, so
    # replicas spread across distinct nodes whenever max_replicas <= n_nodes.
    node_offset = rng.integers(0, n_n, size=s_n)
    pod_node = (node_offset[:, None] + np.arange(r_n)[None, :]) % n_n
    pod_node = pod_node.astype(np.int32)

    steady = _steady_state_demand(base_load, weight, n_tiers)
    init_r = int(cfg.init_replicas)
    per_pod = steady / max(init_r * float(cfg.target_utilization), 1e-6)
    per_pod = np.maximum(per_pod, 1.0).astype(np.float32)
    pod_capacity = np.repeat(per_pod[:, None], r_n, axis=1).astype(np.float32)

    initial_alive = np.zeros((s_n, r_n), dtype=bool)
    initial_alive[:, :init_r] = True

    service_node_mask = np.zeros((s_n, n_n), dtype=np.float32)
    service_node_mask[np.arange(s_n)[:, None], pod_node] = 1.0

    node_cap = cfg.node_pod_capacity
    if node_cap is None:
        node_cap = max(int(cfg.init_replicas) + 1, int(0.75 * s_n * r_n / n_n))
    node_cap = int(node_cap)

    names = tier_names(n_tiers)
    service_names = tuple(f"svc-{i:02d}-{names[tiers[i]]}" for i in range(s_n))
    node_names = tuple(f"node-{i}" for i in range(n_n))
    pod_names = tuple(f"pod-{s:02d}-{r}" for s in range(s_n) for r in range(r_n))

    return Topology(
        n_services=s_n,
        n_nodes=n_n,
        max_replicas=r_n,
        n_tiers=n_tiers,
        service_names=service_names,
        node_names=node_names,
        pod_names=pod_names,
        service_tier=tiers,
        depends_on=dep,
        calls_mask=calls,
        base_call_weight=weight.astype(np.float32),
        base_load=base_load,
        base_latency_ms=base_latency.astype(np.float32),
        pod_node=pod_node,
        pod_capacity=pod_capacity,
        initial_alive=initial_alive,
        service_node_mask=service_node_mask,
        node_pod_capacity=node_cap,
        steady_demand=steady,
    )
