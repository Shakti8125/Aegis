"""Topology generation: determinism, DAG-ness, PLAN.md Phase 2 schema conformance."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.cluster_env import ClusterConfig, ClusterEnv
from simulator.topology_generator import generate_topology


def _topo(seed: int, **overrides):
    cfg = ClusterConfig(**overrides)
    rng = np.random.default_rng(seed)
    return generate_topology(cfg, rng)


# --------------------------------------------------------------- determinism
def test_same_seed_gives_identical_topology():
    a = _topo(1234)
    b = _topo(1234)
    for field in (
        "service_tier",
        "depends_on",
        "calls_mask",
        "base_call_weight",
        "base_load",
        "base_latency_ms",
        "pod_node",
        "pod_capacity",
        "initial_alive",
        "steady_demand",
    ):
        np.testing.assert_array_equal(
            getattr(a, field), getattr(b, field), err_msg=f"{field} diverged"
        )
    assert a.service_names == b.service_names
    assert a.node_pod_capacity == b.node_pod_capacity


def test_different_seeds_give_different_topologies():
    a = _topo(1)
    b = _topo(2)
    differs = (
        not np.array_equal(a.depends_on, b.depends_on)
        or not np.array_equal(a.pod_node, b.pod_node)
        or not np.array_equal(a.base_load, b.base_load)
    )
    assert differs, "two different seeds produced an identical topology"


def test_env_fixed_topology_survives_resets():
    env = ClusterEnv(fixed_topology=True)
    env.reset(seed=5)
    first = env.topology.depends_on.copy()
    for _ in range(3):
        env.reset()
        np.testing.assert_array_equal(env.topology.depends_on, first)


def test_env_unfixed_topology_changes_across_resets():
    env = ClusterEnv(fixed_topology=False)
    env.reset(seed=5)
    seen = [env.topology.depends_on.copy()]
    for _ in range(6):
        env.reset()
        seen.append(env.topology.depends_on.copy())
    assert any(not np.array_equal(seen[0], s) for s in seen[1:])


# --------------------------------------------------------------------- DAG
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11, 42])
def test_depends_on_is_acyclic(seed):
    topo = _topo(seed)
    adj = (topo.depends_on > 0).astype(np.int64)
    # Reachability closure: a cycle would put a nonzero on the diagonal of some
    # power of the adjacency matrix.
    reach = adj.copy()
    power = adj.copy()
    for _ in range(topo.n_services):
        power = (power @ adj > 0).astype(np.int64)
        reach |= power
    assert np.trace(reach) == 0, "DEPENDS_ON contains a cycle"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 7, 11, 42])
def test_dependencies_point_strictly_downward_in_tier(seed):
    topo = _topo(seed)
    src, dst = np.nonzero(topo.depends_on)
    assert src.size > 0
    assert np.all(topo.service_tier[dst] > topo.service_tier[src])


def test_every_non_edge_service_has_an_upstream_dependent():
    topo = _topo(3)
    non_edge = np.flatnonzero(topo.service_tier > 0)
    incoming = topo.depends_on.sum(axis=0)
    assert np.all(incoming[non_edge] > 0)


# ------------------------------------------------------------ schema shape
@pytest.mark.parametrize("seed", [0, 5, 9])
def test_calls_is_a_subset_of_depends_on(seed):
    topo = _topo(seed)
    assert np.all(topo.calls_mask <= topo.depends_on)
    assert topo.calls_mask.sum() > 0
    # Every service that has dependencies keeps at least one live CALLS edge.
    has_dep = topo.depends_on.sum(1) > 0
    assert np.all(topo.calls_mask.sum(1)[has_dep] > 0)


def test_call_weights_sum_to_one_per_calling_service():
    topo = _topo(4)
    rows = topo.calls_mask.sum(1) > 0
    np.testing.assert_allclose(topo.base_call_weight.sum(1)[rows], 1.0, atol=1e-6)
    assert np.all(topo.base_call_weight.sum(1)[~rows] == 0.0)


def test_replicas_spread_across_distinct_nodes():
    cfg = ClusterConfig()
    topo = _topo(8)
    assert cfg.max_replicas <= cfg.n_nodes, "fixture assumes slots can be spread"
    for s in range(topo.n_services):
        nodes = topo.pod_node[s]
        assert len(set(nodes.tolist())) == len(nodes), (
            f"service {s} has two replica slots on the same node: {nodes}"
        )


def test_graph_snapshot_matches_phase2_schema():
    env = ClusterEnv()
    env.reset(seed=17)
    snap = env.graph_snapshot()

    assert set(snap["nodes"]) == {"Service", "Pod", "Node"}
    assert set(snap["relationships"]) == {
        "DEPENDS_ON",
        "INSTANCE_OF",
        "RUNS_ON",
        "CALLS",
    }

    # PLAN.md: node properties health / cpu_pct / mem_pct / restart_count.
    for label, rows in snap["nodes"].items():
        assert rows, f"{label} had no instances"
        for row in rows:
            for prop in ("id", "health", "cpu_pct", "mem_pct", "restart_count"):
                assert prop in row, f"{label} missing {prop}"

    service_ids = {r["id"] for r in snap["nodes"]["Service"]}
    pod_ids = {r["id"] for r in snap["nodes"]["Pod"]}
    node_ids = {r["id"] for r in snap["nodes"]["Node"]}

    # Exactly one INSTANCE_OF and one RUNS_ON per Pod.
    inst = snap["relationships"]["INSTANCE_OF"]
    runs = snap["relationships"]["RUNS_ON"]
    assert sorted(r["source"] for r in inst) == sorted(pod_ids)
    assert sorted(r["source"] for r in runs) == sorted(pod_ids)
    assert all(r["target"] in service_ids for r in inst)
    assert all(r["target"] in node_ids for r in runs)

    dep_pairs = {(r["source"], r["target"]) for r in snap["relationships"]["DEPENDS_ON"]}
    call_pairs = {(r["source"], r["target"]) for r in snap["relationships"]["CALLS"]}
    assert call_pairs, "no CALLS edges emitted"
    assert call_pairs <= dep_pairs, "CALLS is not a subset of DEPENDS_ON"

    for r in snap["relationships"]["CALLS"]:
        assert "p99_latency_ms" in r and "error_rate" in r
        assert r["p99_latency_ms"] >= 0.0
        assert 0.0 <= r["error_rate"] <= 1.0


def test_snapshot_only_lists_alive_pods():
    env = ClusterEnv()
    env.reset(seed=21)
    expected = int(env.pod_alive.sum())
    assert len(env.graph_snapshot()["nodes"]["Pod"]) == expected
