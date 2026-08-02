"""NETWORK_PARTITION: a CALLS edge's error_rate spikes, the caller degrades, reroute/isolate mitigate."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.cluster_env import (
    ACTION_ISOLATE,
    ACTION_NOOP,
    ACTION_REROUTE,
    ClusterEnv,
    make_env,
)
from simulator.fault_injection import FaultEvent, FaultType

PARTITION_TICK = 5
PARTITION_DURATION = 40
PARTITION_MAGNITUDE = 0.9


def _pick_edge(env: ClusterEnv) -> tuple[int, int]:
    """A caller with >= 2 CALLS edges, so `reroute` has somewhere to send traffic."""
    calls = env.topology.calls_mask
    callers = np.flatnonzero(calls.sum(1) >= 2)
    assert callers.size, "fixture needs a service with at least two CALLS edges"
    caller = int(callers[0])
    callee = int(np.flatnonzero(calls[caller])[0])
    return caller, callee


def _env_with_partition(**overrides) -> tuple[ClusterEnv, int, int]:
    kwargs = {"n_faults_range": (0, 0), "noise_std": 0.0, "max_cycles": 300}
    kwargs.update(overrides)
    env = make_env(**kwargs)
    env.reset(seed=2024)
    caller, callee = _pick_edge(env)
    env.fault_schedule = [
        FaultEvent(
            tick=PARTITION_TICK,
            fault_type=FaultType.NETWORK_PARTITION,
            target=(caller, callee),
            duration=PARTITION_DURATION,
            magnitude=PARTITION_MAGNITUDE,
        )
    ]
    env._next_fault = 0
    return env, caller, callee


def _noop(env: ClusterEnv):
    return {a: ACTION_NOOP for a in env.agents}


def _run_to(env: ClusterEnv, tick: int):
    while env.t < tick:
        env.step(_noop(env))


def test_partition_targets_a_real_calls_edge():
    env, i, j = _env_with_partition()
    assert env.topology.calls_mask[i, j] > 0
    assert env.topology.depends_on[i, j] > 0


def test_partition_spikes_the_edge_error_rate():
    env, i, j = _env_with_partition()
    _run_to(env, PARTITION_TICK - 1)
    assert env.edge_fault_error[i, j] == 0.0

    env.step(_noop(env))
    assert env.edge_fault_error[i, j] == pytest.approx(PARTITION_MAGNITUDE)
    assert env.edge_fault_timer[i, j] == PARTITION_DURATION

    snap = env.graph_snapshot()
    names = env.topology.service_names
    edge = next(
        e
        for e in snap["relationships"]["CALLS"]
        if e["source"] == names[i] and e["target"] == names[j]
    )
    assert edge["error_rate"] >= PARTITION_MAGNITUDE - 1e-6
    clean = [
        e
        for e in snap["relationships"]["CALLS"]
        if not (e["source"] == names[i] and e["target"] == names[j])
    ]
    assert all(e["error_rate"] < 0.1 for e in clean), "the spike leaked onto other edges"
    assert snap["active_faults"][0]["type"] == "NETWORK_PARTITION"


def test_partition_degrades_the_dependent_service():
    env, i, j = _env_with_partition()
    _run_to(env, PARTITION_TICK - 1)
    err_before = env.svc_error.copy()
    health_before = env.svc_health.copy()
    assert err_before[i] == pytest.approx(0.0, abs=1e-6)

    env.step(_noop(env))
    assert env.svc_error[i] > env.cfg.sla_error_rate, "caller error did not rise"
    assert env.svc_health[i] < health_before[i], "caller health did not drop"
    assert env._sla_violating[i]
    # The callee itself is fine - the link broke, not the service.
    assert env.svc_error[j] == pytest.approx(err_before[j], abs=1e-6)
    assert env.svc_health[j] == pytest.approx(health_before[j], abs=1e-6)


def test_reroute_mitigates_the_partition():
    env, i, j = _env_with_partition()
    _run_to(env, PARTITION_TICK)
    damaged = float(env.svc_error[i])
    weight_before = float(env.call_weight[i, j])
    assert damaged > env.cfg.sla_error_rate

    for _ in range(4):
        actions = _noop(env)
        actions[f"service_{i}"] = ACTION_REROUTE
        env.step(actions)

    assert env.call_weight[i, j] < weight_before, "reroute did not move traffic"
    assert env.svc_error[i] < damaged / 2.0, "reroute did not mitigate the partition"
    assert env.svc_health[i] > 0.95
    # Traffic is conserved: it moved to the sibling edges, it did not vanish.
    row = env.call_weight[i][env.topology.calls_mask[i] > 0]
    assert row.sum() == pytest.approx(1.0, abs=1e-5)


def test_isolate_mitigates_the_partition():
    env, i, j = _env_with_partition()
    _run_to(env, PARTITION_TICK)
    damaged = float(env.svc_error[i])

    actions = _noop(env)
    actions[f"service_{j}"] = ACTION_ISOLATE
    env.step(actions)
    env.step(_noop(env))

    assert env.svc_isolate_timer[j] > 0
    assert env.svc_error[i] < damaged, "isolating the callee did not help the caller"
    # Isolation is not free: the isolated service fails its own requests.
    assert env.svc_error[j] == pytest.approx(1.0)


def test_doing_nothing_does_not_mitigate():
    """Guards the two mitigation tests above from passing for the wrong reason."""
    env, i, _ = _env_with_partition()
    _run_to(env, PARTITION_TICK)
    damaged = float(env.svc_error[i])
    for _ in range(4):
        env.step(_noop(env))
    assert env.svc_error[i] == pytest.approx(damaged, abs=1e-6)


def test_partition_expires_and_the_caller_recovers():
    env, i, j = _env_with_partition()
    _run_to(env, PARTITION_TICK + PARTITION_DURATION - 1)
    assert env.svc_error[i] > env.cfg.sla_error_rate

    env.step(_noop(env))
    assert env.edge_fault_error[i, j] == 0.0
    env.step(_noop(env))
    assert env.svc_error[i] == pytest.approx(0.0, abs=1e-6)
    assert env.svc_health[i] > 0.99
    assert not env._faults_active()


def test_scheduled_partitions_land_on_calls_edges():
    env = make_env(
        enabled_faults=(FaultType.NETWORK_PARTITION,), n_faults_range=(5, 5)
    )
    env.reset(seed=1234)
    assert len(env.fault_schedule) == 5
    for ev in env.fault_schedule:
        assert ev.fault_type is FaultType.NETWORK_PARTITION
        i, j = ev.target
        assert env.topology.calls_mask[i, j] > 0

    first = env.fault_schedule[0]
    _run_to(env, first.tick)
    assert env.edge_fault_error[first.target] > 0.0
