"""NODE_CPU_SPIKE / NODE_MEM_SPIKE: pressure degrades every pod on that node, and only those.

Noise is switched off in these fixtures so the "only those pods" half of the
claim can be asserted exactly rather than statistically.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.cluster_env import ClusterEnv, make_env
from simulator.fault_injection import FaultEvent, FaultType

SPIKE_TICK = 5
SPIKE_DURATION = 30
SPIKE_MAGNITUDE = 0.45
TARGET_NODE = 2


def _env_with_spike(fault_type: FaultType, **overrides) -> ClusterEnv:
    kwargs = {"n_faults_range": (0, 0), "noise_std": 0.0, "max_cycles": 300}
    kwargs.update(overrides)
    env = make_env(**kwargs)
    env.reset(seed=2024)
    env.fault_schedule = [
        FaultEvent(
            tick=SPIKE_TICK,
            fault_type=fault_type,
            target=(TARGET_NODE,),
            duration=SPIKE_DURATION,
            magnitude=SPIKE_MAGNITUDE,
        )
    ]
    env._next_fault = 0
    return env


def _noop(env: ClusterEnv):
    return {a: 0 for a in env.agents}


def _run_to(env: ClusterEnv, tick: int):
    while env.t < tick:
        env.step(_noop(env))


def _masks(env: ClusterEnv):
    on_node = env.topology.pod_node == TARGET_NODE
    alive = env.pod_alive
    return on_node & alive, (~on_node) & alive


def test_fixture_actually_splits_pods_across_nodes():
    env = _env_with_spike(FaultType.NODE_CPU_SPIKE)
    on, off = _masks(env)
    assert on.sum() > 0, "no alive pods on the target node"
    assert off.sum() > 0, "every alive pod is on the target node"


@pytest.mark.parametrize(
    "fault_type,attr",
    [
        (FaultType.NODE_CPU_SPIKE, "pod_cpu"),
        (FaultType.NODE_MEM_SPIKE, "pod_mem"),
    ],
)
def test_spike_raises_utilisation_on_that_node_only(fault_type, attr):
    env = _env_with_spike(fault_type)
    _run_to(env, SPIKE_TICK - 1)
    before = getattr(env, attr).copy()
    on, off = _masks(env)

    env.step(_noop(env))
    after = getattr(env, attr)

    np.testing.assert_allclose(
        after[on] - before[on], SPIKE_MAGNITUDE, atol=1e-6,
        err_msg="pods on the spiked node did not take the full pressure",
    )
    np.testing.assert_array_equal(
        after[off], before[off],
        err_msg="a pod on an unaffected node moved - the spike is not node-local",
    )


@pytest.mark.parametrize(
    "fault_type", [FaultType.NODE_CPU_SPIKE, FaultType.NODE_MEM_SPIKE]
)
def test_spike_degrades_pod_health_on_that_node_only(fault_type):
    env = _env_with_spike(fault_type)
    _run_to(env, SPIKE_TICK - 1)
    on, off = _masks(env)
    health_before = env.pod_health.copy()
    assert np.all(health_before[on] > 0.9)

    env.step(_noop(env))
    assert np.all(env.pod_health[on] < 0.9), "spiked pods stayed healthy"
    np.testing.assert_array_equal(
        env.pod_health[off], health_before[off],
        err_msg="pods off the spiked node lost health",
    )
    # Pods stay up - a resource spike degrades, it does not evict.
    assert np.all(env.pod_up[on])


def test_cpu_spike_shows_up_on_the_node_metric_and_in_the_snapshot():
    env = _env_with_spike(FaultType.NODE_CPU_SPIKE)
    _run_to(env, SPIKE_TICK - 1)
    node_cpu_before = env.node_cpu.copy()
    env.step(_noop(env))

    assert env.node_cpu[TARGET_NODE] > node_cpu_before[TARGET_NODE] + 0.3
    others = [n for n in range(env.n_nodes) if n != TARGET_NODE]
    np.testing.assert_array_equal(env.node_cpu[others], node_cpu_before[others])

    snap = env.graph_snapshot()
    row = snap["nodes"]["Node"][TARGET_NODE]
    assert row["cpu_pct"] > 60.0
    assert row["health"] < 0.5
    assert snap["active_faults"][0]["type"] == "NODE_CPU_SPIKE"


def test_spike_degrades_only_services_that_have_pods_on_the_node():
    env = _env_with_spike(FaultType.NODE_CPU_SPIKE)
    _run_to(env, SPIKE_TICK - 1)
    hosted = env.topology.service_node_mask[:, TARGET_NODE] > 0
    # Services keep the slot mapping even when scaled down; restrict to services
    # with a *live* pod there.
    on, _ = _masks(env)
    live_hosted = on.any(axis=1)
    not_hosted = ~hosted
    assert live_hosted.any() and not_hosted.any()

    def mean_alive_health(e):
        alive = e.pod_alive
        return (e.pod_health * alive).sum(1) / np.maximum(alive.sum(1), 1)

    before = mean_alive_health(env)
    env.step(_noop(env))
    after = mean_alive_health(env)

    assert np.all(after[live_hosted] < before[live_hosted]), (
        "a service with a live pod on the spiked node was not degraded"
    )
    np.testing.assert_array_equal(
        after[not_hosted], before[not_hosted],
        err_msg="a service with no pods on the spiked node was degraded",
    )


def test_spike_expires_and_the_node_recovers():
    env = _env_with_spike(FaultType.NODE_CPU_SPIKE)
    _run_to(env, SPIKE_TICK - 1)
    cpu_baseline = env.pod_cpu.copy()
    _run_to(env, SPIKE_TICK + SPIKE_DURATION - 1)
    on, _ = _masks(env)
    assert np.all(env.pod_health[on] < 0.9), "spike ended early"
    assert env.node_cpu_pressure[TARGET_NODE] == pytest.approx(SPIKE_MAGNITUDE)

    env.step(_noop(env))
    assert env.node_cpu_pressure[TARGET_NODE] == 0.0
    assert env.node_fault_timer[TARGET_NODE] == 0
    np.testing.assert_allclose(env.pod_cpu, cpu_baseline, atol=1e-6)
    assert np.all(env.pod_health[env.pod_alive] > 0.99)
    assert not env._faults_active()


def test_scheduled_node_spikes_target_real_nodes():
    env = make_env(
        enabled_faults=(FaultType.NODE_CPU_SPIKE, FaultType.NODE_MEM_SPIKE),
        n_faults_range=(6, 6),
    )
    env.reset(seed=99)
    assert len(env.fault_schedule) == 6
    for ev in env.fault_schedule:
        assert ev.fault_type in (FaultType.NODE_CPU_SPIKE, FaultType.NODE_MEM_SPIKE)
        assert 0 <= ev.target[0] < env.n_nodes
        assert env.cfg.node_spike_magnitude_range[0] <= ev.magnitude
        assert ev.magnitude <= env.cfg.node_spike_magnitude_range[1]

    first = env.fault_schedule[0]
    _run_to(env, first.tick)
    if first.fault_type is FaultType.NODE_CPU_SPIKE:
        assert env.node_cpu_pressure[first.target[0]] > 0.0
    else:
        assert env.node_mem_pressure[first.target[0]] > 0.0
