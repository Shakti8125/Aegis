"""CASCADING_LATENCY: latency injected at the data tier climbs to the edge tier with decay.

The load-bearing assertion is the **tier-ordering of impact over time**.  Merely
asserting "the edge tier was eventually affected" passes just as well for a
global latency spike, which is not what this fault is.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.cluster_env import ACTION_NOOP, ClusterEnv, make_env
from simulator.fault_injection import FaultEvent, FaultType

INJECT_TICK = 5
INJECT_DURATION = 40
INJECT_MS = 400.0


def _find_chain(env: ClusterEnv) -> tuple[int, int, int]:
    """An edge -> mid -> data path along CALLS edges."""
    calls = env.topology.calls_mask
    tier = env.topology.service_tier
    for a in np.flatnonzero(tier == 0):
        for b in np.flatnonzero(calls[a]):
            if tier[b] != 1:
                continue
            for c in np.flatnonzero(calls[b]):
                if tier[c] == 2:
                    return int(a), int(b), int(c)
    raise AssertionError("topology has no edge->mid->data CALLS chain")


def _env_with_cascade(**overrides) -> tuple[ClusterEnv, tuple[int, int, int]]:
    kwargs = {"n_faults_range": (0, 0), "noise_std": 0.0, "max_cycles": 300}
    kwargs.update(overrides)
    env = make_env(**kwargs)
    env.reset(seed=2024)
    chain = _find_chain(env)
    env.fault_schedule = [
        FaultEvent(
            tick=INJECT_TICK,
            fault_type=FaultType.CASCADING_LATENCY,
            target=(chain[2],),
            duration=INJECT_DURATION,
            magnitude=INJECT_MS,
        )
    ]
    env._next_fault = 0
    return env, chain


def _noop(env: ClusterEnv):
    return {a: ACTION_NOOP for a in env.agents}


def _run_to(env: ClusterEnv, tick: int):
    while env.t < tick:
        env.step(_noop(env))


def _first_impact_ticks(env: ClusterEnv, horizon: int, threshold: float = 1.0):
    """Tick at which each service's latency first exceeds its pre-fault baseline."""
    _run_to(env, INJECT_TICK - 1)
    baseline = env.svc_latency.copy()
    first = np.full(env.n_services, -1, dtype=np.int64)
    while env.t < INJECT_TICK - 1 + horizon:
        env.step(_noop(env))
        hit = (first < 0) & (env.svc_latency - baseline > threshold)
        first[hit] = env.t
    return first, baseline


def test_injection_lands_on_a_data_tier_service():
    env, chain = _env_with_cascade()
    assert env.topology.service_tier[chain[2]] == env.cfg.n_tiers - 1
    _run_to(env, INJECT_TICK)
    assert env.svc_latency_fault[chain[2]] == pytest.approx(INJECT_MS)
    assert env.svc_latency[chain[2]] > INJECT_MS


def test_impact_reaches_tiers_in_dependency_order():
    env, chain = _env_with_cascade()
    edge, mid, data = chain
    first, _ = _first_impact_ticks(env, horizon=15)

    assert first[data] == INJECT_TICK
    assert first[mid] > 0 and first[edge] > 0, "the cascade never reached the edge tier"
    assert first[data] < first[mid] < first[edge], (
        "impact order was "
        f"data={first[data]}, mid={first[mid]}, edge={first[edge]} - "
        "this is a global spike, not a cascade"
    )
    # One dependency hop per tick.
    assert first[mid] == first[data] + 1
    assert first[edge] == first[mid] + 1


def test_impact_decays_with_distance_from_the_injection_point():
    env, chain = _env_with_cascade()
    edge, mid, data = chain
    _run_to(env, INJECT_TICK - 1)
    baseline = env.svc_latency.copy()
    _run_to(env, INJECT_TICK + 8)
    delta = env.svc_latency - baseline

    assert delta[data] > delta[mid] > delta[edge] > 0.0, (
        f"latency did not decay along the chain: {delta[[data, mid, edge]]}"
    )
    assert delta[edge] < env.cfg.cascade_decay * delta[mid] + 1e-3


def test_cascade_is_confined_to_upstream_callers():
    """Services with no CALLS path to the faulted one must not move at all."""
    env, chain = _env_with_cascade()
    data = chain[2]

    reach = env.topology.calls_mask > 0
    ancestors = np.zeros(env.n_services, dtype=bool)
    frontier = {data}
    while frontier:
        nxt = set()
        for node in frontier:
            for caller in np.flatnonzero(reach[:, node]):
                if not ancestors[caller]:
                    ancestors[caller] = True
                    nxt.add(int(caller))
        frontier = nxt
    unaffected = ~ancestors
    unaffected[data] = False
    assert unaffected.any(), "fixture needs at least one off-path service"

    _run_to(env, INJECT_TICK - 1)
    baseline = env.svc_latency.copy()
    _run_to(env, INJECT_TICK + 10)
    np.testing.assert_allclose(
        env.svc_latency[unaffected], baseline[unaffected], atol=1e-4,
        err_msg="an off-path service moved - the cascade is not following CALLS edges",
    )


def test_cascade_drives_sla_violations_upstream():
    env, chain = _env_with_cascade()
    edge, mid, data = chain
    _run_to(env, INJECT_TICK - 1)
    health_before = env.svc_health.copy()
    _run_to(env, INJECT_TICK + 10)
    assert env.svc_latency[data] > env.cfg.sla_latency_ms
    assert env.svc_latency[mid] > env.cfg.sla_latency_ms
    assert env._sla_violating[data] and env._sla_violating[mid]
    assert env.sla_violation_rate > 0.0
    # Latency-only damage: pods are all up, so health degrades but does not crater.
    assert env.svc_health[data] < health_before[data]
    assert env.svc_health[data] < 0.95
    assert env.svc_health[mid] < health_before[mid]


def test_cascade_expires_and_latency_returns_to_baseline():
    env, chain = _env_with_cascade()
    _run_to(env, INJECT_TICK - 1)
    baseline = env.svc_latency.copy()

    _run_to(env, INJECT_TICK + INJECT_DURATION - 1)
    assert env.svc_latency_fault[chain[2]] == pytest.approx(INJECT_MS)

    env.step(_noop(env))
    assert env.svc_latency_fault[chain[2]] == 0.0
    # Unwinding takes one tick per hop, exactly like the build-up.
    _run_to(env, INJECT_TICK + INJECT_DURATION + env.cfg.n_tiers + 1)
    np.testing.assert_allclose(env.svc_latency, baseline, atol=1e-3)
    assert not env._faults_active()


def test_scheduled_cascades_target_the_deepest_tier():
    env = make_env(
        enabled_faults=(FaultType.CASCADING_LATENCY,), n_faults_range=(5, 5)
    )
    env.reset(seed=777)
    assert len(env.fault_schedule) == 5
    for ev in env.fault_schedule:
        assert ev.fault_type is FaultType.CASCADING_LATENCY
        assert env.topology.service_tier[ev.target[0]] == env.cfg.n_tiers - 1
        assert env.cfg.cascade_magnitude_ms_range[0] <= ev.magnitude
        assert ev.magnitude <= env.cfg.cascade_magnitude_ms_range[1]

    first = env.fault_schedule[0]
    _run_to(env, first.tick)
    assert env.svc_latency_fault[first.target[0]] > 0.0
