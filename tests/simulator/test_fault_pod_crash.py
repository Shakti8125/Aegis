"""POD_CRASH fixture: crash lands on schedule, capacity drops, restart recovers it."""

from __future__ import annotations

import numpy as np
import pytest

from simulator.cluster_env import (
    ACTION_NOOP,
    ACTION_RESTART,
    ClusterEnv,
    make_env,
)
from simulator.fault_injection import FaultEvent, FaultType


CRASH_TICK = 6
CRASH_DURATION = 60
TARGET_SERVICE = 3
TARGET_SLOT = 0


def _env_with_single_crash(**overrides) -> ClusterEnv:
    """Deterministic single-fault fixture: replace the drawn schedule outright."""
    kwargs = {"n_faults_range": (0, 0), "max_cycles": 200}
    kwargs.update(overrides)
    env = make_env(**kwargs)
    env.reset(seed=2024)
    env.fault_schedule = [
        FaultEvent(
            tick=CRASH_TICK,
            fault_type=FaultType.POD_CRASH,
            target=(TARGET_SERVICE, TARGET_SLOT),
            duration=CRASH_DURATION,
            magnitude=1.0,
        )
    ]
    env._next_fault = 0
    return env


def _noop(env: ClusterEnv):
    return {a: ACTION_NOOP for a in env.agents}


def test_crash_fires_exactly_at_its_scheduled_tick():
    env = _env_with_single_crash()
    for _ in range(CRASH_TICK - 1):
        env.step(_noop(env))
    assert env.t == CRASH_TICK - 1
    assert env.pod_up[TARGET_SERVICE, TARGET_SLOT], "pod went down early"
    assert not env.pod_crashed.any()

    env.step(_noop(env))
    assert env.t == CRASH_TICK
    assert env.pod_crashed[TARGET_SERVICE, TARGET_SLOT]
    assert not env.pod_up[TARGET_SERVICE, TARGET_SLOT]
    assert env.pod_alive[TARGET_SERVICE, TARGET_SLOT], "crash must not delete the replica"
    assert env.fired_faults and env.fired_faults[0].tick == CRASH_TICK


def test_crash_takes_the_pod_out_of_service_capacity():
    env = _env_with_single_crash()
    for _ in range(CRASH_TICK - 1):
        env.step(_noop(env))
    cap_before = float(env.svc_capacity[TARGET_SERVICE])
    ready_before = int(env.svc_ready[TARGET_SERVICE])

    env.step(_noop(env))
    assert env.svc_ready[TARGET_SERVICE] == ready_before - 1
    assert env.svc_capacity[TARGET_SERVICE] < cap_before
    assert env.svc_replicas[TARGET_SERVICE] == env.cfg.init_replicas
    assert env.pod_health[TARGET_SERVICE, TARGET_SLOT] == 0.0


def test_crash_degrades_only_the_crashed_service_directly():
    env = _env_with_single_crash()
    for _ in range(CRASH_TICK):
        env.step(_noop(env))
    others = [s for s in range(env.n_services) if s != TARGET_SERVICE]
    assert env.svc_health[TARGET_SERVICE] < 1.0
    # Downstream services (the ones the crashed service calls) are untouched;
    # upstream ones may inherit latency, so only assert on ready-pod counts.
    for s in others:
        assert env.svc_ready[s] == env.cfg.init_replicas


def test_crash_is_visible_in_the_graph_snapshot():
    env = _env_with_single_crash()
    for _ in range(CRASH_TICK):
        env.step(_noop(env))
    snap = env.graph_snapshot()
    pod_id = env.topology.pod_names[TARGET_SERVICE * env.max_replicas + TARGET_SLOT]
    row = next(p for p in snap["nodes"]["Pod"] if p["id"] == pod_id)
    assert row["status"] == "CrashLoopBackOff"
    assert row["health"] == 0.0
    assert snap["active_faults"], "active fault not reported"
    assert snap["active_faults"][0]["type"] == "POD_CRASH"


def test_restart_recovers_the_pod_after_restart_latency():
    env = _env_with_single_crash()
    latency = env.cfg.restart_latency
    for _ in range(CRASH_TICK):
        env.step(_noop(env))
    restarts_before = int(env.pod_restarts[TARGET_SERVICE, TARGET_SLOT])

    actions = _noop(env)
    actions[f"service_{TARGET_SERVICE}"] = ACTION_RESTART
    env.step(actions)
    restart_tick = env.t

    assert env.pod_restarts[TARGET_SERVICE, TARGET_SLOT] == restarts_before + 1
    assert not env.pod_crashed[TARGET_SERVICE, TARGET_SLOT], "restart should clear the crash"
    assert env.pod_down_timer[TARGET_SERVICE, TARGET_SLOT] == latency

    for _ in range(latency - 1):
        env.step(_noop(env))
        assert not env.pod_up[TARGET_SERVICE, TARGET_SLOT], (
            f"pod came back too early at t={env.t}"
        )

    env.step(_noop(env))
    assert env.t == restart_tick + latency
    assert env.pod_up[TARGET_SERVICE, TARGET_SLOT]
    assert env.pod_health[TARGET_SERVICE, TARGET_SLOT] == pytest.approx(1.0, abs=0.05)
    assert env.svc_ready[TARGET_SERVICE] == env.cfg.init_replicas


def test_restart_beats_waiting_out_the_crash():
    latency = ClusterEnv().cfg.restart_latency
    healer = _env_with_single_crash()
    idler = _env_with_single_crash()
    for _ in range(CRASH_TICK):
        healer.step(_noop(healer))
        idler.step(_noop(idler))

    actions = _noop(healer)
    actions[f"service_{TARGET_SERVICE}"] = ACTION_RESTART
    healer.step(actions)
    idler.step(_noop(idler))
    for _ in range(latency + 2):
        healer.step(_noop(healer))
        idler.step(_noop(idler))

    assert healer.pod_up[TARGET_SERVICE, TARGET_SLOT]
    assert not idler.pod_up[TARGET_SERVICE, TARGET_SLOT], (
        "crash duration is too short for this fixture to be meaningful"
    )
    assert healer.svc_health[TARGET_SERVICE] > idler.svc_health[TARGET_SERVICE]


def test_restart_increments_restart_count_in_the_snapshot():
    env = _env_with_single_crash()
    for _ in range(CRASH_TICK):
        env.step(_noop(env))
    actions = _noop(env)
    actions[f"service_{TARGET_SERVICE}"] = ACTION_RESTART
    env.step(actions)

    snap = env.graph_snapshot()
    svc = snap["nodes"]["Service"][TARGET_SERVICE]
    assert svc["restart_count"] == 1
    pod_id = env.topology.pod_names[TARGET_SERVICE * env.max_replicas + TARGET_SLOT]
    pod = next(p for p in snap["nodes"]["Pod"] if p["id"] == pod_id)
    assert pod["restart_count"] == 1
    assert pod["status"] == "Restarting"


def test_scale_up_also_restores_capacity():
    env = _env_with_single_crash()
    for _ in range(CRASH_TICK):
        env.step(_noop(env))
    cap_crashed = float(env.svc_capacity[TARGET_SERVICE])

    actions = _noop(env)
    actions[f"service_{TARGET_SERVICE}"] = 2  # scale_up
    env.step(actions)
    for _ in range(env.cfg.scale_latency + 1):
        env.step(_noop(env))

    assert env.svc_replicas[TARGET_SERVICE] == env.cfg.init_replicas + 1
    assert env.svc_capacity[TARGET_SERVICE] > cap_crashed


def test_crash_self_heals_when_its_duration_expires():
    env = _env_with_single_crash(max_cycles=400)
    for _ in range(CRASH_TICK + CRASH_DURATION - 1):
        env.step(_noop(env))
    assert not env.pod_up[TARGET_SERVICE, TARGET_SLOT]
    env.step(_noop(env))
    assert env.pod_up[TARGET_SERVICE, TARGET_SLOT]
    assert not env.pod_crashed.any()


def test_scheduled_crash_from_the_seeded_rng_lands_on_a_real_pod():
    env = make_env(enabled_faults=(FaultType.POD_CRASH,), n_faults_range=(3, 3))
    env.reset(seed=4711)
    assert len(env.fault_schedule) == 3
    for ev in env.fault_schedule:
        assert ev.fault_type is FaultType.POD_CRASH
        s, r = ev.target
        assert 0 <= s < env.n_services
        assert env.topology.initial_alive[s, r], "crash targeted a slot that is never alive"

    first = env.fault_schedule[0]
    while env.t < first.tick:
        env.step(_noop(env))
    assert env.pod_crashed[first.target], "scheduled crash did not land"
