"""Same seed + same actions -> bit-identical trajectory. Different seeds diverge.

This is Phase 1's headline guarantee: PLAN.md's "done when" clause requires
fault scenarios to be reproducible from a seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from simulator.cluster_env import N_ACTIONS, ClusterEnv, make_env
from simulator.fault_injection import FaultType

ALL_FAULTS = tuple(FaultType)


def _schedule_tuples(env: ClusterEnv):
    return [
        (e.tick, int(e.fault_type), e.target, e.duration, round(e.magnitude, 9))
        for e in env.fault_schedule
    ]


def _rollout(env: ClusterEnv, seed: int, actions: list[dict[str, int]]):
    obs, infos = env.reset(seed=seed)
    trace = [
        (
            np.stack([obs[a] for a in env.possible_agents]),
            np.zeros(len(env.possible_agents), dtype=np.float32),
        )
    ]
    for act in actions:
        if not env.agents:
            break
        o, r, term, trunc, inf = env.step({a: act[a] for a in env.agents})
        trace.append(
            (
                np.stack([o[a] for a in env.possible_agents]),
                np.array([r[a] for a in env.possible_agents], dtype=np.float32),
            )
        )
    return trace


def _action_sequence(n_steps: int, agents: list[str], seed: int):
    rng = np.random.default_rng(seed)
    return [
        {a: int(x) for a, x in zip(agents, rng.integers(0, N_ACTIONS, len(agents)))}
        for _ in range(n_steps)
    ]


@pytest.mark.parametrize("faults", [(FaultType.POD_CRASH,), ALL_FAULTS])
def test_same_seed_same_actions_is_bit_identical(faults):
    kwargs = dict(enabled_faults=faults, n_faults_range=(3, 6))
    e1, e2 = make_env(**kwargs), make_env(**kwargs)
    acts = _action_sequence(120, list(e1.possible_agents), 99)

    t1 = _rollout(e1, 4242, acts)
    t2 = _rollout(e2, 4242, acts)

    assert _schedule_tuples(e1) == _schedule_tuples(e2)
    assert len(t1) == len(t2)
    for i, ((o1, r1), (o2, r2)) in enumerate(zip(t1, t2)):
        np.testing.assert_array_equal(o1, o2, err_msg=f"observations diverged at {i}")
        np.testing.assert_array_equal(r1, r2, err_msg=f"rewards diverged at {i}")
    assert e1.t == e2.t
    assert e1.terminal_reason == e2.terminal_reason


@pytest.mark.parametrize("faults", [(FaultType.POD_CRASH,), ALL_FAULTS])
def test_full_step_tuples_are_identical(faults):
    kwargs = dict(enabled_faults=faults, n_faults_range=(2, 5))
    e1, e2 = make_env(**kwargs), make_env(**kwargs)
    e1.reset(seed=77)
    e2.reset(seed=77)
    rng = np.random.default_rng(5)
    while e1.agents:
        act = {a: int(x) for a, x in zip(e1.agents, rng.integers(0, N_ACTIONS, len(e1.agents)))}
        o1, r1, t1, c1, i1 = e1.step(act)
        o2, r2, t2, c2, i2 = e2.step(act)
        assert r1 == r2
        assert t1 == t2 and c1 == c2
        assert i1 == i2
        for a in o1:
            np.testing.assert_array_equal(o1[a], o2[a])
    assert not e2.agents


@pytest.mark.parametrize("faults", [(FaultType.POD_CRASH,), ALL_FAULTS])
def test_different_seeds_diverge(faults):
    kwargs = dict(enabled_faults=faults, n_faults_range=(3, 6))
    e1, e2 = make_env(**kwargs), make_env(**kwargs)
    acts = _action_sequence(80, list(e1.possible_agents), 3)
    t1 = _rollout(e1, 1, acts)
    t2 = _rollout(e2, 2, acts)

    schedules_differ = _schedule_tuples(e1) != _schedule_tuples(e2)
    obs_differ = any(
        not np.array_equal(a[0], b[0]) for a, b in zip(t1, t2)
    ) or len(t1) != len(t2)
    assert schedules_differ, "two different seeds drew the same fault schedule"
    assert obs_differ, "two different seeds produced identical observations"


@pytest.mark.parametrize("faults", [(FaultType.POD_CRASH,), ALL_FAULTS])
def test_reseeding_the_same_env_replays_the_same_episode(faults):
    env = make_env(enabled_faults=faults, n_faults_range=(2, 5))
    first = None
    for _ in range(3):
        obs, _ = env.reset(seed=31337)
        snap = (_schedule_tuples(env), np.stack([obs[a] for a in env.possible_agents]))
        if first is None:
            first = snap
        else:
            assert snap[0] == first[0]
            np.testing.assert_array_equal(snap[1], first[1])
        # Burn some steps so the RNG state would drift if reset were not exact.
        for _ in range(7):
            env.step({a: 1 for a in env.agents})


def test_bare_reset_advances_to_a_new_but_reproducible_episode():
    e1, e2 = make_env(n_faults_range=(2, 5)), make_env(n_faults_range=(2, 5))
    e1.reset(seed=808)
    e2.reset(seed=808)
    seen = [_schedule_tuples(e1)]
    for _ in range(4):
        e1.reset()
        e2.reset()
        assert _schedule_tuples(e1) == _schedule_tuples(e2), "bare reset is not reproducible"
        seen.append(_schedule_tuples(e1))
    assert any(s != seen[0] for s in seen[1:]), "bare reset never changed the episode"


@pytest.mark.parametrize("faults", [(FaultType.POD_CRASH,), ALL_FAULTS])
def test_fault_schedule_is_sorted_and_inside_the_window(faults):
    env = make_env(enabled_faults=faults, n_faults_range=(4, 8))
    env.reset(seed=1001)
    ticks = [e.tick for e in env.fault_schedule]
    assert ticks == sorted(ticks)
    assert env.fault_schedule, "no faults were scheduled"
    assert min(ticks) >= env.cfg.fault_start_tick
    assert max(ticks) <= int(env.max_cycles * env.cfg.fault_window_frac)
    assert {e.fault_type for e in env.fault_schedule} <= set(faults)


@pytest.mark.parametrize("faults", [(FaultType.POD_CRASH,), ALL_FAULTS])
def test_every_scheduled_fault_actually_fires(faults):
    env = make_env(enabled_faults=faults, n_faults_range=(4, 8), max_cycles=400)
    env.reset(seed=555)
    scheduled = list(env.fault_schedule)
    while env.agents:
        env.step({a: 0 for a in env.agents})
    assert env.fired_faults == scheduled
