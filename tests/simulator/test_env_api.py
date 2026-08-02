"""PettingZoo ParallelEnv conformance plus the shape guarantees Phase 4 relies on."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces
from pettingzoo.test import parallel_api_test, parallel_seed_test

from simulator.cluster_env import N_ACTIONS, ClusterConfig, ClusterEnv, make_env


def test_parallel_api_conformance():
    env = make_env()
    parallel_api_test(env, num_cycles=250)


def test_parallel_seed_conformance():
    parallel_seed_test(lambda: make_env(), num_cycles=120)


def test_spaces_are_stable_objects():
    env = make_env()
    env.reset(seed=0)
    for agent in env.possible_agents:
        assert env.observation_space(agent) is env.observation_space(agent)
        assert env.action_space(agent) is env.action_space(agent)


def test_space_shapes():
    cfg = ClusterConfig()
    env = ClusterEnv(cfg)
    obs, infos = env.reset(seed=3)

    assert len(env.possible_agents) == cfg.n_services
    assert env.obs_dim == 35 + cfg.n_tiers

    for agent in env.possible_agents:
        space = env.observation_space(agent)
        assert isinstance(space, spaces.Box)
        assert space.shape == (env.obs_dim,)
        assert space.dtype == np.float32
        assert float(space.low.min()) == 0.0
        assert float(space.high.max()) == 1.0

        act = env.action_space(agent)
        assert isinstance(act, spaces.Discrete)
        assert act.n == N_ACTIONS

        vec = obs[agent]
        assert vec.shape == (env.obs_dim,)
        assert vec.dtype == np.float32
        assert space.contains(vec)


def test_state_shape_and_bounds():
    env = make_env()
    env.reset(seed=3)
    st = env.state()
    assert st.shape == (env.state_dim,)
    assert st.dtype == np.float32
    assert env.state_space.contains(st)

    rng = np.random.default_rng(0)
    for _ in range(60):
        if not env.agents:
            break
        env.step({a: int(rng.integers(0, N_ACTIONS)) for a in env.agents})
        st = env.state()
        assert env.state_space.contains(st), "state left its declared Box"


def test_observations_stay_inside_the_box_under_random_play():
    env = make_env()
    rng = np.random.default_rng(11)
    for ep in range(4):
        obs, _ = env.reset(seed=100 + ep)
        while env.agents:
            actions = {a: int(rng.integers(0, N_ACTIONS)) for a in env.agents}
            obs, _, _, _, _ = env.step(actions)
            for agent, vec in obs.items():
                assert np.all(np.isfinite(vec))
                assert env.observation_space(agent).contains(vec)


def test_agent_roster_is_constant_within_an_episode():
    env = make_env()
    env.reset(seed=9)
    roster = list(env.agents)
    assert roster == env.possible_agents

    rng = np.random.default_rng(2)
    steps = 0
    while env.agents:
        assert list(env.agents) == roster, "agent roster changed mid-episode"
        env.step({a: int(rng.integers(0, N_ACTIONS)) for a in env.agents})
        steps += 1
        if env.agents:
            assert list(env.agents) == roster
    assert steps > 0
    # PettingZoo requires the roster to empty exactly at episode end.
    assert env.agents == []


def test_all_agents_terminate_or_truncate_simultaneously():
    env = make_env()
    env.reset(seed=13)
    rng = np.random.default_rng(4)
    while env.agents:
        _, _, term, trunc, _ = env.step(
            {a: int(rng.integers(0, N_ACTIONS)) for a in env.agents}
        )
        assert len(set(term.values())) == 1, "terminations disagree across agents"
        assert len(set(trunc.values())) == 1, "truncations disagree across agents"
    assert env.terminal_reason in {"recovered", "collapsed", "truncated"}


def test_truncation_happens_at_max_cycles():
    # No faults -> nothing to recover from and no terminal condition, so the
    # only way out is truncation at max_cycles.
    env = make_env(n_faults_range=(0, 0), max_cycles=25)
    env.reset(seed=1)
    steps = 0
    while env.agents:
        _, _, term, trunc, _ = env.step({a: 0 for a in env.agents})
        steps += 1
    assert steps == 25
    assert env.terminal_reason == "truncated"
    assert all(trunc.values())
    assert not any(term.values())


def test_infos_carry_reward_components_and_invalid_action_counter():
    env = make_env()
    _, infos = env.reset(seed=6)
    expected = {
        "sla_violation",
        "latency",
        "availability",
        "action_cost",
        "invalid_action",
        "terminal",
    }
    for info in infos.values():
        assert set(info["reward_components"]) == expected

    _, rewards, _, _, infos = env.step({a: 0 for a in env.agents})
    for agent, info in infos.items():
        comps = info["reward_components"]
        assert set(comps) == expected
        assert rewards[agent] == pytest.approx(sum(comps.values()), rel=1e-5, abs=1e-5)
        assert isinstance(info["invalid_action"], int)


def test_invalid_actions_degrade_to_noop_and_are_counted():
    # min_replicas == max_replicas: every scale action is illegal.
    env = make_env(min_replicas=2, init_replicas=2, max_replicas=2, n_faults_range=(0, 0))
    env.reset(seed=2)
    before = env.pod_alive.copy()
    _, _, _, _, infos = env.step({a: 3 for a in env.agents})  # scale_down
    np.testing.assert_array_equal(env.pod_alive, before)
    for info in infos.values():
        assert info["invalid_action"] == 1
        assert info["invalid_action_now"] is True
        assert info["reward_components"]["action_cost"] == 0.0
        assert info["reward_components"]["invalid_action"] < 0.0


def test_out_of_range_action_does_not_raise():
    env = make_env()
    env.reset(seed=0)
    _, _, _, _, infos = env.step({a: 99 for a in env.agents})
    assert all(i["invalid_action_now"] for i in infos.values())


def test_render_returns_text():
    env = make_env(render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert isinstance(text, str) and "tick=" in text
