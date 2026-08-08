"""Reward components stay separate, stay addressable, and add back up.

CLAUDE.md's first non-negotiable is that no reward component is ever collapsed
into a single scalar. These tests are what keeps that true as the code moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from marl.reward import (
    COMPONENT_NAMES,
    UNIT_ENV_WEIGHTS,
    RewardAccumulator,
    RewardConfig,
    RewardShaper,
)
from marl.vec_env import make_scenario_env
from simulator.cluster_env import ACTION_COST, ACTION_ISOLATE, ACTION_NOOP


EXPECTED = {
    "sla_violation",
    "latency",
    "availability",
    "action_cost",
    "invalid_action",
    "terminal",
}


def test_component_names_match_the_environments_contract():
    assert set(COMPONENT_NAMES) == EXPECTED
    env = make_scenario_env("mixed")
    _, infos = env.reset(seed=0)
    assert set(infos[env.agents[0]]["reward_components"]) == EXPECTED


def test_unit_weights_make_the_env_report_raw_signals():
    """With unit weights the env's components are the physical quantities."""
    env = make_scenario_env("mixed", max_cycles=50)
    for key, value in UNIT_ENV_WEIGHTS.items():
        assert getattr(env.cfg, key) == value

    env.reset(seed=7)
    agents = list(env.agents)
    _, _, _, _, infos = env.step({a: ACTION_NOOP for a in agents})
    comps = infos[agents[0]]["reward_components"]
    assert comps["sla_violation"] == pytest.approx(-env.sla_violation_rate, abs=1e-6)
    assert comps["availability"] == pytest.approx(float(env.svc_health.mean()), abs=1e-5)


def test_shaped_total_is_exactly_the_sum_of_its_components():
    shaper = RewardShaper()
    env = make_scenario_env("mixed", max_cycles=40)
    env.reset(seed=11)
    agents = list(env.agents)
    rng = np.random.default_rng(3)
    while env.agents:
        actions = {a: int(rng.integers(0, 6)) for a in env.agents}
        _, _, _, _, infos = env.step(actions)
        total, comps = shaper.shape_infos(infos, agents)
        assert set(comps) == EXPECTED
        stacked = np.sum([comps[k] for k in COMPONENT_NAMES], axis=0)
        np.testing.assert_allclose(total, stacked, rtol=1e-5, atol=1e-6)
        assert np.all(np.isfinite(total))


def test_each_component_is_separately_addressable_and_independently_weighted():
    """Changing one weight moves exactly one component."""
    raw = {
        "sla_violation": np.array([-0.5], dtype=np.float32),
        "latency": np.array([-0.2], dtype=np.float32),
        "availability": np.array([0.4], dtype=np.float32),
        "action_cost": np.array([-0.25], dtype=np.float32),
        "invalid_action": np.array([-1.0], dtype=np.float32),
        "terminal": np.array([0.0], dtype=np.float32),
    }
    base = RewardConfig()
    _, before = RewardShaper(base).shape(raw)
    bumped = RewardConfig(**{**base.as_dict(), "w_sla": base.w_sla * 2.0})
    _, after = RewardShaper(bumped).shape(raw)

    assert after["sla_violation"][0] == pytest.approx(2.0 * before["sla_violation"][0])
    for key in COMPONENT_NAMES:
        if key == "sla_violation":
            continue
        assert after[key][0] == pytest.approx(before[key][0]), f"{key} moved"


def test_availability_is_a_shortfall_penalty_not_a_survival_bonus():
    """A perfectly healthy cluster scores 0, never a positive per-tick payout.

    A positive availability term rewards *not* recovering (keep collecting it),
    which is the exact reward hack this re-baselining exists to remove.
    """
    shaper = RewardShaper()
    healthy = {k: np.zeros(1, dtype=np.float32) for k in COMPONENT_NAMES}
    healthy["availability"] = np.ones(1, dtype=np.float32)
    _, comps = shaper.shape(healthy)
    assert comps["availability"][0] == pytest.approx(0.0)

    degraded = dict(healthy)
    degraded["availability"] = np.full(1, 0.5, dtype=np.float32)
    _, comps = shaper.shape(degraded)
    assert comps["availability"][0] < 0.0


def test_dense_components_are_never_positive():
    """Only `terminal` may be positive; everything else is a cost."""
    rng = np.random.default_rng(0)
    raw = {
        "sla_violation": -rng.random(32).astype(np.float32),
        "latency": -rng.random(32).astype(np.float32),
        "availability": rng.random(32).astype(np.float32),
        "action_cost": -rng.choice(ACTION_COST, size=32).astype(np.float32),
        "invalid_action": -(rng.random(32) < 0.3).astype(np.float32),
        "terminal": np.zeros(32, dtype=np.float32),
    }
    _, comps = RewardShaper().shape(raw)
    for key in ("sla_violation", "latency", "availability", "action_cost", "invalid_action"):
        assert np.all(comps[key] <= 1e-7), key


def test_terminal_weights_are_asymmetric_and_signed_correctly():
    cfg = RewardConfig()
    shaper = RewardShaper(cfg)
    zeros = {k: np.zeros(1, dtype=np.float32) for k in COMPONENT_NAMES}
    zeros["availability"] = np.ones(1, dtype=np.float32)

    recovered = dict(zeros, terminal=np.ones(1, dtype=np.float32))
    collapsed = dict(zeros, terminal=-np.ones(1, dtype=np.float32))
    _, rec = shaper.shape(recovered)
    _, col = shaper.shape(collapsed)
    assert rec["terminal"][0] == pytest.approx(cfg.w_recovered)
    assert col["terminal"][0] == pytest.approx(-cfg.w_collapsed)
    # The collapse penalty has to outweigh the dense stream it cuts short,
    # otherwise "destroy the cluster early" is a profitable exploit.
    assert cfg.w_collapsed > cfg.w_recovered


def test_action_cost_reaches_the_reward_and_is_charged_to_the_acting_agent():
    env = make_scenario_env("mixed", max_cycles=30)
    env.reset(seed=5)
    agents = list(env.agents)
    actions = {a: ACTION_NOOP for a in agents}
    actions[agents[0]] = ACTION_ISOLATE
    _, _, _, _, infos = env.step(actions)
    _, comps = RewardShaper().shape_infos(infos, agents)
    assert comps["action_cost"][0] < 0.0
    assert np.all(comps["action_cost"][1:] == 0.0)


def test_accumulator_keeps_components_separate():
    acc = RewardAccumulator()
    comps = {k: np.full(4, float(i + 1), dtype=np.float32)
             for i, k in enumerate(COMPONENT_NAMES)}
    acc.add(comps)
    acc.add(comps)
    sums = acc.sums()
    for i, key in enumerate(COMPONENT_NAMES):
        assert sums[key] == pytest.approx(2 * 4 * (i + 1))
    assert sums["total"] == pytest.approx(sum(sums[k] for k in COMPONENT_NAMES))
    assert acc.samples == 8

    means = acc.mean_per_sample()
    for i, key in enumerate(COMPONENT_NAMES):
        assert means[key] == pytest.approx(float(i + 1))


def test_env_scalar_reward_still_equals_its_own_component_sum():
    """Sanity check on the boundary we depend on: the env is self-consistent."""
    env = make_scenario_env("mixed", max_cycles=25)
    env.reset(seed=2)
    rng = np.random.default_rng(1)
    while env.agents:
        _, rewards, _, _, infos = env.step(
            {a: int(rng.integers(0, 6)) for a in env.agents}
        )
        for agent, reward in rewards.items():
            total = sum(infos[agent]["reward_components"].values())
            assert reward == pytest.approx(total, rel=1e-5, abs=1e-5)
