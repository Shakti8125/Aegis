"""GAE(lambda) correctness, including the terminal-vs-truncation distinction.

The time-limit bug (treating ``max_cycles`` truncation as a real terminal) is
silent - training still runs, the numbers just come out subtly wrong - so it
gets hand-computed fixtures rather than a smoke test.
"""

from __future__ import annotations

import numpy as np
import pytest

from marl.mappo import compute_gae


GAMMA = 0.9
LAM = 0.5


def _call(rewards, values, terminated, truncated, final_values, last_values,
          gamma=GAMMA, lam=LAM):
    return compute_gae(
        rewards=np.asarray(rewards, dtype=np.float32),
        values=np.asarray(values, dtype=np.float32),
        terminated=np.asarray(terminated, dtype=bool),
        truncated=np.asarray(truncated, dtype=bool),
        final_values=np.asarray(final_values, dtype=np.float32),
        last_values=np.asarray(last_values, dtype=np.float32),
        gamma=gamma,
        gae_lambda=lam,
    )


def test_gae_matches_a_hand_computed_case():
    # T = 3, no episode boundaries, bootstrap V(s_3) = 4.
    #   delta_2 = 1 + 0.9*4 - 3 = 1.6                 -> A_2 = 1.6
    #   delta_1 = 1 + 0.9*3 - 2 = 1.7                 -> A_1 = 1.7 + 0.45*1.6
    #   delta_0 = 1 + 0.9*2 - 1 = 1.8                 -> A_0 = 1.8 + 0.45*A_1
    adv, ret = _call(
        rewards=[[1.0], [1.0], [1.0]],
        values=[[1.0], [2.0], [3.0]],
        terminated=[[False], [False], [False]],
        truncated=[[False], [False], [False]],
        final_values=[[0.0], [0.0], [0.0]],
        last_values=[4.0],
    )
    a2 = 1.6
    a1 = 1.7 + GAMMA * LAM * a2
    a0 = 1.8 + GAMMA * LAM * a1
    np.testing.assert_allclose(adv[:, 0], [a0, a1, a2], rtol=1e-6)
    # returns = advantages + values, the standard GAE value target.
    np.testing.assert_allclose(ret[:, 0], [a0 + 1.0, a1 + 2.0, a2 + 3.0], rtol=1e-6)


def test_termination_bootstraps_zero():
    # Episode terminates at t=0. There is no successor, so delta_0 = r - V(s_0)
    # regardless of what final_values or values[1] happen to hold.
    adv, _ = _call(
        rewards=[[2.0], [0.0]],
        values=[[1.0], [50.0]],
        terminated=[[True], [False]],
        truncated=[[False], [False]],
        final_values=[[99.0], [0.0]],  # deliberately absurd; must be ignored
        last_values=[7.0],
    )
    assert adv[0, 0] == pytest.approx(2.0 - 1.0)


def test_truncation_bootstraps_the_final_value():
    # Same trajectory, but the episode was cut off by max_cycles instead. The
    # MDP continues, so V(s_final) IS the bootstrap.
    final_v = 5.0
    adv, _ = _call(
        rewards=[[2.0], [0.0]],
        values=[[1.0], [50.0]],
        terminated=[[False], [False]],
        truncated=[[True], [False]],
        final_values=[[final_v], [0.0]],
        last_values=[7.0],
    )
    assert adv[0, 0] == pytest.approx(2.0 + GAMMA * final_v - 1.0)


def test_truncation_and_termination_differ():
    """The whole point: identical rollouts must NOT produce identical targets."""
    common = dict(
        rewards=[[2.0], [0.0]],
        values=[[1.0], [3.0]],
        final_values=[[5.0], [0.0]],
        last_values=[7.0],
    )
    term_adv, _ = _call(
        terminated=[[True], [False]], truncated=[[False], [False]], **common
    )
    trunc_adv, _ = _call(
        terminated=[[False], [False]], truncated=[[True], [False]], **common
    )
    assert term_adv[0, 0] != pytest.approx(trunc_adv[0, 0])
    assert trunc_adv[0, 0] > term_adv[0, 0]


@pytest.mark.parametrize("kind", ["terminated", "truncated"])
def test_the_lambda_chain_never_crosses_an_episode_boundary(kind):
    """A_t must not inherit A_{t+1} from a *different* episode.

    The env auto-resets, so step t+1 belongs to a fresh episode whenever the
    episode ended at t. Bleeding the advantage across that seam is the same
    class of bug as bootstrapping through it.
    """
    terminated = [[kind == "terminated"], [False], [False]]
    truncated = [[kind == "truncated"], [False], [False]]
    adv, _ = _call(
        rewards=[[1.0], [10.0], [10.0]],
        values=[[1.0], [0.0], [0.0]],
        terminated=terminated,
        truncated=truncated,
        final_values=[[2.0], [0.0], [0.0]],
        last_values=[0.0],
    )
    boot = 0.0 if kind == "terminated" else GAMMA * 2.0
    # Exactly delta_0 - no gamma*lambda*A_1 term.
    assert adv[0, 0] == pytest.approx(1.0 + boot - 1.0)


def test_lambda_one_recovers_the_monte_carlo_return():
    """With lambda = 1 and a terminal at the end, returns are the discounted sum."""
    rewards = [[1.0], [2.0], [3.0]]
    adv, ret = _call(
        rewards=rewards,
        values=[[0.5], [0.25], [0.125]],
        terminated=[[False], [False], [True]],
        truncated=[[False], [False], [False]],
        final_values=[[0.0], [0.0], [0.0]],
        last_values=[0.0],
        lam=1.0,
    )
    g0 = 1.0 + GAMMA * 2.0 + GAMMA**2 * 3.0
    g1 = 2.0 + GAMMA * 3.0
    np.testing.assert_allclose(ret[:, 0], [g0, g1, 3.0], rtol=1e-5)


def test_shapes_broadcast_over_envs_and_agents():
    t, e, n = 5, 3, 4
    rng = np.random.default_rng(0)
    adv, ret = _call(
        rewards=rng.normal(size=(t, e, n)),
        values=rng.normal(size=(t, e, n)),
        terminated=np.zeros((t, e, n), dtype=bool),
        truncated=np.zeros((t, e, n), dtype=bool),
        final_values=np.zeros((t, e, n)),
        last_values=rng.normal(size=(e, n)),
    )
    assert adv.shape == (t, e, n) and ret.shape == (t, e, n)
    assert np.all(np.isfinite(adv)) and np.all(np.isfinite(ret))


def test_per_agent_rewards_produce_per_agent_advantages():
    """Agents share V(s) but not rewards, so their advantages must differ."""
    adv, _ = _call(
        rewards=[[1.0, 5.0]],
        values=[[2.0, 2.0]],
        terminated=[[True, True]],
        truncated=[[False, False]],
        final_values=[[0.0, 0.0]],
        last_values=[0.0, 0.0],
    )
    assert adv[0, 0] == pytest.approx(-1.0)
    assert adv[0, 1] == pytest.approx(3.0)
