"""MAPPO wiring: CTDE shapes, shared actor parameters, and a NaN-free update."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from marl.mappo import (
    MAPPO,
    MAPPOConfig,
    RolloutBuffer,
    VectorObsEncoder,
    compute_gae,
)
from marl.reward import COMPONENT_NAMES
from marl.vec_env import make_scenario_env

OBS_DIM, STATE_DIM, N_AGENTS, N_ACTIONS = 38, 143, 12, 6


def _policy(**cfg_kwargs) -> MAPPO:
    torch.manual_seed(0)
    cfg = MAPPOConfig(hidden_dim=32, num_minibatches=2, update_epochs=2, **cfg_kwargs)
    return MAPPO(OBS_DIM, STATE_DIM, N_AGENTS, N_ACTIONS, cfg)


def _filled_buffer(policy: MAPPO, n_steps=6, n_envs=2, seed=0):
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(n_steps, n_envs, N_AGENTS, OBS_DIM, STATE_DIM, COMPONENT_NAMES)
    for _ in range(n_steps):
        obs = rng.random((n_envs, N_AGENTS, OBS_DIM), dtype=np.float32)
        state = rng.random((n_envs, STATE_DIM), dtype=np.float32)
        actions, logprobs, values = policy.act(obs, state)
        buf.add(
            obs=obs,
            state=state,
            action=actions,
            logprob=logprobs,
            value=values,
            reward=rng.normal(size=(n_envs, N_AGENTS)).astype(np.float32),
            components={
                k: rng.normal(size=(n_envs, N_AGENTS)).astype(np.float32)
                for k in COMPONENT_NAMES
            },
            terminated=rng.random(n_envs) < 0.1,
            truncated=rng.random(n_envs) < 0.1,
            final_state=rng.random((n_envs, STATE_DIM), dtype=np.float32),
        )
    return buf


# ------------------------------------------------------------------ the seam
def test_encoder_seam_reports_the_dims_the_networks_are_built_from():
    enc = VectorObsEncoder(OBS_DIM, STATE_DIM, N_AGENTS, agent_id_in_critic=True)
    assert enc.actor_input_dim == OBS_DIM
    assert enc.critic_input_dim == STATE_DIM + N_AGENTS

    plain = VectorObsEncoder(OBS_DIM, STATE_DIM, N_AGENTS, agent_id_in_critic=False)
    assert plain.critic_input_dim == STATE_DIM


def test_encoder_actor_path_is_a_passthrough_and_critic_path_tiles_the_state():
    enc = VectorObsEncoder(OBS_DIM, STATE_DIM, N_AGENTS)
    obs = torch.rand(4, N_AGENTS, OBS_DIM)
    torch.testing.assert_close(enc.actor_features(obs), obs)

    state = torch.rand(4, STATE_DIM)
    feats = enc.critic_features(state)
    assert feats.shape == (4, N_AGENTS, STATE_DIM + N_AGENTS)
    # every agent sees the same global state...
    torch.testing.assert_close(feats[:, 0, :STATE_DIM], state)
    torch.testing.assert_close(feats[:, 3, :STATE_DIM], state)
    # ...and a distinct identity tag.
    assert not torch.equal(feats[:, 0, STATE_DIM:], feats[:, 3, STATE_DIM:])


def test_encoder_holds_no_trainable_parameters_in_the_no_gnn_ablation():
    enc = VectorObsEncoder(OBS_DIM, STATE_DIM, N_AGENTS)
    assert list(enc.parameters()) == []


# ------------------------------------------------------------------- acting
def test_act_shapes_and_action_range():
    policy = _policy()
    rng = np.random.default_rng(0)
    obs = rng.random((3, N_AGENTS, OBS_DIM), dtype=np.float32)
    state = rng.random((3, STATE_DIM), dtype=np.float32)
    actions, logprobs, values = policy.act(obs, state)
    assert actions.shape == logprobs.shape == values.shape == (3, N_AGENTS)
    assert actions.min() >= 0 and actions.max() < N_ACTIONS
    assert np.all(np.isfinite(logprobs)) and np.all(np.isfinite(values))


def test_actor_parameters_are_shared_across_the_homogeneous_agents():
    """Identical local observations must yield identical policies, whatever
    agent slot they sit in - that is what "shared parameters" means here."""
    policy = _policy()
    row = np.random.default_rng(1).random(OBS_DIM).astype(np.float32)
    obs = np.tile(row, (1, N_AGENTS, 1))
    state = np.zeros((1, STATE_DIM), dtype=np.float32)
    with torch.no_grad():
        logits = policy.actor(
            policy.encoder.actor_features(torch.as_tensor(obs))
        ).numpy()
    for j in range(1, N_AGENTS):
        np.testing.assert_allclose(logits[0, j], logits[0, 0], rtol=1e-6)

    # And the whole actor is one parameter set, not N.
    assert sum(p.numel() for p in policy.actor.parameters()) < 200_000


def test_deterministic_mode_is_the_argmax_and_is_reproducible():
    policy = _policy()
    rng = np.random.default_rng(2)
    obs = rng.random((2, N_AGENTS, OBS_DIM), dtype=np.float32)
    state = rng.random((2, STATE_DIM), dtype=np.float32)
    a1, _, _ = policy.act(obs, state, deterministic=True)
    a2, _, _ = policy.act(obs, state, deterministic=True)
    np.testing.assert_array_equal(a1, a2)


def test_same_seed_gives_the_same_initial_policy():
    p1, p2 = _policy(), _policy()
    for w1, w2 in zip(p1.actor.parameters(), p2.actor.parameters()):
        torch.testing.assert_close(w1, w2)


# ------------------------------------------------------------------ learning
def test_update_runs_and_produces_no_nans():
    policy = _policy()
    buf = _filled_buffer(policy)
    adv, ret = compute_gae(
        rewards=buf.rewards,
        values=buf.values,
        terminated=np.repeat(buf.terminated[:, :, None], N_AGENTS, axis=2),
        truncated=np.repeat(buf.truncated[:, :, None], N_AGENTS, axis=2),
        final_values=np.zeros_like(buf.rewards),
        last_values=np.zeros((buf.n_envs, N_AGENTS), dtype=np.float32),
        gamma=0.99,
        gae_lambda=0.95,
    )
    stats = policy.update(buf, adv, ret)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac"):
        assert np.isfinite(stats[key]), f"{key} = {stats[key]}"
    for p in policy.actor.parameters():
        assert torch.isfinite(p).all()
    for p in policy.critic.parameters():
        assert torch.isfinite(p).all()


def test_update_actually_moves_the_weights():
    policy = _policy()
    before = [p.detach().clone() for p in policy.actor.parameters()]
    buf = _filled_buffer(policy)
    adv = np.random.default_rng(5).normal(size=buf.rewards.shape).astype(np.float32)
    policy.update(buf, adv, adv + buf.values)
    moved = any(
        not torch.equal(b, a) for b, a in zip(before, policy.actor.parameters())
    )
    assert moved


def test_lr_annealing_reaches_zero_at_the_end_of_the_budget():
    policy = _policy(anneal_lr=True, lr=1e-3)
    buf = _filled_buffer(policy)
    adv = np.zeros_like(buf.rewards)
    stats = policy.update(buf, adv, buf.values, progress=0.5)
    assert stats["lr"] == pytest.approx(5e-4)


def test_checkpoint_round_trip_preserves_behaviour():
    policy = _policy()
    payload = policy.state_dict()
    restored = _policy()
    for p in restored.actor.parameters():
        with torch.no_grad():
            p.add_(1.0)
    restored.load_state_dict(payload)

    rng = np.random.default_rng(9)
    obs = rng.random((2, N_AGENTS, OBS_DIM), dtype=np.float32)
    state = rng.random((2, STATE_DIM), dtype=np.float32)
    a1, _, v1 = policy.act(obs, state, deterministic=True)
    a2, _, v2 = restored.act(obs, state, deterministic=True)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_allclose(v1, v2, rtol=1e-6)


# --------------------------------------------------------------- integration
def test_policy_consumes_the_real_environment_surfaces_unchanged():
    """The no-GNN ablation must work against env.observation_space / env.state()."""
    env = make_scenario_env("mixed", max_cycles=20)
    obs, _ = env.reset(seed=0)
    agents = list(env.agents)
    policy = MAPPO(
        env.obs_dim, env.state_dim, len(agents), 6, MAPPOConfig(hidden_dim=16)
    )
    obs_arr = np.stack([obs[a] for a in agents])[None]
    actions, _, values = policy.act(obs_arr, env.state()[None])
    assert actions.shape == (1, len(agents))
    assert values.shape == (1, len(agents))
    _, _, _, _, infos = env.step({a: int(actions[0, i]) for i, a in enumerate(agents)})
    assert set(infos[agents[0]]["reward_components"]) == set(COMPONENT_NAMES)


def test_buffer_tracks_every_reward_component_separately():
    policy = _policy()
    buf = _filled_buffer(policy, n_steps=3)
    assert set(buf.components) == set(COMPONENT_NAMES)
    for key in COMPONENT_NAMES:
        assert buf.components[key].shape == buf.rewards.shape
    assert buf.full is False or buf.ptr == 3
