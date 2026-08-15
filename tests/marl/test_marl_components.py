"""Unit tests for Aegis MARL and Graph-AI components:
- HGT Encoder
- HAPPO
- In-Policy Action Masking
- QMIX
- COMA
- PPO-Lagrangian
- Decision Transformer
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from encoder.features import snapshot_to_hetero_data
from encoder.hgt_encoder import HGTGraphEncoder
from marl.action_mask import (
    ActionMasker,
    MaskedCategorical,
    apply_action_mask,
    compute_action_mask_from_obs,
)
from marl.coma import COMACritic, compute_coma_advantage
from marl.decision_transformer import DecisionTransformer, DecisionTransformerConfig
from marl.happo import HAPPO, HAPPOConfig
from marl.mappo import RolloutBuffer
from marl.ppo_lagrangian import PPOLagrangian, PPOLagrangianConfig
from marl.qmix import QMIX, QMixer
from marl.reward import COMPONENT_NAMES


# 1. Test In-Policy Action Masking
def test_action_masking_logits_and_distribution():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])

    masked_logits = apply_action_mask(logits, mask)
    assert masked_logits[0, 3] <= -1e8
    assert masked_logits[0, 4] <= -1e8
    assert torch.isclose(masked_logits[0, 0], torch.tensor(1.0))

    dist = MaskedCategorical(logits=logits, mask=mask)
    probs = dist.probs[0]
    assert probs[3] == pytest.approx(0.0, abs=1e-6)
    assert probs[4] == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(dist.entropy()).all()

    # Precondition masking test
    obs = torch.zeros(1, 12, 38)
    obs[0, 0, 3] = 1.0  # replicas = 1 -> action 3 (scale down) forbidden
    obs[0, 0, 9] = 1.0  # isolated = 1 -> action 4 (isolate) forbidden, action 5 (reconnect) valid
    computed_mask = compute_action_mask_from_obs(obs)
    assert computed_mask[0, 0, 3] == 0.0
    assert computed_mask[0, 0, 4] == 0.0
    assert computed_mask[0, 0, 5] == 1.0


# 2. Test HGT Encoder
def test_hgt_encoder_forward_and_memory():
    # Build dummy snapshot
    snapshot = {
        "tick": 1,
        "nodes": {
            "Service": [
                {"id": "svc-0", "cpu_pct": 50, "mem_pct": 40, "restart_count": 0, "replicas": 2, "ready_replicas": 2, "health": 1.0},
                {"id": "svc-1", "cpu_pct": 80, "mem_pct": 70, "restart_count": 1, "replicas": 3, "ready_replicas": 2, "health": 0.6},
            ],
            "Pod": [
                {"id": "pod-0", "cpu_pct": 50, "mem_pct": 40, "restart_count": 0, "status": "Running", "health": 1.0},
            ],
            "Node": [
                {"id": "node-0", "cpu_pct": 60, "mem_pct": 50, "restart_count": 0, "pod_count": 1, "pod_capacity": 10, "health": 1.0},
            ],
        },
        "relationships": {
            "DEPENDS_ON": [{"source": "svc-0", "target": "svc-1"}],
            "CALLS": [{"source": "svc-0", "target": "svc-1", "p99_latency_ms": 120.0, "error_rate": 0.01}],
            "INSTANCE_OF": [{"source": "pod-0", "target": "svc-0"}],
            "RUNS_ON": [{"source": "pod-0", "target": "node-0"}],
        },
    }
    data = snapshot_to_hetero_data(snapshot)
    encoder = HGTGraphEncoder()

    out1 = encoder(data)
    assert "Service" in out1.node_embeddings
    assert out1.agent_observations.shape == (2, 64)
    assert out1.global_embedding.shape == (1, 128)

    # Pass memory into second step
    out2 = encoder(data, prev_memory=out1.memory)
    assert out2.agent_observations.shape == (2, 64)
    assert torch.isfinite(out2.global_embedding).all()


# 3. Test HAPPO Trainer
def test_happo_act_and_update():
    policy = HAPPO(obs_dim=38, state_dim=143, n_agents=12, n_actions=6)
    obs = np.random.randn(2, 12, 38).astype(np.float32)
    state = np.random.randn(2, 143).astype(np.float32)

    actions, logprobs, values = policy.act(obs, state)
    assert actions.shape == (2, 12)
    assert logprobs.shape == (2, 12)
    assert values.shape == (2, 12)

    # Dummy buffer and advantages
    buf = RolloutBuffer(n_steps=4, n_envs=2, n_agents=12, obs_dim=38, state_dim=143, component_names=COMPONENT_NAMES)
    for _ in range(4):
        buf.add(
            obs=obs,
            state=state,
            action=actions,
            logprob=logprobs,
            value=values,
            reward=np.zeros((2, 12), dtype=np.float32),
            components={k: np.zeros((2, 12), dtype=np.float32) for k in COMPONENT_NAMES},
            terminated=np.zeros(2, dtype=bool),
            truncated=np.zeros(2, dtype=bool),
            final_state=state,
        )

    adv = np.random.randn(4, 2, 12).astype(np.float32)
    ret = adv + buf.values
    stats = policy.update(buf, adv, ret)
    assert np.isfinite(stats["policy_loss"])
    assert np.isfinite(stats["value_loss"])


# 4. Test QMIX Mixer & System
def test_qmix_monotonicity_and_loss():
    mixer = QMixer(n_agents=4, state_dim=32, embed_dim=16)
    agent_qs = torch.rand(8, 4)
    state = torch.randn(8, 32)

    q_tot = mixer(agent_qs, state)
    assert q_tot.shape == (8, 1)

    # Test Monotonicity property: Increasing agent Q-value must NOT decrease Q_tot
    agent_qs_increased = agent_qs.clone()
    agent_qs_increased[:, 0] += 1.0
    q_tot_increased = mixer(agent_qs_increased, state)
    assert (q_tot_increased >= q_tot - 1e-6).all()

    # Test full QMIX module
    qmix = QMIX(obs_dim=16, state_dim=32, n_agents=4, n_actions=6)
    obs = torch.randn(8, 4, 16)
    actions = torch.randint(0, 6, (8, 4))
    rewards = torch.randn(8)
    next_obs = torch.randn(8, 4, 16)
    next_state = torch.randn(8, 32)
    dones = torch.zeros(8)

    loss = qmix.compute_loss(obs, state, actions, rewards, next_obs, next_state, dones)
    assert torch.isfinite(loss)


# 5. Test COMA Advantage
def test_coma_critic_and_advantage():
    critic = COMACritic(state_dim=32, n_agents=4, n_actions=6)
    state = torch.randn(8, 32)
    joint_actions = torch.randint(0, 6, (8, 4))

    q_values = critic(state, joint_actions)
    assert q_values.shape == (8, 4, 6)

    agent_probs = torch.softmax(torch.randn(8, 4, 6), dim=-1)
    advantage = compute_coma_advantage(q_values, joint_actions, agent_probs)
    assert advantage.shape == (8, 4)
    assert torch.isfinite(advantage).all()


# 6. Test Safe RL PPO-Lagrangian
def test_ppo_lagrangian_primal_dual():
    safe_ppo = PPOLagrangian(obs_dim=38, state_dim=143, n_agents=12, n_actions=6)
    buf = RolloutBuffer(n_steps=4, n_envs=2, n_agents=12, obs_dim=38, state_dim=143, component_names=COMPONENT_NAMES)
    obs = np.random.randn(2, 12, 38).astype(np.float32)
    state = np.random.randn(2, 143).astype(np.float32)
    actions = np.random.randint(0, 6, (2, 12))
    logprobs = np.random.randn(2, 12).astype(np.float32)
    values = np.random.randn(2, 12).astype(np.float32)

    for _ in range(4):
        buf.add(
            obs=obs,
            state=state,
            action=actions,
            logprob=logprobs,
            value=values,
            reward=np.zeros((2, 12), dtype=np.float32),
            components={k: np.zeros((2, 12), dtype=np.float32) for k in COMPONENT_NAMES},
            terminated=np.zeros(2, dtype=bool),
            truncated=np.zeros(2, dtype=bool),
            final_state=state,
        )

    adv = np.random.randn(4, 2, 12).astype(np.float32)
    ret = adv + buf.values
    stats = safe_ppo.update(
        rollout_buffer=buf,
        reward_adv=adv,
        reward_returns=ret,
        sla_adv=adv,
        sla_returns=ret,
        action_cost_adv=adv,
        action_cost_returns=ret,
    )
    assert "policy_loss" in stats
    assert "lambda_sla" in stats
    assert stats["lambda_sla"] >= 0.0


# 7. Test Decision Transformer
def test_decision_transformer_forward_and_action():
    dt = DecisionTransformer(state_dim=32, n_actions=6)
    b_size, seq_len = 4, 10

    states = torch.randn(b_size, seq_len, 32)
    actions = torch.randint(0, 6, (b_size, seq_len))
    returns_to_go = torch.randn(b_size, seq_len, 1)
    timesteps = torch.tile(torch.arange(seq_len), (b_size, 1))

    preds = dt(states, actions, returns_to_go, timesteps)
    assert preds.shape == (b_size, seq_len, 6)

    sampled_act = dt.get_action(states, actions, returns_to_go, timesteps)
    assert sampled_act.shape == (b_size,)
