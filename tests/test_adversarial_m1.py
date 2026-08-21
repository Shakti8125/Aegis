"""Empirical Adversarial Stress Test Suite for Milestone 1 Core Libraries."""

from __future__ import annotations

import math
from typing import Any
import numpy as np
import pytest
import torch

from marl.action_mask import (
    ActionMasker,
    MaskedCategorical,
    apply_action_mask,
    compute_action_mask_from_obs,
)
from marl.mappo import RolloutBuffer
from marl.ppo_lagrangian import PPOLagrangian, PPOLagrangianConfig
from marl.qmix import QMIX, QMIXConfig, QMixer
from marl.reward import COMPONENT_NAMES


# ==============================================================================
# 1. ACTION MASKING EMPIRICAL ADVERSARIAL CHALLENGES
# ==============================================================================

class TestActionMaskingAdversarial:
    """Stress tests and mathematical invariant checks for action masking."""

    @pytest.mark.parametrize("seed", range(10))
    def test_entropy_non_negativity_random_logits(self, seed: int):
        """Verify H(P) >= 0 across diverse random logits and valid action masks."""
        torch.manual_seed(seed)
        batch_size = 32
        num_actions = 6
        logits = torch.randn(batch_size, num_actions) * 10.0  # High variance logits

        # Generate non-zero random masks (at least 1 action valid)
        mask = torch.randint(0, 2, (batch_size, num_actions)).float()
        for i in range(batch_size):
            if mask[i].sum() == 0:
                mask[i, 0] = 1.0

        dist = MaskedCategorical(logits=logits, mask=mask)
        entropy = dist.entropy()

        assert torch.isfinite(entropy).all(), "Entropy must be finite"
        assert (entropy >= -1e-6).all(), f"Entropy must be non-negative, got {entropy.min()}"

    def test_entropy_extreme_logits_and_bounds(self):
        """Verify entropy behavior under extreme logit magnitudes (-1000 to +1000)."""
        logits = torch.tensor([
            [1000.0, -1000.0, 0.0, -500.0, 500.0, -100.0],
            [1e4, -1e4, 0.0, 0.0, 0.0, 0.0],
            [-1e5, -1e5, -1e5, -1e5, -1e5, 10.0],
        ])
        mask = torch.tensor([
            [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])
        dist = MaskedCategorical(logits=logits, mask=mask)
        entropy = dist.entropy()

        assert torch.isfinite(entropy).all()
        assert (entropy >= -1e-6).all()
        # Single valid action in row 2 must have entropy exactly ~0
        assert entropy[2].item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("c", [-1000.0, -42.0, 0.0, 42.0, 1000.0])
    def test_logit_shift_invariance(self, c: float):
        """Mathematical Invariant: H(logits + C) == H(logits) and probs(logits + C) == probs(logits)."""
        logits = torch.tensor([[2.5, -1.0, 4.0, 0.5, -3.0, 1.2]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])

        dist_base = MaskedCategorical(logits=logits, mask=mask)
        dist_shifted = MaskedCategorical(logits=logits + c, mask=mask)

        # Probabilities must match identically within float32 tolerance
        assert torch.allclose(dist_base.probs, dist_shifted.probs, atol=1e-5)
        # Entropy must match identically
        assert torch.isclose(dist_base.entropy(), dist_shifted.entropy(), atol=1e-5)

    @pytest.mark.parametrize("c", [-1e4, 1e4])
    def test_logit_shift_invariance_large_c(self, c: float):
        """Verify shift invariance holds under float32 within IEEE 754 precision limits (5e-5)."""
        logits = torch.tensor([[2.5, -1.0, 4.0, 0.5, -3.0, 1.2]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])

        dist_base = MaskedCategorical(logits=logits, mask=mask)
        dist_shifted = MaskedCategorical(logits=logits + c, mask=mask)

        assert torch.allclose(dist_base.probs, dist_shifted.probs, atol=5e-5)
        assert torch.isclose(dist_base.entropy(), dist_shifted.entropy(), atol=5e-5)

    def test_logit_shift_invariance_extreme_float64(self):
        """Verify shift invariance holds under float64 even for massive offsets (+-1e6)."""
        logits = torch.tensor([[2.5, -1.0, 4.0, 0.5, -3.0, 1.2]], dtype=torch.float64)
        mask = torch.tensor([[1.0, 1.0, 1.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
        c = 1e6

        dist_base = MaskedCategorical(logits=logits, mask=mask)
        dist_shifted = MaskedCategorical(logits=logits + c, mask=mask)

        assert torch.allclose(dist_base.probs, dist_shifted.probs, atol=1e-6)
        assert torch.isclose(dist_base.entropy(), dist_shifted.entropy(), atol=1e-6)

    @pytest.mark.parametrize("m_valid", [1, 2, 3, 4, 5, 6])
    def test_equal_logits_entropy_oracle(self, m_valid: int):
        """Oracle test: For M valid equal logits, H = ln(M) exactly."""
        logits = torch.zeros(1, 6)
        mask = torch.zeros(1, 6)
        mask[0, :m_valid] = 1.0

        dist = MaskedCategorical(logits=logits, mask=mask)
        expected_h = math.log(m_valid)

        assert dist.entropy().item() == pytest.approx(expected_h, abs=1e-5)

    def test_masked_probabilities_sum_to_one_and_forbidden_are_zero(self):
        """Verify sum(p) == 1.0 and p_forbidden == 0.0 for masked distributions."""
        logits = torch.randn(50, 6) * 5.0
        mask = torch.zeros(50, 6)
        for i in range(50):
            k = (i % 5) + 1  # 1 to 5 valid actions
            indices = np.random.choice(6, k, replace=False)
            mask[i, indices] = 1.0

        dist = MaskedCategorical(logits=logits, mask=mask)
        probs = dist.probs

        # 1. Sum of probabilities per row must be 1.0
        assert torch.allclose(probs.sum(dim=-1), torch.ones(50), atol=1e-5)

        # 2. Forbidden positions must have probability 0.0
        forbidden_probs = probs[mask == 0.0]
        assert (forbidden_probs < 1e-12).all()

        # 3. Empirical sampling: Sample 5000 times, no forbidden action should EVER be drawn
        samples = dist.sample((100,))  # Shape (100, 50)
        for row in range(50):
            valid_indices = torch.where(mask[row] == 1.0)[0]
            sampled_acts = samples[:, row]
            for act in sampled_acts:
                assert act.item() in valid_indices.tolist()

    def test_action_masker_module(self):
        """Test ActionMasker nn.Module and its forward / dist methods across batch shapes."""
        layer = ActionMasker(mask_value=-1e9)
        # 3D tensor (B, N, num_actions)
        logits = torch.randn(4, 3, 6)
        mask = torch.ones(4, 3, 6)
        mask[:, :, 3:] = 0.0  # mask out actions 3, 4, 5

        masked_logits = layer(logits, mask)
        assert (masked_logits[:, :, 3:] <= -1e8).all()
        assert torch.allclose(masked_logits[:, :, :3], logits[:, :, :3])

        dist = layer.dist(logits, mask)
        assert dist.probs[:, :, 3:].sum().item() == pytest.approx(0.0, abs=1e-7)

    @pytest.mark.parametrize("replica_frac,should_mask_scale_down", [
        (0.0, True),
        (0.05, True),
        (0.10, True),
        (0.1001, False),
        (0.20, False),
        (1.0, False),
        (1.5, False),
    ])
    def test_compute_action_mask_obs_scale_down_boundary(self, replica_frac: float, should_mask_scale_down: bool):
        """Boundary test for replica_frac (index 6): SCALE_DOWN (action 3) masked if <= 0.1."""
        obs = torch.zeros(1, 10)
        obs[0, 6] = replica_frac
        obs[0, 8] = 0.0  # isolate_timer = 0

        mask = compute_action_mask_from_obs(obs)
        if should_mask_scale_down:
            assert mask[0, 3].item() == 0.0, f"Expected action 3 masked for replica_frac={replica_frac}"
        else:
            assert mask[0, 3].item() == 1.0, f"Expected action 3 valid for replica_frac={replica_frac}"

    @pytest.mark.parametrize("isolate_timer,should_mask_isolate", [
        (0.0, False),
        (-0.5, False),
        (0.0001, True),
        (0.5, True),
        (1.0, True),
        (10.0, True),
    ])
    def test_compute_action_mask_obs_isolate_boundary(self, isolate_timer: float, should_mask_isolate: bool):
        """Boundary test for isolate_timer (index 8): ISOLATE (action 4) masked if > 0.0."""
        obs = torch.zeros(1, 10)
        obs[0, 6] = 0.5  # replica_frac > 0.1
        obs[0, 8] = isolate_timer

        mask = compute_action_mask_from_obs(obs)
        if should_mask_isolate:
            assert mask[0, 4].item() == 0.0, f"Expected action 4 masked for isolate_timer={isolate_timer}"
        else:
            assert mask[0, 4].item() == 1.0, f"Expected action 4 valid for isolate_timer={isolate_timer}"

    def test_compute_action_mask_reroute_invariant(self):
        """Verify Action 5 (REROUTE) is always valid (1.0) regardless of isolation state."""
        for isolate_val in [0.0, 0.5, 1.0]:
            obs = torch.zeros(1, 10)
            obs[0, 8] = isolate_val
            mask = compute_action_mask_from_obs(obs)
            assert mask[0, 5].item() == 1.0, "Action 5 (REROUTE) must remain valid"

    def test_compute_action_mask_input_formats_and_short_obs(self):
        """Test compute_action_mask_from_obs with lists, 1D/2D/3D tensors, and short obs."""
        # 1. Non-tensor list
        obs_list = [[0.0] * 10]
        obs_list[0][6] = 0.05
        mask_list = compute_action_mask_from_obs(obs_list)
        assert isinstance(mask_list, torch.Tensor)
        assert mask_list[0, 3] == 0.0

        # 2. Short observation vector (dim < 9) -> should return all ones
        obs_short = torch.zeros(4, 8)
        mask_short = compute_action_mask_from_obs(obs_short)
        assert mask_short.shape == (4, 6)
        assert (mask_short == 1.0).all()

        # 3. 3D batch observation tensor (B, N, obs_dim)
        obs_3d = torch.zeros(2, 4, 12)
        obs_3d[1, 2, 6] = 0.05  # env 1, agent 2: min replicas
        mask_3d = compute_action_mask_from_obs(obs_3d)
        assert mask_3d.shape == (2, 4, 6)
        assert mask_3d[1, 2, 3] == 0.0
        assert mask_3d[0, 0, 3] == 0.0  # 0.0 <= 0.1


# ==============================================================================
# 2. QMIX EMPIRICAL ADVERSARIAL CHALLENGES
# ==============================================================================

class TestQMIXAdversarial:
    """Stress tests and mathematical monotonicity checks for QMIX."""

    @pytest.mark.parametrize("b_size", [1, 4, 32])
    @pytest.mark.parametrize("n_agents", [1, 2, 4, 12])
    def test_qmix_reward_tensor_shapes(self, b_size: int, n_agents: int):
        """Stress-test QMIX compute_loss across 1D (B,), 2D (B, 1), and 2D per-agent (B, N) rewards."""
        obs_dim = 16
        state_dim = 32
        qmix = QMIX(obs_dim=obs_dim, state_dim=state_dim, n_agents=n_agents, n_actions=6)

        obs = torch.randn(b_size, n_agents, obs_dim)
        states = torch.randn(b_size, state_dim)
        actions = torch.randint(0, 6, (b_size, n_agents))
        next_obs = torch.randn(b_size, n_agents, obs_dim)
        next_states = torch.randn(b_size, state_dim)
        dones = torch.zeros(b_size)

        # 1. 1D rewards: (B,)
        rewards_1d = torch.randn(b_size)
        loss_1d = qmix.compute_loss(obs, states, actions, rewards_1d, next_obs, next_states, dones)
        assert torch.isfinite(loss_1d)
        assert loss_1d.ndim == 0

        # 2. 2D scalar rewards: (B, 1)
        rewards_2d_scalar = torch.randn(b_size, 1)
        loss_2d_scalar = qmix.compute_loss(obs, states, actions, rewards_2d_scalar, next_obs, next_states, dones)
        assert torch.isfinite(loss_2d_scalar)

        # 3. 2D per-agent rewards: (B, N)
        rewards_2d_agent = torch.randn(b_size, n_agents)
        loss_2d_agent = qmix.compute_loss(obs, states, actions, rewards_2d_agent, next_obs, next_states, dones)
        assert torch.isfinite(loss_2d_agent)

    @pytest.mark.parametrize("n_agents", [1, 3, 8])
    def test_qmix_monotonicity_via_gradients(self, n_agents: int):
        """Theoretical Invariant: dQ_tot / dq_i >= 0 everywhere for all agents."""
        state_dim = 24
        mixer = QMixer(n_agents=n_agents, state_dim=state_dim, embed_dim=16)

        batch_size = 16
        states = torch.randn(batch_size, state_dim)
        agent_qs = torch.randn(batch_size, n_agents, requires_grad=True)

        q_tot = mixer(agent_qs, states)  # (B, 1)
        assert q_tot.shape == (batch_size, 1)

        # Compute gradient d(sum(Q_tot)) / d(agent_qs)
        q_tot.sum().backward()

        assert agent_qs.grad is not None
        # Monotonicity requires all gradients to be non-negative
        assert (agent_qs.grad >= -1e-6).all(), f"Negative gradient detected: {agent_qs.grad.min()}"

    def test_qmix_act_shapes_and_device(self):
        """Test QMIX act() method with numpy arrays and torch tensors, checking epsilon branches."""
        qmix = QMIX(obs_dim=12, state_dim=24, n_agents=4, n_actions=6)

        # Numpy input
        obs_np = np.random.randn(8, 4, 12).astype(np.float32)
        acts_greedy = qmix.act(obs_np, epsilon=0.0)
        assert acts_greedy.shape == (8, 4)
        assert isinstance(acts_greedy, np.ndarray)

        acts_eps = qmix.act(obs_np, epsilon=1.0)
        assert acts_eps.shape == (8, 4)
        assert (acts_eps >= 0).all() and (acts_eps < 6).all()

        # Single-batch input (1, 4, 12)
        obs_single = np.random.randn(1, 4, 12).astype(np.float32)
        acts_single = qmix.act(obs_single, epsilon=0.1)
        assert acts_single.shape == (1, 4)

    def test_qmix_target_network_update(self):
        """Verify target networks synchronize parameters correctly."""
        qmix = QMIX(obs_dim=8, state_dim=16, n_agents=2, n_actions=4)

        # Perturb main parameters
        with torch.no_grad():
            for p in qmix.agent_net.parameters():
                p.add_(1.0)
            for p in qmix.mixer.parameters():
                p.add_(1.0)

        # Target should initially be different
        p_agent = next(qmix.agent_net.parameters())
        p_target = next(qmix.target_agent_net.parameters())
        assert not torch.allclose(p_agent, p_target)

        # Update targets
        qmix.update_target_nets()
        p_target_after = next(qmix.target_agent_net.parameters())
        assert torch.allclose(p_agent, p_target_after)


# ==============================================================================
# 3. PPO-LAGRANGIAN EMPIRICAL ADVERSARIAL CHALLENGES
# ==============================================================================

class TestPPOLagrangianAdversarial:
    """Stress tests and dual convergence checks for PPO-Lagrangian."""

    def test_lagrangian_multipliers_directional_updates(self):
        """Mathematical Invariant: Multiplier increases on constraint violation, decreases on satisfaction."""
        cfg = PPOLagrangianConfig(
            sla_cost_limit=0.10,
            action_cost_limit=0.15,
            init_lambda_sla=0.1,
            init_lambda_cost=0.1,
            lr_lagrange=0.05,
        )

        # Instance A: Constraint Violation (SLA cost 0.50 > 0.10) -> lambda_sla MUST INCREASE
        ppo_violation = PPOLagrangian(obs_dim=10, state_dim=20, n_agents=2, n_actions=6, config=cfg)
        init_sla_v = ppo_violation.lambda_sla
        l_sla_v, _ = ppo_violation.update_lagrange_multipliers(mean_sla_cost=0.50, mean_action_cost=0.05)
        assert l_sla_v > init_sla_v, f"Expected lambda_sla to increase, got {l_sla_v} <= {init_sla_v}"

        # Instance B: Constraint Satisfaction (SLA cost 0.01 < 0.10) -> lambda_sla MUST DECREASE
        ppo_satisfied = PPOLagrangian(obs_dim=10, state_dim=20, n_agents=2, n_actions=6, config=cfg)
        init_sla_s = ppo_satisfied.lambda_sla
        l_sla_s, _ = ppo_satisfied.update_lagrange_multipliers(mean_sla_cost=0.01, mean_action_cost=0.05)
        assert l_sla_s < init_sla_s, f"Expected lambda_sla to decrease, got {l_sla_s} >= {init_sla_s}"

    def test_lagrangian_multiplier_clamping_and_extreme_stress(self):
        """Stress-test multipliers under 200 consecutive massive violations and zero-cost steps."""
        cfg = PPOLagrangianConfig(
            sla_cost_limit=0.10,
            action_cost_limit=0.15,
            lr_lagrange=0.1,
        )
        ppo = PPOLagrangian(obs_dim=10, state_dim=20, n_agents=2, config=cfg)

        # 200 massive violations
        for _ in range(200):
            l_sla, l_cost = ppo.update_lagrange_multipliers(mean_sla_cost=100.0, mean_action_cost=100.0)
            assert np.isfinite(l_sla) and np.isfinite(l_cost)
            # Bound: exp(5.0) ~ 148.41
            assert l_sla <= math.exp(5.0) + 1e-3
            assert l_cost <= math.exp(5.0) + 1e-3

        # 200 zero violations
        for _ in range(200):
            l_sla, l_cost = ppo.update_lagrange_multipliers(mean_sla_cost=0.0, mean_action_cost=0.0)
            assert np.isfinite(l_sla) and np.isfinite(l_cost)
            # Bound: exp(-10.0) ~ 4.54e-5
            assert l_sla >= math.exp(-10.0) - 1e-6
            assert l_cost >= math.exp(-10.0) - 1e-6

    def test_ppo_lagrangian_full_update_cycle_and_gradient_flow(self):
        """Verify full primal-dual update with 3 critics and actor gradient flow."""
        ppo = PPOLagrangian(obs_dim=8, state_dim=16, n_agents=3, n_actions=6)

        n_steps, n_envs, n_agents = 4, 2, 3
        buf = RolloutBuffer(n_steps=n_steps, n_envs=n_envs, n_agents=n_agents, obs_dim=8, state_dim=16, component_names=COMPONENT_NAMES)

        obs = np.random.randn(n_envs, n_agents, 8).astype(np.float32)
        state = np.random.randn(n_envs, 16).astype(np.float32)
        actions = np.random.randint(0, 6, (n_envs, n_agents))
        logprobs = np.random.randn(n_envs, n_agents).astype(np.float32)
        values = np.random.randn(n_envs, n_agents).astype(np.float32)

        for _ in range(n_steps):
            buf.add(
                obs=obs,
                state=state,
                action=actions,
                logprob=logprobs,
                value=values,
                reward=np.random.randn(n_envs, n_agents).astype(np.float32),
                components={k: np.random.randn(n_envs, n_agents).astype(np.float32) for k in COMPONENT_NAMES},
                terminated=np.zeros(n_envs, dtype=bool),
                truncated=np.zeros(n_envs, dtype=bool),
                final_state=state,
            )

        reward_adv = np.random.randn(n_steps, n_envs, n_agents).astype(np.float32)
        reward_ret = reward_adv + buf.values
        sla_adv = np.random.randn(n_steps, n_envs, n_agents).astype(np.float32)
        sla_ret = sla_adv + 0.5
        cost_adv = np.random.randn(n_steps, n_envs, n_agents).astype(np.float32)
        cost_ret = cost_adv + 0.2

        stats = ppo.update(
            rollout_buffer=buf,
            reward_adv=reward_adv,
            reward_returns=reward_ret,
            sla_adv=sla_adv,
            sla_returns=sla_ret,
            action_cost_adv=cost_adv,
            action_cost_returns=cost_ret,
        )

        assert "policy_loss" in stats and np.isfinite(stats["policy_loss"])
        assert "critic_loss" in stats and np.isfinite(stats["critic_loss"])
        assert "entropy" in stats and np.isfinite(stats["entropy"])
        assert "lambda_sla" in stats and stats["lambda_sla"] > 0.0
        assert "lambda_cost" in stats and stats["lambda_cost"] > 0.0
        assert "mean_sla_cost" in stats
        assert "mean_action_cost" in stats
