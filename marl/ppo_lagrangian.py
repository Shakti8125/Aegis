"""Safe RL PPO-Lagrangian Primal-Dual Optimization.

Adapts dual Lagrange multipliers for SLA violations and resource action costs
separately, enforcing safety constraints without collapsing reward terms into a scalar.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from marl.mappo import MAPPOConfig, ObservationEncoder, VectorObsEncoder, _mlp
from marl.reward import COMPONENT_NAMES


@dataclass(frozen=True)
class PPOLagrangianConfig:
    """Hyperparameters for PPO-Lagrangian."""

    hidden_dim: int = 128
    n_hidden_layers: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    lr_lagrange: float = 1e-2

    # Safety thresholds (limits) for costs
    sla_cost_limit: float = 0.10
    action_cost_limit: float = 0.15

    # Initial Lagrange multipliers
    init_lambda_sla: float = 0.1
    init_lambda_cost: float = 0.1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class PPOLagrangian(nn.Module):
    """PPO-Lagrangian trainer with dual multiplier adaptation for SLA and action costs."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int = 6,
        config: PPOLagrangianConfig | None = None,
        encoder: ObservationEncoder | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.cfg = config or PPOLagrangianConfig()

        self.encoder = encoder or VectorObsEncoder(
            self.obs_dim, self.state_dim, self.n_agents
        )

        self.actor = _mlp(
            self.encoder.actor_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
            self.n_actions,
            out_std=0.01,
        )

        # Primary value critic for reward return
        self.reward_critic = _mlp(
            self.encoder.critic_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
            1,
            out_std=1.0,
        )

        # Cost critics for SLA violation and action cost
        self.sla_cost_critic = _mlp(
            self.encoder.critic_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
            1,
            out_std=1.0,
        )
        self.action_cost_critic = _mlp(
            self.encoder.critic_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
            1,
            out_std=1.0,
        )

        # Dual Lagrange multipliers (log space for positivity)
        self.log_lambda_sla = nn.Parameter(
            torch.tensor(np.log(self.cfg.init_lambda_sla), dtype=torch.float32)
        )
        self.log_lambda_cost = nn.Parameter(
            torch.tensor(np.log(self.cfg.init_lambda_cost), dtype=torch.float32)
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_optimizer = torch.optim.Adam(
            list(self.reward_critic.parameters())
            + list(self.sla_cost_critic.parameters())
            + list(self.action_cost_critic.parameters()),
            lr=self.cfg.lr,
        )
        self.lagrange_optimizer = torch.optim.Adam(
            [self.log_lambda_sla, self.log_lambda_cost], lr=self.cfg.lr_lagrange
        )

    @property
    def lambda_sla(self) -> float:
        return float(torch.exp(self.log_lambda_sla).item())

    @property
    def lambda_cost(self) -> float:
        return float(torch.exp(self.log_lambda_cost).item())

    def update_lagrange_multipliers(
        self,
        mean_sla_cost: float,
        mean_action_cost: float,
    ) -> tuple[float, float]:
        """Dual step updating Lagrange multipliers based on constraint violations."""
        loss_sla = -self.log_lambda_sla * (mean_sla_cost - self.cfg.sla_cost_limit)
        loss_cost = -self.log_lambda_cost * (mean_action_cost - self.cfg.action_cost_limit)
        total_lagrange_loss = loss_sla + loss_cost

        self.lagrange_optimizer.zero_grad()
        total_lagrange_loss.backward()
        self.lagrange_optimizer.step()

        # Clamp log multipliers to prevent numerical blowup
        with torch.no_grad():
            self.log_lambda_sla.clamp_(min=-10.0, max=5.0)
            self.log_lambda_cost.clamp_(min=-10.0, max=5.0)

        return self.lambda_sla, self.lambda_cost

    def update(
        self,
        rollout_buffer: Any,
        reward_adv: np.ndarray,
        reward_returns: np.ndarray,
        sla_adv: np.ndarray,
        sla_returns: np.ndarray,
        action_cost_adv: np.ndarray,
        action_cost_returns: np.ndarray,
    ) -> dict[str, float]:
        """Primal-dual update step for PPO-Lagrangian."""
        self.train()
        cfg = self.cfg

        # Mean cost evaluation for dual updates
        if hasattr(rollout_buffer, "components") and "sla_violation" in rollout_buffer.components:
            sla_mean = float(np.abs(np.mean(rollout_buffer.components["sla_violation"])))
            action_cost_mean = float(np.abs(np.mean(rollout_buffer.components["action_cost"])))
        else:
            sla_mean = float(np.mean(sla_returns))
            action_cost_mean = float(np.mean(action_cost_returns))

        l_sla, l_cost = self.update_lagrange_multipliers(sla_mean, action_cost_mean)

        # Combine advantages: A_total = A_reward - lambda_sla * A_sla - lambda_cost * A_action_cost
        tot_adv = reward_adv - l_sla * sla_adv - l_cost * action_cost_adv
        tot_adv = (tot_adv - tot_adv.mean()) / (tot_adv.std() + 1e-8)

        states_data = getattr(rollout_buffer, "states", getattr(rollout_buffer, "state", None))
        actions_data = getattr(rollout_buffer, "actions", getattr(rollout_buffer, "action", None))
        logprobs_data = getattr(rollout_buffer, "logprobs", getattr(rollout_buffer, "logprob", None))

        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(rollout_buffer.obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(states_data, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions_data, dtype=torch.long, device=device)
        old_logprobs_t = torch.as_tensor(logprobs_data, dtype=torch.float32, device=device)
        tot_adv_t = torch.as_tensor(tot_adv, dtype=torch.float32, device=device)

        act_feats = self.encoder.actor_features(obs_t)
        crit_feats = self.encoder.critic_features(state_t)

        logits = self.actor(act_feats)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(actions_t)
        entropy = dist.entropy().mean()

        ratio = torch.exp(new_logprobs - old_logprobs_t)
        surr1 = ratio * tot_adv_t
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * tot_adv_t
        policy_loss = -torch.min(surr1, surr2).mean() - cfg.ent_coef * entropy

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
        self.actor_optimizer.step()

        # Update Critics
        rew_val = self.reward_critic(crit_feats).squeeze(-1)
        sla_val = self.sla_cost_critic(crit_feats).squeeze(-1)
        act_val = self.action_cost_critic(crit_feats).squeeze(-1)

        loss_rew_val = nn.MSELoss()(rew_val, torch.as_tensor(reward_returns, dtype=torch.float32, device=device))
        loss_sla_val = nn.MSELoss()(sla_val, torch.as_tensor(sla_returns, dtype=torch.float32, device=device))
        loss_act_val = nn.MSELoss()(act_val, torch.as_tensor(action_cost_returns, dtype=torch.float32, device=device))

        critic_loss = loss_rew_val + loss_sla_val + loss_act_val

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.reward_critic.parameters())
            + list(self.sla_cost_critic.parameters())
            + list(self.action_cost_critic.parameters()),
            cfg.max_grad_norm,
        )
        self.critic_optimizer.step()

        return {
            "policy_loss": float(policy_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "entropy": float(entropy.item()),
            "lambda_sla": l_sla,
            "lambda_cost": l_cost,
            "mean_sla_cost": sla_mean,
            "mean_action_cost": action_cost_mean,
        }
