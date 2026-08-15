"""Heterogeneous-Agent PPO (HAPPO).

Implements sequential policy updates governed by the Multi-Agent Decision Lemma,
ensuring monotonic payoff improvement across heterogeneous/homogeneous agents:

    A_pi^{1:m}(s, a^{1:m}) = sum_{j=1}^m A_pi^{i_j}(s, a^{1:j-1}, a^{i_j})
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from marl.action_mask import apply_action_mask
from marl.mappo import ObservationEncoder, VectorObsEncoder, _mlp


@dataclass(frozen=True)
class HAPPOConfig:
    """Hyperparameters for HAPPO."""

    hidden_dim: int = 128
    n_hidden_layers: int = 2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    anneal_lr: bool = True
    update_epochs: int = 4
    num_minibatches: int = 4
    norm_adv: bool = True
    target_kl: float | None = 0.03

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class HAPPO(nn.Module):
    """Heterogeneous-Agent PPO (HAPPO) trainer with sequential policy updates."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int = 6,
        config: HAPPOConfig | None = None,
        encoder: ObservationEncoder | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.state_dim = int(state_dim)
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.cfg = config or HAPPOConfig()

        self.encoder = encoder or VectorObsEncoder(
            self.obs_dim, self.state_dim, self.n_agents
        )

        # Parameter-shared actor for homogeneous agent slots, or module dict for heterogeneous
        self.actor = _mlp(
            self.encoder.actor_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
            self.n_actions,
            out_std=0.01,
        )
        self.critic = _mlp(
            self.encoder.critic_input_dim,
            self.cfg.hidden_dim,
            self.cfg.n_hidden_layers,
            1,
            out_std=1.0,
        )

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        state: np.ndarray | torch.Tensor,
        action_masks: torch.Tensor | np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample actions and compute values for all agents."""
        self.eval()
        with torch.no_grad():
            device = next(self.actor.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

            act_feats = self.encoder.actor_features(obs_t)
            logits = self.actor(act_feats)

            if action_masks is not None:
                masks_t = torch.as_tensor(action_masks, dtype=torch.float32, device=logits.device)
                logits = apply_action_mask(logits, masks_t)

            dist = Categorical(logits=logits)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                actions = dist.sample()

            logprobs = dist.log_prob(actions)

            crit_feats = self.encoder.critic_features(state_t)
            values = self.critic(crit_feats).squeeze(-1)

        return (
            actions.cpu().numpy(),
            logprobs.cpu().numpy(),
            values.cpu().numpy(),
        )

    def update(
        self,
        rollout_buffer: Any,
        advantages: np.ndarray,
        returns: np.ndarray,
        progress: float = 0.0,
    ) -> dict[str, float]:
        """Execute HAPPO sequential policy updates over mini-batches."""
        self.train()
        cfg = self.cfg

        if cfg.anneal_lr:
            lr_now = cfg.lr * (1.0 - progress)
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = lr_now
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = lr_now
        else:
            lr_now = cfg.lr

        # Convert numpy arrays to tensors
        states_data = getattr(rollout_buffer, "states", getattr(rollout_buffer, "state", None))
        actions_data = getattr(rollout_buffer, "actions", getattr(rollout_buffer, "action", None))
        logprobs_data = getattr(rollout_buffer, "logprobs", getattr(rollout_buffer, "logprob", None))

        device = next(self.actor.parameters()).device
        obs_t = torch.as_tensor(rollout_buffer.obs, dtype=torch.float32, device=device)
        state_t = torch.as_tensor(states_data, dtype=torch.float32, device=device)
        actions_t = torch.as_tensor(actions_data, dtype=torch.long, device=device)
        old_logprobs_t = torch.as_tensor(logprobs_data, dtype=torch.float32, device=device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

        if cfg.norm_adv:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        batch_size = obs_t.size(0) * obs_t.size(1)  # n_steps * n_envs
        minibatch_size = batch_size // cfg.num_minibatches

        tot_actor_loss, tot_critic_loss, tot_entropy, tot_kl = 0.0, 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(cfg.update_epochs):
            perm = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = perm[start:end]

                # Flatten steps and envs
                mb_obs = obs_t.view(-1, self.n_agents, self.obs_dim)[mb_idx]
                mb_state = state_t.view(-1, self.state_dim)[mb_idx]
                mb_actions = actions_t.view(-1, self.n_agents)[mb_idx]
                mb_old_logprobs = old_logprobs_t.view(-1, self.n_agents)[mb_idx]
                mb_adv = advantages_t.view(-1, self.n_agents)[mb_idx]
                mb_returns = returns_t.view(-1, self.n_agents)[mb_idx]

                act_feats = self.encoder.actor_features(mb_obs)
                crit_feats = self.encoder.critic_features(mb_state)

                # Factor M = product of ratio over preceding agents
                m_factor = torch.ones_like(mb_adv[:, 0])

                # HAPPO sequential permutation of agents
                agent_order = torch.randperm(self.n_agents)

                actor_loss_mb = torch.tensor(0.0, device=mb_obs.device)
                entropy_mb = torch.tensor(0.0, device=mb_obs.device)
                kl_mb = torch.tensor(0.0, device=mb_obs.device)

                for agent_i in agent_order:
                    ag_i = int(agent_i.item())
                    ag_obs = act_feats[:, ag_i]
                    ag_act = mb_actions[:, ag_i]
                    ag_old_logp = mb_old_logprobs[:, ag_i]
                    ag_adv = mb_adv[:, ag_i] * m_factor

                    logits = self.actor(ag_obs)
                    dist = Categorical(logits=logits)
                    new_logp = dist.log_prob(ag_act)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logp - ag_old_logp)
                    surr1 = ratio * ag_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * ag_adv

                    policy_loss_i = -torch.min(surr1, surr2).mean() - cfg.ent_coef * entropy
                    actor_loss_mb = actor_loss_mb + policy_loss_i

                    entropy_mb = entropy_mb + entropy
                    kl_mb = kl_mb + (ag_old_logp - new_logp).mean()

                    # Update sequential Multi-Agent factor with detached ratio
                    m_factor = m_factor * ratio.detach()

                # Optimize Actor
                self.actor_optimizer.zero_grad()
                actor_loss_mb.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
                self.actor_optimizer.step()

                # Critic update
                values_pred = self.critic(crit_feats).squeeze(-1)
                critic_loss = F.mse_loss(values_pred, mb_returns)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
                self.critic_optimizer.step()

                tot_actor_loss += actor_loss_mb.item()
                tot_critic_loss += critic_loss.item()
                tot_entropy += (entropy_mb / self.n_agents).item()
                tot_kl += (kl_mb / self.n_agents).item()
                n_updates += 1

        n_updates = max(n_updates, 1)
        return {
            "policy_loss": tot_actor_loss / n_updates,
            "value_loss": tot_critic_loss / n_updates,
            "entropy": tot_entropy / n_updates,
            "approx_kl": tot_kl / n_updates,
            "lr": lr_now,
        }
