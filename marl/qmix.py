"""QMIX Value Decomposition Hypernetwork.

Factorizes joint action values Q_tot(S, a) monotonically from individual agent
action values q_i(o_i, a_i) based on global graph state features:

    dQ_tot / dq_i >= 0
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class QMIXConfig:
    """Hyperparameters for QMIX value decomposition."""

    hidden_dim: int = 64
    mixer_embed_dim: int = 32
    hypernet_embed_dim: int = 64
    lr: float = 5e-4
    gamma: float = 0.99
    target_update_interval: int = 100
    max_grad_norm: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class QMixer(nn.Module):
    """QMIX Monotonic Value Decomposition Mixer with State-Dependent Hypernetworks."""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork for W1: (state_dim) -> (n_agents * embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, n_agents * embed_dim),
        )

        # Hypernetwork for b1: (state_dim) -> (embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for W2: (state_dim) -> (embed_dim * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, embed_dim),
        )

        # Hypernetwork for b2 (V(s)): 2-layer MLP (state_dim) -> (1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed_dim),
            nn.ReLU(),
            nn.Linear(hypernet_embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Factorize joint Q_tot(S, a) monotonically.

        Args:
            agent_qs: Tensor of shape (batch_size, n_agents) or (batch_size, n_agents, 1)
            states: Tensor of shape (batch_size, state_dim)

        Returns:
            Q_tot tensor of shape (batch_size, 1)
        """
        if agent_qs.dim() == 2:
            agent_qs = agent_qs.unsqueeze(-1)  # (B, N, 1)

        b_size = agent_qs.size(0)
        agent_qs = agent_qs.view(b_size, 1, self.n_agents)  # (B, 1, N)

        # Generate non-negative weights W1 via abs()
        w1 = torch.abs(self.hyper_w1(states)).view(b_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(states).view(b_size, 1, self.embed_dim)

        # Layer 1: hidden = ELU(q * W1 + b1)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # (B, 1, embed_dim)

        # Generate non-negative weights W2 via abs()
        w2 = torch.abs(self.hyper_w2(states)).view(b_size, self.embed_dim, 1)
        b2 = self.hyper_b2(states).view(b_size, 1, 1)

        # Layer 2: Q_tot = hidden * W2 + b2
        q_tot = torch.bmm(hidden, w2) + b2  # (B, 1, 1)
        return q_tot.view(b_size, 1)


class QMixAgent(nn.Module):
    """Individual agent Q-network."""

    def __init__(self, obs_dim: int, n_actions: int = 6, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class QMIX(nn.Module):
    """QMIX MARL System with Individual Agent Nets & Centralized Mixer."""

    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        n_agents: int,
        n_actions: int = 6,
        config: QMIXConfig | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.cfg = config or QMIXConfig()

        self.agent_net = QMixAgent(obs_dim, n_actions, self.cfg.hidden_dim)
        self.target_agent_net = QMixAgent(obs_dim, n_actions, self.cfg.hidden_dim)
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())

        self.mixer = QMixer(n_agents, state_dim, self.cfg.mixer_embed_dim, self.cfg.hypernet_embed_dim)
        self.target_mixer = QMixer(n_agents, state_dim, self.cfg.mixer_embed_dim, self.cfg.hypernet_embed_dim)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.agent_net.parameters()) + list(self.mixer.parameters()),
            lr=self.cfg.lr,
        )

    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        epsilon: float = 0.05,
    ) -> np.ndarray:
        """Epsilon-greedy action selection for individual agents."""
        self.eval()
        with torch.no_grad():
            device = next(self.agent_net.parameters()).device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            qs = self.agent_net(obs_t)  # (..., N, n_actions)

            actions = torch.argmax(qs, dim=-1).cpu().numpy()
            if epsilon > 0.0:
                rng = np.random.default_rng()
                random_actions = rng.integers(0, self.n_actions, size=actions.shape)
                mask = rng.random(size=actions.shape) < epsilon
                actions = np.where(mask, random_actions, actions)

        return actions

    def compute_loss(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Compute TD loss for joint Q_tot values."""
        b_size = obs.size(0)

        # Support both 1D and 2D per-agent reward tensors
        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1, keepdim=True)
        else:
            rewards = rewards.view(b_size, 1)

        dones = dones.view(b_size, 1)

        # Current Q-values for chosen actions
        mac_qs = self.agent_net(obs)  # (B, N, n_actions)
        chosen_qs = torch.gather(mac_qs, dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)  # (B, N)
        q_tot = self.mixer(chosen_qs, states)  # (B, 1)

        # Target Q-values (max over next actions)
        with torch.no_grad():
            target_mac_qs = self.target_agent_net(next_obs)
            max_next_qs = target_mac_qs.max(dim=-1)[0]  # (B, N)
            target_q_tot = self.target_mixer(max_next_qs, next_states)  # (B, 1)

            y = rewards + (1.0 - dones) * self.cfg.gamma * target_q_tot

        td_error = q_tot - y
        loss = (td_error ** 2).mean()
        return loss

    def update_target_nets(self) -> None:
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
