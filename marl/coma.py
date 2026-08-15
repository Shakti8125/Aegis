"""Counterfactual Multi-Agent (COMA) Advantage Critic.

Calculates counterfactual baselines per agent to solve the multi-agent credit assignment problem:

    A^i(S, a) = Q(S, a) - sum_{\hat{a}^i} pi^i(\hat{a}^i | o^i) * Q(S, (a^{-i}, \hat{a}^i))
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class COMACritic(nn.Module):
    """Centralized COMA Critic evaluating Q(S, (a^{-i}, a^i)) for all candidate actions a^i."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int,
        n_actions: int = 6,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.n_actions = n_actions

        # Joint action input encoding (one-hot joint actions excluding agent i)
        joint_action_dim = (n_agents - 1) * n_actions

        # Per-agent critic head
        self.input_dim = state_dim + joint_action_dim + n_agents
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

        # Agent ID one-hot buffer
        self.register_buffer("_agent_ids", torch.eye(n_agents), persistent=False)

    def forward(self, state: torch.Tensor, joint_actions: torch.Tensor) -> torch.Tensor:
        """Compute Q-values for all agents and all possible actions.

        Args:
            state: Global state tensor of shape (batch_size, state_dim).
            joint_actions: Actions tensor of shape (batch_size, n_agents).

        Returns:
            Q-values tensor of shape (batch_size, n_agents, n_actions).
        """
        b_size = state.size(0)

        # One-hot encode joint actions: (B, N, n_actions)
        actions_onehot = F.one_hot(joint_actions, num_classes=self.n_actions).float()

        q_values_list = []
        for i in range(self.n_agents):
            # Other agents' joint actions (a^{-i})
            other_indices = [j for j in range(self.n_agents) if j != i]
            other_actions = actions_onehot[:, other_indices].reshape(b_size, -1)

            agent_id = self._agent_ids[i].unsqueeze(0).expand(b_size, -1)

            # Construct input vector for agent i
            inp = torch.cat([state, other_actions, agent_id], dim=-1)
            q_i = self.net(inp)  # (B, n_actions)
            q_values_list.append(q_i)

        q_values = torch.stack(q_values_list, dim=1)  # (B, N, n_actions)
        return q_values


def compute_coma_advantage(
    q_values: torch.Tensor,
    joint_actions: torch.Tensor,
    agent_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute COMA counterfactual advantage per agent.

    Args:
        q_values: Tensor of shape (batch_size, n_agents, n_actions) from COMACritic.
        joint_actions: LongTensor of shape (batch_size, n_agents) of executed actions.
        agent_probs: Tensor of shape (batch_size, n_agents, n_actions) policy probabilities.

    Returns:
        Counterfactual advantage tensor of shape (batch_size, n_agents).
    """
    # Q(S, a) for the actual executed joint actions
    q_taken = q_values.gather(-1, joint_actions.unsqueeze(-1)).squeeze(-1)  # (B, N)

    # Counterfactual baseline: b^i(S, a^{-i}) = sum_{\hat{a}^i} pi^i(\hat{a}^i | o^i) * Q(S, (a^{-i}, \hat{a}^i))
    baseline = (agent_probs * q_values).sum(dim=-1)  # (B, N)

    # Counterfactual advantage: A^i(S, a) = Q(S, a) - b^i(S, a^{-i})
    advantage = q_taken - baseline
    return advantage
