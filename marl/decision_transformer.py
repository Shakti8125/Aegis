"""Decision Transformer for Offline RL Trajectory Sequence Modeling.

Models microservice incident recovery trajectories as autoregressive sequence tokens:
    (R_1, s_1, a_1, R_2, s_2, a_2, ..., R_T, s_T, a_T)
where R_t is the Return-to-Go (RTG).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DecisionTransformerConfig:
    """Hyperparameters for Decision Transformer."""

    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 3
    max_ep_len: int = 1000
    dropout: float = 0.1
    lr: float = 1e-4

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self Attention for autoregressive trajectory modeling."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(b_size, seq_len, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Standard Transformer Decoder Block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer for Microservice Recovery Trajectories."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 6,
        config: DecisionTransformerConfig | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = config or DecisionTransformerConfig()

        hidden_dim = self.cfg.hidden_dim

        # Input Encoders
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Embedding(n_actions + 1, hidden_dim)  # +1 for padding / placeholder
        self.embed_timestep = nn.Embedding(self.cfg.max_ep_len, hidden_dim)

        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, self.cfg.num_heads, self.cfg.dropout)
                for _ in range(self.cfg.num_layers)
            ]
        )

        # Prediction Head: predict action from state token embedding
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive trajectory forward pass.

        Args:
            states: (batch_size, seq_len, state_dim)
            actions: (batch_size, seq_len) LongTensor of action IDs
            returns_to_go: (batch_size, seq_len, 1) FloatTensor
            timesteps: (batch_size, seq_len) LongTensor of step indices

        Returns:
            action_preds: (batch_size, seq_len, n_actions) logits
        """
        b_size, seq_len, _ = states.shape

        time_embeddings = self.embed_timestep(timesteps)

        r_emb = self.embed_return(returns_to_go) + time_embeddings
        s_emb = self.embed_state(states) + time_embeddings
        a_emb = self.embed_action(actions) + time_embeddings

        # Interleave tokens: [R_1, s_1, a_1, R_2, s_2, a_2, ...]
        stacked_tokens = (
            torch.stack((r_emb, s_emb, a_emb), dim=2)
            .permute(0, 1, 2, 3)
            .reshape(b_size, 3 * seq_len, self.cfg.hidden_dim)
        )

        stacked_tokens = self.embed_ln(stacked_tokens)

        # Pass through causal transformer blocks
        x = stacked_tokens
        for block in self.blocks:
            x = block(x)

        # Extract outputs corresponding to state tokens (index 1, 4, 7, ...)
        # s_t tokens are at positions 3*t + 1
        state_reprs = x[:, 1::3]  # (b_size, seq_len, hidden_dim)

        # Predict actions logits
        action_preds = self.predict_action(state_reprs)
        return action_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Sample action at the current timestep for evaluation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(states, actions, returns_to_go, timesteps)
            last_logits = logits[:, -1, :]  # Predict next action for the latest step
            action = torch.argmax(last_logits, dim=-1)
        return action
