"""In-Policy Differentiable Action Masking.

Applies large negative logit offsets (-1e9) to forbidden actions prior to
softmax sampling, ensuring invalid actions receive zero probability and
preventing policy gradient distortion caused by post-hoc veto overrides.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.distributions import Categorical


def apply_action_mask(
    logits: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = -1e9,
) -> torch.Tensor:
    """Apply negative logit offsets to invalid actions.

    Args:
        logits: Raw policy logits tensor of shape (..., num_actions).
        mask: Binary tensor of shape (..., num_actions) where 1 indicates valid
              and 0 indicates forbidden.
        mask_value: Negative float added to forbidden logits (default -1e9).

    Returns:
        Masked logits tensor of the same shape as ``logits``.
    """
    mask = mask.to(device=logits.device, dtype=logits.dtype)
    # Adding (1 - mask) * mask_value sets forbidden positions to logits - 1e9
    return logits + (1.0 - mask) * abs(mask_value) * -1.0


class MaskedCategorical(Categorical):
    """Categorical distribution with built-in action masking and safe entropy."""

    def __init__(
        self,
        logits: torch.Tensor | None = None,
        probs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        mask_value: float = -1e9,
        validate_args: Any = None,
    ) -> None:
        if logits is not None and mask is not None:
            logits = apply_action_mask(logits, mask, mask_value)
        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        self.mask = mask

    def entropy(self) -> torch.Tensor:
        """Compute entropy ignoring zero-probability (masked) actions safely."""
        log_p = torch.log(self.probs.clamp_min(1e-12))
        p_log_p = torch.where(self.probs > 0, self.probs * log_p, torch.zeros_like(self.probs))
        return -p_log_p.sum(dim=-1)


class ActionMasker(nn.Module):
    """Layer wrapper for applying action masks to policy logit generators."""

    def __init__(self, mask_value: float = -1e9) -> None:
        super().__init__()
        self.mask_value = mask_value

    def forward(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return apply_action_mask(logits, mask, self.mask_value)

    def dist(self, logits: torch.Tensor, mask: torch.Tensor | None = None) -> MaskedCategorical:
        return MaskedCategorical(logits=logits, mask=mask, mask_value=self.mask_value)


def compute_action_mask_from_obs(
    obs: torch.Tensor | Sequence[Sequence[float]],
    num_actions: int = 6,
) -> torch.Tensor:
    """Compute binary action mask based on service observation feature state.

    Assumes Aegis service action index mapping:
      0: NO_OP
      1: RESTART
      2: SCALE_UP
      3: SCALE_DOWN
      4: ISOLATE
      5: REROUTE
    """
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    batch_shape = obs.shape[:-1]
    mask = torch.ones((*batch_shape, num_actions), dtype=torch.float32, device=obs.device)

    # In ClusterEnv vector obs:
    # index 6: replicas / max_replicas, index 8: isolate_timer / isolate_duration
    if obs.shape[-1] >= 9:
        replica_frac = obs[..., 6]
        isolate_timer = obs[..., 8]

        # Action 3 (SCALE_DOWN): invalid if at or below min replicas (replica_frac <= 0.1)
        mask[..., 3] = torch.where(replica_frac <= 0.1, 0.0, mask[..., 3])
        # Action 4 (ISOLATE): invalid if already isolating (isolate_timer > 0.0)
        mask[..., 4] = torch.where(isolate_timer > 0.0, 0.0, mask[..., 4])

    return mask
