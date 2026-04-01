"""
training/rollout_buffer.py — Experience buffer for PPO.

Stores a single rollout batch and exposes mini-batch sampling
for the PPO update loop.

Fields stored per sample
─────────────────────────
  prompt_ids          (B, T_p)     tokenised prompt
  prompt_mask         (B, T_p)     attention mask for prompt
  response_ids        (B, T_r)     generated response token ids
  old_log_probs       (B, T_r)     log π_old(a_t | s_t) per response token
  ref_log_probs       (B, T_r)     log π_ref(a_t | s_t) per response token
  rewards             (B,)         scalar reward r(s, a)
  values              (B,)         V_old(s) at the prompt end
  advantages          (B,)         GAE-estimated advantage Â(s, a)
  returns             (B,)         discounted return target for the value fn

Note: for text-generation RLHF with a single-step reward (reward is only
given at the *end* of the response, not per token), the advantage is simply:

    Â_i = r_i + γ · V(s_{i+1}) - V(s_i)

which for the terminal step reduces to:

    Â_i = r_i - V(s_i)

We apply GAE over the *response* token sequence treating each token as a
time step, but since the reward is zero for all non-terminal tokens this
simplifies to a backward sum with λ-weighting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional
import torch
import numpy as np


@dataclass
class RolloutBatch:
    """A single completed rollout batch ready for PPO updates."""

    prompt_ids: torch.Tensor         # (B, T_p)
    prompt_mask: torch.Tensor        # (B, T_p)
    response_ids: torch.Tensor       # (B, T_r)
    old_log_probs: torch.Tensor      # (B, T_r)  — sum over response tokens
    ref_log_probs: torch.Tensor      # (B, T_r)
    rewards: torch.Tensor            # (B,)
    values: torch.Tensor             # (B,)
    advantages: torch.Tensor         # (B,)
    returns: torch.Tensor            # (B,)

    # Optional metadata for logging
    texts: Optional[List[str]] = field(default=None, repr=False)
    reward_components: Optional[Dict[str, torch.Tensor]] = field(
        default=None, repr=False
    )

    def __len__(self) -> int:
        return self.rewards.size(0)

    def to(self, device) -> "RolloutBatch":
        """Move all tensors to device (in-place)."""
        for attr in (
            "prompt_ids", "prompt_mask", "response_ids",
            "old_log_probs", "ref_log_probs",
            "rewards", "values", "advantages", "returns",
        ):
            val = getattr(self, attr)
            if isinstance(val, torch.Tensor):
                setattr(self, attr, val.to(device))
        return self

    def mini_batches(
        self, mini_batch_size: int, shuffle: bool = True
    ) -> Iterator["RolloutBatch"]:
        """Yield mini-batches by slicing the batch dimension."""
        N = len(self)
        indices = torch.randperm(N) if shuffle else torch.arange(N)
        for start in range(0, N, mini_batch_size):
            idx = indices[start : start + mini_batch_size]
            yield RolloutBatch(
                prompt_ids=self.prompt_ids[idx],
                prompt_mask=self.prompt_mask[idx],
                response_ids=self.response_ids[idx],
                old_log_probs=self.old_log_probs[idx],
                ref_log_probs=self.ref_log_probs[idx],
                rewards=self.rewards[idx],
                values=self.values[idx],
                advantages=self.advantages[idx],
                returns=self.returns[idx],
            )


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,     # (B,)  terminal reward per episode
    values: torch.Tensor,      # (B,)  V(s) at the start of each episode
    gamma: float = 1.0,
    lam: float = 0.95,
    next_values: Optional[torch.Tensor] = None,  # (B,) — 0 if terminal
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Single-step GAE for RLHF where reward is only given at the last token.

    Returns:
        advantages : (B,)
        returns    : (B,)
    """
    if next_values is None:
        next_values = torch.zeros_like(values)

    delta = rewards + gamma * next_values - values
    # For single-step episodes GAE reduces to TD residual
    advantages = delta
    returns = advantages + values
    return advantages, returns


def compute_gae_sequence(
    token_rewards: torch.Tensor,   # (B, T)  per-token rewards
    token_values: torch.Tensor,    # (B, T)  per-token value estimates
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-step GAE over a token sequence (used when per-token rewards
    are available, e.g. with KL penalties folded in per token).

    Returns:
        advantages : (B, T)
        returns    : (B, T)
    """
    B, T = token_rewards.shape
    advantages = torch.zeros_like(token_rewards)
    gae = torch.zeros(B, device=token_rewards.device)

    for t in reversed(range(T)):
        next_val = token_values[:, t + 1] if t < T - 1 else torch.zeros(B)
        delta = token_rewards[:, t] + gamma * next_val - token_values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae

    returns = advantages + token_values
    return advantages, returns


def whiten(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Zero-mean, unit-variance normalisation."""
    return (t - t.mean()) / (t.std() + eps)
