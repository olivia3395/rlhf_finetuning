"""
training/ppo_trainer.py — Core Proximal Policy Optimisation update.

Algorithm (InstructGPT / TRL style)
────────────────────────────────────
For each rollout batch:

  1. Compute per-token KL penalty:
         kl_t = log π_old(a_t) - log π_ref(a_t)

  2. Fold KL into per-token reward:
         r̂_t = r_env · 𝟙[t = T] - β · kl_t

  3. Compute advantages via GAE over the token sequence.

  4. For ppo_epochs epochs:
       a. Re-compute log π_θ(a_t | s_t) under the current policy.
       b. Compute probability ratio:
              ρ_t = exp(log π_θ - log π_old)
       c. Clipped surrogate loss:
              L_CLIP = -E[min(ρ_t Â_t, clip(ρ_t, 1-ε, 1+ε) Â_t)]
       d. Value function loss:
              L_VF = 0.5 · MSE(V_θ(s), returns)
       e. Entropy bonus:
              L_ENT = -H[π_θ(·|s)]
       f. Total loss:
              L = L_CLIP + c_vf · L_VF + c_ent · L_ENT
       g. Gradient step with gradient clipping.

  5. Adapt β via PID-style rule based on observed KL vs target_kl.

References
──────────
  [1] Schulman et al. (2017)  "Proximal Policy Optimization Algorithms"
  [2] Ouyang et al. (2022)    "Training language models to follow instructions
                               with human feedback"  (InstructGPT)
  [3] Ziegler et al. (2019)   "Fine-Tuning Language Models from Human Preferences"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from config import PPOConfig, TrainingConfig
from models.policy_model import PolicyModel, ReferenceModel
from training.rollout_buffer import (
    RolloutBatch,
    compute_gae_sequence,
    whiten,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PPO statistics dataclass
# ---------------------------------------------------------------------------

@dataclass
class PPOStats:
    """Aggregated statistics from a single PPO update."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    total_loss: float = 0.0
    approx_kl: float = 0.0           # mean KL between new and old policy
    kl_penalty: float = 0.0          # mean β · KL(π_old ‖ π_ref)
    clip_fraction: float = 0.0       # fraction of clipped ratios
    kl_coef: float = 0.0             # current adaptive β
    mean_reward: float = 0.0
    mean_advantage: float = 0.0
    n_updates: int = 0

    def __add__(self, other: "PPOStats") -> "PPOStats":
        return PPOStats(
            policy_loss=self.policy_loss + other.policy_loss,
            value_loss=self.value_loss + other.value_loss,
            entropy=self.entropy + other.entropy,
            total_loss=self.total_loss + other.total_loss,
            approx_kl=self.approx_kl + other.approx_kl,
            kl_penalty=self.kl_penalty + other.kl_penalty,
            clip_fraction=self.clip_fraction + other.clip_fraction,
            kl_coef=self.kl_coef + other.kl_coef,
            mean_reward=self.mean_reward + other.mean_reward,
            mean_advantage=self.mean_advantage + other.mean_advantage,
            n_updates=self.n_updates + other.n_updates,
        )

    def mean(self) -> "PPOStats":
        n = max(self.n_updates, 1)
        return PPOStats(
            policy_loss=self.policy_loss / n,
            value_loss=self.value_loss / n,
            entropy=self.entropy / n,
            total_loss=self.total_loss / n,
            approx_kl=self.approx_kl / n,
            kl_penalty=self.kl_penalty / n,
            clip_fraction=self.clip_fraction / n,
            kl_coef=self.kl_coef / n,
            mean_reward=self.mean_reward / n,
            mean_advantage=self.mean_advantage / n,
            n_updates=self.n_updates,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "train/policy_loss": self.policy_loss,
            "train/value_loss": self.value_loss,
            "train/entropy": self.entropy,
            "train/total_loss": self.total_loss,
            "train/approx_kl": self.approx_kl,
            "train/kl_penalty": self.kl_penalty,
            "train/clip_fraction": self.clip_fraction,
            "train/kl_coef": self.kl_coef,
            "train/mean_reward": self.mean_reward,
            "train/mean_advantage": self.mean_advantage,
        }


# ---------------------------------------------------------------------------
# Adaptive KL controller
# ---------------------------------------------------------------------------

class AdaptiveKLController:
    """
    Adjusts the KL coefficient β to keep the observed KL near a target.

    Rule (from Ziegler et al.):
        if KL > target * 1.5:  β ← β * (1 + speed)
        if KL < target / 1.5:  β ← β * (1 - speed)
    """

    def __init__(
        self,
        init_kl_coef: float = 0.2,
        target_kl: Optional[float] = 0.1,
        speed: float = 0.02,
    ):
        self.value = init_kl_coef
        self.target = target_kl
        self.speed = speed

    def update(self, observed_kl: float):
        if self.target is None:
            return
        if observed_kl > self.target * 1.5:
            self.value *= 1 + self.speed
        elif observed_kl < self.target / 1.5:
            self.value *= 1 - self.speed
        # Clamp to a reasonable range
        self.value = float(torch.tensor(self.value).clamp(0.01, 2.0))


# ---------------------------------------------------------------------------
# PPO Trainer
# ---------------------------------------------------------------------------

class PPOTrainer:
    """
    Implements the PPO update step for language model fine-tuning.

    Typical usage inside the RLHF training loop:

        trainer = PPOTrainer(policy, ref_model, ppo_cfg, train_cfg, device)
        for step in range(total_steps):
            batch = rollout(policy, prompts, reward_fn)
            stats = trainer.update(batch)
    """

    def __init__(
        self,
        policy: PolicyModel,
        ref_model: ReferenceModel,
        ppo_cfg: PPOConfig,
        train_cfg: TrainingConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.ppo_cfg = ppo_cfg
        self.train_cfg = train_cfg
        self.device = device

        # Optimiser — only update the *policy* parameters
        self.optimizer = AdamW(
            policy.parameters(),
            lr=train_cfg.learning_rate,
            eps=train_cfg.adam_epsilon,
            weight_decay=train_cfg.weight_decay,
        )

        # LR scheduler with linear warmup
        self.scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=train_cfg.warmup_steps,
        )

        # Adaptive KL controller
        self.kl_ctl = AdaptiveKLController(
            init_kl_coef=ppo_cfg.init_kl_coef,
            target_kl=ppo_cfg.target_kl,
            speed=ppo_cfg.kl_adapt_speed,
        )

    # ------------------------------------------------------------------
    # KL-penalised reward computation
    # ------------------------------------------------------------------

    def _compute_token_rewards(
        self, batch: RolloutBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build per-token rewards by folding the KL penalty into the
        scalar environment reward (which is only added at the last token).

        token_rewards[b, t] = -β · KL_t             for t < T-1
        token_rewards[b, T-1] = r[b] - β · KL_{T-1}

        Returns:
            token_rewards : (B, T_r)
            kl_penalty    : (B,)  mean KL per sample (for logging)
        """
        B, T_r = batch.response_ids.shape

        # KL per token: log π_old - log π_ref
        kl_tokens = batch.old_log_probs - batch.ref_log_probs   # (B, T_r)
        kl_tokens = kl_tokens.clamp(min=0.0)                    # truncated KL

        beta = self.kl_ctl.value
        token_rewards = -beta * kl_tokens                       # (B, T_r)

        # Add scalar env reward at the last response token
        token_rewards[:, -1] += batch.rewards                   # broadcast (B,)

        kl_penalty = (beta * kl_tokens).mean(dim=1)             # (B,)
        return token_rewards, kl_penalty

    # ------------------------------------------------------------------
    # Single PPO update step
    # ------------------------------------------------------------------

    def update(self, batch: RolloutBatch) -> PPOStats:
        """
        Run ppo_epochs × mini-batch PPO updates on the provided rollout.

        Returns aggregated PPOStats.
        """
        batch = batch.to(self.device)
        self.policy.train()

        # ── 1. Compute token-level rewards with KL penalty ───────────────
        token_rewards, kl_penalty = self._compute_token_rewards(batch)

        # ── 2. Compute per-token value targets using backbone ────────────
        #   We need V(s_t) for each response token to apply sequence GAE.
        #   For efficiency we compute V only at the summary level (last token),
        #   so we use single-step GAE: Â = r - V(s_prompt_end)
        advantages, returns = self._compute_advantages(token_rewards, batch)

        if self.ppo_cfg.whiten_advantages:
            advantages = whiten(advantages)

        # Cache old stats for logging
        mean_reward = batch.rewards.mean().item()
        mean_adv = advantages.mean().item()

        # Replace batch advantages / returns with freshly computed ones
        batch.advantages = advantages
        batch.returns = returns

        # ── 3. PPO update loop ───────────────────────────────────────────
        accumulated = PPOStats(
            kl_coef=self.kl_ctl.value,
            mean_reward=mean_reward,
            mean_advantage=mean_adv,
            kl_penalty=kl_penalty.mean().item(),
        )

        for _epoch in range(self.ppo_cfg.ppo_epochs):
            for mb in batch.mini_batches(self.ppo_cfg.mini_batch_size):
                stats = self._ppo_mini_batch_update(mb)
                accumulated = accumulated + stats

        # ── 4. Update KL coefficient ─────────────────────────────────────
        approx_kl = accumulated.approx_kl / max(accumulated.n_updates, 1)
        self.kl_ctl.update(approx_kl)

        result = accumulated.mean()
        result.kl_coef = self.kl_ctl.value
        result.mean_reward = mean_reward
        result.mean_advantage = mean_adv
        result.kl_penalty = kl_penalty.mean().item()
        return result

    # ------------------------------------------------------------------

    def _compute_advantages(
        self, token_rewards: torch.Tensor, batch: RolloutBatch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use sequence-level GAE over response tokens.

        token_rewards: (B, T_r)
        batch.values : (B,)  — V at the last prompt token

        We bootstrap token values as V(prompt) for t=0 and 0 for t>0
        (single-step approximation, standard in RLHF implementations).
        """
        B, T_r = token_rewards.shape
        # Build per-token value tensor: only t=0 has the prompt value estimate
        token_values = torch.zeros(B, T_r, device=self.device)
        token_values[:, 0] = batch.values

        adv, ret = compute_gae_sequence(
            token_rewards, token_values,
            gamma=self.ppo_cfg.gamma,
            lam=self.ppo_cfg.lam,
        )
        # For simplicity use summary stats: advantage = sum over tokens
        return adv.sum(dim=1), ret.sum(dim=1)

    # ------------------------------------------------------------------

    def _ppo_mini_batch_update(self, mb: RolloutBatch) -> PPOStats:
        """Single gradient update on one mini-batch."""
        self.optimizer.zero_grad()

        # ── Current policy log-probs ─────────────────────────────────────
        new_log_probs = self.policy.compute_log_probs(
            mb.prompt_ids, mb.prompt_mask, mb.response_ids
        )                                                      # (B, T_r)

        # ── Current value estimates ──────────────────────────────────────
        new_values = self.policy.compute_values(
            mb.prompt_ids, mb.prompt_mask
        )                                                      # (B,)

        # ── Probability ratio (per token, then summed / averaged) ────────
        # Sum log-probs over response tokens for sequence-level ratio
        old_lp_sum = mb.old_log_probs.sum(dim=1)              # (B,)
        new_lp_sum = new_log_probs.sum(dim=1)                 # (B,)
        log_ratio = new_lp_sum - old_lp_sum
        ratio = log_ratio.exp()                                # (B,)

        # ── Clipped surrogate loss ───────────────────────────────────────
        adv = mb.advantages
        surr1 = ratio * adv
        surr2 = ratio.clamp(
            1 - self.ppo_cfg.clip_epsilon,
            1 + self.ppo_cfg.clip_epsilon,
        ) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── Value function loss ──────────────────────────────────────────
        value_loss = F.mse_loss(new_values, mb.returns)

        # ── Entropy bonus ────────────────────────────────────────────────
        # Approximate entropy from current log-probs (per token, then mean)
        entropy = -(new_log_probs * new_log_probs.exp()).sum(dim=1).mean()

        # ── Total loss ───────────────────────────────────────────────────
        loss = (
            policy_loss
            + self.ppo_cfg.vf_coef * value_loss
            - self.ppo_cfg.entropy_coef * entropy
        )

        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy.parameters(), self.ppo_cfg.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()

        # ── Diagnostics ──────────────────────────────────────────────────
        with torch.no_grad():
            approx_kl = (old_lp_sum - new_lp_sum).mean().abs().item()
            clip_frac = (
                (ratio - 1.0).abs() > self.ppo_cfg.clip_epsilon
            ).float().mean().item()

        return PPOStats(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy=entropy.item(),
            total_loss=loss.item(),
            approx_kl=approx_kl,
            clip_fraction=clip_frac,
            n_updates=1,
        )
