"""
training/rlhf_trainer.py — End-to-end RLHF training orchestrator.

Responsibilities
────────────────
  1. Rollout collection
       • Sample prompt batch from dataset
       • Generate responses with the current policy
       • Score responses with the composite reward function
       • Compute reference log-probs for KL penalty

  2. PPO update (delegates to PPOTrainer)

  3. Evaluation
       • Generate samples on a fixed prompt set
       • Report mean reward, KL, and qualitative examples

  4. Checkpointing & logging

Data flow
─────────
  prompts ──► PolicyModel.generate() ──► texts
                                          │
                  ┌─────────────────────┘
                  ▼
           CompositeReward(texts) ──► rewards
                  │
                  └──► RolloutBatch ──► PPOTrainer.update()
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import Config
from models.policy_model import PolicyModel, ReferenceModel, load_tokenizer
from rewards.reward_functions import CompositeReward
from training.ppo_trainer import PPOTrainer, PPOStats
from training.rollout_buffer import RolloutBatch
from data.dataset import PromptDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rollout collector
# ---------------------------------------------------------------------------

class RolloutCollector:
    """
    Generates responses and scores them to build a RolloutBatch.
    """

    def __init__(
        self,
        policy: PolicyModel,
        ref_model: ReferenceModel,
        reward_fn: CompositeReward,
        tokenizer,
        cfg: Config,
        device: torch.device,
    ):
        self.policy = policy
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device

    @torch.no_grad()
    def collect(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> RolloutBatch:
        """
        Generate a rollout batch from the given prompts.

        Returns a fully populated RolloutBatch (on CPU) ready for PPO update.
        """
        self.policy.eval()

        prompt_ids = prompt_ids.to(self.device)
        prompt_mask = prompt_mask.to(self.device)

        # ── 1. Generate responses ────────────────────────────────────────
        gen = self.policy.generate(
            prompt_ids, prompt_mask,
            tokenizer=self.tokenizer,
            cfg=self.cfg.model,
        )
        response_ids = gen["response_ids"]          # (B, T_r)
        old_log_probs = gen["response_log_probs"]   # (B, T_r)
        values = gen["values"]                       # (B,)

        # ── 2. Decode to text for reward scoring ─────────────────────────
        texts = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )

        # ── 3. Compute rewards ───────────────────────────────────────────
        rewards, reward_components = self.reward_fn(
            texts, return_components=True
        )
        rewards = rewards.to(self.device)

        # ── 4. Reference log-probs ───────────────────────────────────────
        ref_log_probs = self.ref_model.compute_log_probs(
            prompt_ids, prompt_mask, response_ids
        )                                           # (B, T_r)

        return RolloutBatch(
            prompt_ids=prompt_ids.cpu(),
            prompt_mask=prompt_mask.cpu(),
            response_ids=response_ids.cpu(),
            old_log_probs=old_log_probs.cpu(),
            ref_log_probs=ref_log_probs.cpu(),
            rewards=rewards.cpu(),
            values=values.cpu(),
            advantages=torch.zeros_like(rewards),   # filled by PPOTrainer
            returns=torch.zeros_like(rewards),
            texts=texts,
            reward_components={k: v.cpu() for k, v in reward_components.items()},
        )


# ---------------------------------------------------------------------------
# Main RLHF Trainer
# ---------------------------------------------------------------------------

class RLHFTrainer:
    """
    Orchestrates the full RLHF-PPO training pipeline.

    Usage:
        cfg = positive_sentiment_config()
        trainer = RLHFTrainer(cfg)
        trainer.train()
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        # Output directory
        self.output_dir = Path(cfg.training.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Models ──────────────────────────────────────────────────────
        logger.info(f"Loading policy model: {cfg.model.model_name}")
        self.tokenizer = load_tokenizer(cfg.model.model_name)
        self.policy = PolicyModel(cfg.model).to(self.device)

        logger.info(f"Loading reference model: {cfg.model.ref_model_name}")
        self.ref_model = ReferenceModel(cfg.model.ref_model_name).to(self.device)

        # ── Reward function ──────────────────────────────────────────────
        logger.info("Initialising reward function …")
        self.reward_fn = CompositeReward(cfg.reward, device=str(self.device))

        # ── PPO trainer ──────────────────────────────────────────────────
        self.ppo_trainer = PPOTrainer(
            policy=self.policy,
            ref_model=self.ref_model,
            ppo_cfg=cfg.ppo,
            train_cfg=cfg.training,
            device=self.device,
        )

        # ── Rollout collector ────────────────────────────────────────────
        self.collector = RolloutCollector(
            policy=self.policy,
            ref_model=self.ref_model,
            reward_fn=self.reward_fn,
            tokenizer=self.tokenizer,
            cfg=cfg,
            device=self.device,
        )

        # ── Dataset ──────────────────────────────────────────────────────
        logger.info(f"Loading dataset: {cfg.training.dataset_name}")
        self.dataset = PromptDataset(
            dataset_name=cfg.training.dataset_name,
            split=cfg.training.dataset_split,
            tokenizer=self.tokenizer,
            max_prompt_length=cfg.training.max_prompt_length,
            seed=cfg.training.seed,
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=self.dataset.collate_fn,
        )
        self._data_iter = iter(self.dataloader)

        # ── W&B ──────────────────────────────────────────────────────────
        self.wandb = None
        if cfg.training.wandb_project:
            try:
                import wandb
                wandb.init(
                    project=cfg.training.wandb_project,
                    config=vars(cfg),
                    name=f"rlhf-ppo-{int(time.time())}",
                )
                self.wandb = wandb
            except ImportError:
                logger.warning("wandb not installed — skipping W&B logging")

        self.global_step = 0

    # ------------------------------------------------------------------
    # Data sampling helper
    # ------------------------------------------------------------------

    def _next_prompt_batch(self):
        try:
            batch = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self.dataloader)
            batch = next(self._data_iter)
        return batch["input_ids"], batch["attention_mask"]

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self):
        logger.info(f"\n{self.cfg.summary()}")
        logger.info("Starting RLHF-PPO training …")

        total_steps = self.cfg.training.total_steps
        log_every = self.cfg.training.log_every
        save_every = self.cfg.training.save_every
        eval_every = self.cfg.training.eval_every

        running_stats = PPOStats()
        t_start = time.time()

        for step in range(1, total_steps + 1):
            self.global_step = step

            # ── Rollout ─────────────────────────────────────────────────
            prompt_ids, prompt_mask = self._next_prompt_batch()
            rollout = self.collector.collect(prompt_ids, prompt_mask)

            # ── PPO update ──────────────────────────────────────────────
            stats = self.ppo_trainer.update(rollout)
            running_stats = running_stats + stats
            running_stats.n_updates += 1

            # ── Logging ─────────────────────────────────────────────────
            if step % log_every == 0:
                elapsed = time.time() - t_start
                mean_stats = running_stats.mean()
                self._log_step(step, mean_stats, elapsed, rollout)
                if self.wandb:
                    self.wandb.log(mean_stats.to_dict(), step=step)
                running_stats = PPOStats()

            # ── Evaluation ──────────────────────────────────────────────
            if step % eval_every == 0:
                self.evaluate()

            # ── Checkpoint ──────────────────────────────────────────────
            if step % save_every == 0:
                self.save_checkpoint(step)

        logger.info("Training complete!")
        self.save_checkpoint("final")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_step(
        self,
        step: int,
        stats: PPOStats,
        elapsed: float,
        rollout: RolloutBatch,
    ):
        steps_per_sec = self.cfg.training.log_every / max(elapsed, 1e-6)
        print(
            f"\n[Step {step:>5}] "
            f"reward={stats.mean_reward:+.3f}  "
            f"policy_loss={stats.policy_loss:.4f}  "
            f"value_loss={stats.value_loss:.4f}  "
            f"approx_kl={stats.approx_kl:.4f}  "
            f"kl_coef={stats.kl_coef:.4f}  "
            f"clip_frac={stats.clip_fraction:.3f}  "
            f"({steps_per_sec:.1f} steps/s)"
        )
        # Print one example
        if rollout.texts:
            print(f"  Example: {rollout.texts[0][:120]!r}")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self):
        """Generate samples on a fixed prompt set and report metrics."""
        logger.info(f"\n{'─'*60}\n  Evaluation @ step {self.global_step}\n{'─'*60}")
        self.policy.eval()

        # Use the first batch of the dataset as a fixed eval set
        eval_prompts = self.dataset.get_eval_prompts(
            n=self.cfg.training.eval_prompts
        )
        prompt_ids = eval_prompts["input_ids"].to(self.device)
        prompt_mask = eval_prompts["attention_mask"].to(self.device)

        gen = self.policy.generate(
            prompt_ids, prompt_mask,
            tokenizer=self.tokenizer,
            cfg=self.cfg.model,
        )
        texts = self.tokenizer.batch_decode(
            gen["response_ids"], skip_special_tokens=True
        )

        # Score
        total_rewards, components = self.reward_fn(
            texts, return_components=True
        )
        mean_r = total_rewards.mean().item()

        print(f"  eval/mean_reward  = {mean_r:+.4f}")
        for name, r in components.items():
            print(f"  eval/{name:<20} = {r.mean().item():+.4f}")

        # Print 3 qualitative examples
        print("\n  Sample outputs:")
        for i in range(min(3, len(texts))):
            print(f"  [{i+1}] {texts[i][:150]!r}")

        if self.wandb:
            log_dict = {"eval/mean_reward": mean_r}
            log_dict.update({
                f"eval/{k}": v.mean().item() for k, v in components.items()
            })
            self.wandb.log(log_dict, step=self.global_step)

        logger.info("─" * 60)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag):
        ckpt_dir = self.output_dir / f"checkpoint-{tag}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # Save backbone + value head
        self.policy.backbone.save_pretrained(ckpt_dir / "backbone")
        self.tokenizer.save_pretrained(ckpt_dir / "backbone")
        torch.save(
            self.policy.value_head.state_dict(),
            ckpt_dir / "value_head.pt",
        )
        torch.save(
            self.ppo_trainer.optimizer.state_dict(),
            ckpt_dir / "optimizer.pt",
        )
        logger.info(f"Checkpoint saved → {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir: str):
        ckpt_dir = Path(ckpt_dir)
        from transformers import AutoModelForCausalLM
        self.policy.backbone = AutoModelForCausalLM.from_pretrained(
            ckpt_dir / "backbone"
        ).to(self.device)
        self.policy.value_head.load_state_dict(
            torch.load(ckpt_dir / "value_head.pt", map_location=self.device)
        )
        logger.info(f"Checkpoint loaded ← {ckpt_dir}")
