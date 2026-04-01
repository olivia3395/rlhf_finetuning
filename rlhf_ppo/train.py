"""
train.py — Entry point for RLHF-PPO training.

Quick start
───────────
  # Train GPT-2 to generate positive-sentiment text (default)
  python train.py

  # Detoxification mode
  python train.py --mode detox

  # Balanced multi-objective
  python train.py --mode balanced

  # Custom settings
  python train.py --model gpt2-medium --steps 1000 --batch-size 16

  # Use synthetic prompts (offline, no HuggingFace download needed)
  python train.py --synthetic

CLI flags
─────────
  --mode        {sentiment, detox, balanced}   preset reward config
  --model       HuggingFace model name         (default: gpt2)
  --steps       total training steps           (default: 500)
  --batch-size  rollout batch size             (default: 32)
  --lr          learning rate                  (default: 1.4e-5)
  --output-dir  checkpoint output directory    (default: outputs/rlhf_ppo)
  --wandb       W&B project name               (default: disabled)
  --synthetic   use synthetic offline prompts  (flag)
  --seed        random seed                    (default: 42)
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

# ── Add project root to path ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    Config,
    balanced_config,
    detoxification_config,
    positive_sentiment_config,
)
from training.rlhf_trainer import RLHFTrainer

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO-based RLHF for controllable text generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["sentiment", "detox", "balanced"],
        default="sentiment",
        help="Preset reward configuration",
    )
    parser.add_argument("--model", default="gpt2", help="HuggingFace model name")
    parser.add_argument("--steps", type=int, default=500, help="Total training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Rollout batch size")
    parser.add_argument("--lr", type=float, default=1.4e-5, help="Learning rate")
    parser.add_argument(
        "--output-dir", default="outputs/rlhf_ppo", help="Checkpoint directory"
    )
    parser.add_argument("--wandb", default=None, help="W&B project name (optional)")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic offline prompts (no internet required)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--log-every", type=int, default=10, help="Logging interval (steps)"
    )
    parser.add_argument(
        "--eval-every", type=int, default=50, help="Evaluation interval (steps)"
    )
    parser.add_argument(
        "--save-every", type=int, default=100, help="Checkpoint interval (steps)"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=64, help="Max tokens to generate"
    )
    parser.add_argument(
        "--kl-coef", type=float, default=0.2, help="Initial KL penalty coefficient"
    )
    parser.add_argument(
        "--clip-epsilon", type=float, default=0.2, help="PPO clip ratio epsilon"
    )
    parser.add_argument(
        "--ppo-epochs", type=int, default=4, help="PPO optimisation epochs per step"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def build_config(args: argparse.Namespace) -> Config:
    """Create a Config from CLI arguments, starting from a preset."""
    preset_map = {
        "sentiment": positive_sentiment_config,
        "detox": detoxification_config,
        "balanced": balanced_config,
    }
    cfg = preset_map[args.mode]()

    # Override with CLI values
    cfg.model.model_name = args.model
    cfg.model.ref_model_name = args.model
    cfg.model.max_new_tokens = args.max_new_tokens

    cfg.training.total_steps = args.steps
    cfg.training.batch_size = args.batch_size
    cfg.training.learning_rate = args.lr
    cfg.training.output_dir = args.output_dir
    cfg.training.wandb_project = args.wandb
    cfg.training.seed = args.seed
    cfg.training.log_every = args.log_every
    cfg.training.eval_every = args.eval_every
    cfg.training.save_every = args.save_every

    cfg.ppo.init_kl_coef = args.kl_coef
    cfg.ppo.clip_epsilon = args.clip_epsilon
    cfg.ppo.ppo_epochs = args.ppo_epochs

    if args.synthetic:
        # Override dataset with a flag the trainer checks
        cfg.training.dataset_name = "__synthetic__"

    return cfg


# ---------------------------------------------------------------------------
# Synthetic dataset monkey-patch
# ---------------------------------------------------------------------------

def patch_synthetic_dataset(trainer: "RLHFTrainer"):
    """Replace the HF dataset with the offline synthetic one."""
    from data.dataset import SyntheticPromptDataset
    from torch.utils.data import DataLoader

    logger.info("Using SYNTHETIC prompts (offline mode)")
    trainer.dataset = SyntheticPromptDataset(
        tokenizer=trainer.tokenizer,
        n_samples=2000,
        seed=trainer.cfg.training.seed,
    )
    trainer.dataloader = DataLoader(
        trainer.dataset,
        batch_size=trainer.cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=trainer.dataset.collate_fn,
    )
    trainer._data_iter = iter(trainer.dataloader)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Config
    cfg = build_config(args)

    # Print config summary
    print(cfg.summary())

    # Build trainer
    trainer = RLHFTrainer(cfg)

    # Optionally replace dataset with synthetic prompts
    if args.synthetic or cfg.training.dataset_name == "__synthetic__":
        patch_synthetic_dataset(trainer)

    # Train!
    trainer.train()

    print("\n✅  Training complete!")
    print(f"   Checkpoints saved to: {cfg.training.output_dir}")


if __name__ == "__main__":
    main()
