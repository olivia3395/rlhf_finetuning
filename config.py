"""
config.py — Central configuration for PPO-based RLHF training.

All hyperparameters, paths, and feature flags live here so that
train.py, evaluate.py, and the demo can import a single source of truth.
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Policy / value-head model settings."""

    # Base language model (any causal-LM on HuggingFace Hub)
    model_name: str = "gpt2"

    # Reference model kept frozen for KL-divergence penalty
    ref_model_name: Optional[str] = None          # None → same as model_name

    # Generation
    max_new_tokens: int = 64
    min_new_tokens: int = 8
    do_sample: bool = True
    top_k: int = 50
    top_p: float = 0.95
    temperature: float = 1.0

    # Value head
    value_head_hidden_size: int = 256
    value_head_dropout: float = 0.1


# ---------------------------------------------------------------------------
# PPO configuration
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Proximal Policy Optimisation hyper-parameters."""

    # Clip ratio ε
    clip_epsilon: float = 0.2

    # Value function loss coefficient
    vf_coef: float = 0.5

    # Entropy bonus coefficient (encourages exploration)
    entropy_coef: float = 0.01

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Discount factor γ
    gamma: float = 1.0

    # GAE λ (Generalised Advantage Estimation)
    lam: float = 0.95

    # Number of PPO optimisation epochs per rollout batch
    ppo_epochs: int = 4

    # Mini-batch size inside each PPO epoch
    mini_batch_size: int = 8

    # Target KL divergence for adaptive penalty (None → fixed β)
    target_kl: Optional[float] = 0.1

    # Initial KL penalty coefficient β
    init_kl_coef: float = 0.2

    # KL penalty adaptation speed
    kl_adapt_speed: float = 0.02

    # Whiten advantages before PPO update
    whiten_advantages: bool = True


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """Weights and settings for the composite reward function."""

    # Sentiment reward: +1 for positive, -1 for negative
    sentiment_weight: float = 1.0

    # Toxicity penalty (negative reward for toxic text)
    toxicity_weight: float = 0.5

    # Fluency reward (log-perplexity under a frozen reference LM)
    fluency_weight: float = 0.3

    # Length penalty: penalise outputs that are too short or too long
    length_weight: float = 0.1

    # Target sentiment label ("POSITIVE" | "NEGATIVE" | None → no constraint)
    target_sentiment: Optional[str] = "POSITIVE"

    # Maximum allowed toxicity score (above this → max penalty)
    max_toxicity: float = 0.3

    # Ideal response length (tokens)
    ideal_length: int = 48

    # Reward normalisation (running mean/std whitening)
    normalize_rewards: bool = True
    reward_norm_momentum: float = 0.01


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """End-to-end training loop settings."""

    # Output directory for checkpoints and logs
    output_dir: str = "outputs/rlhf_ppo"

    # Dataset (HuggingFace Hub identifier or local path)
    dataset_name: str = "imdb"
    dataset_split: str = "train"

    # Number of prompt samples per rollout step
    batch_size: int = 32

    # Total number of training steps (rollout + PPO update = 1 step)
    total_steps: int = 500

    # Logging interval (steps)
    log_every: int = 10

    # Checkpoint interval (steps)
    save_every: int = 100

    # Evaluation interval (steps)
    eval_every: int = 50

    # Number of evaluation prompts
    eval_prompts: int = 64

    # Random seed
    seed: int = 42

    # Learning rate for actor + critic
    learning_rate: float = 1.4e-5

    # Adam ε
    adam_epsilon: float = 1e-8

    # Weight decay
    weight_decay: float = 0.0

    # Warmup steps for LR scheduler
    warmup_steps: int = 20

    # Mixed-precision training ("fp16" | "bf16" | "no")
    mixed_precision: str = "no"

    # Weights & Biases project name (None → disable W&B)
    wandb_project: Optional[str] = None

    # Max prompt length (tokens)
    max_prompt_length: int = 64


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Aggregates all sub-configs into a single object."""

    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        # Resolve reference model name
        if self.model.ref_model_name is None:
            self.model.ref_model_name = self.model.model_name

    def summary(self) -> str:
        lines = ["=" * 60, "  RLHF-PPO Configuration", "=" * 60]
        for section_name in ("model", "ppo", "reward", "training"):
            section = getattr(self, section_name)
            lines.append(f"\n[{section_name.upper()}]")
            for k, v in vars(section).items():
                lines.append(f"  {k:<30} {v}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preset factory functions
# ---------------------------------------------------------------------------

def positive_sentiment_config() -> Config:
    """Train GPT-2 to produce positive-sentiment continuations."""
    cfg = Config()
    cfg.reward.target_sentiment = "POSITIVE"
    cfg.reward.sentiment_weight = 1.5
    cfg.reward.toxicity_weight = 0.5
    cfg.reward.fluency_weight = 0.2
    return cfg


def detoxification_config() -> Config:
    """Train GPT-2 to avoid toxic language."""
    cfg = Config()
    cfg.reward.target_sentiment = None
    cfg.reward.sentiment_weight = 0.0
    cfg.reward.toxicity_weight = 2.0
    cfg.reward.fluency_weight = 0.5
    return cfg


def balanced_config() -> Config:
    """Balanced multi-objective reward."""
    cfg = Config()
    cfg.reward.target_sentiment = "POSITIVE"
    cfg.reward.sentiment_weight = 1.0
    cfg.reward.toxicity_weight = 1.0
    cfg.reward.fluency_weight = 0.5
    cfg.reward.length_weight = 0.2
    return cfg
