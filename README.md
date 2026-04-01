# RLHF-PPO: Controllable Text Generation via Reinforcement Learning from Human Feedback

A **complete, production-quality** implementation of PPO-based RLHF for
controllable language model fine-tuning. Covers everything from the mathematical
foundations to a working training loop, multi-objective reward design, and
evaluation utilities.

---

## Table of Contents

1. [Overview](#overview)
2. [Theory: How RLHF-PPO Works](#theory)
3. [Architecture](#architecture)
4. [Reward Functions](#reward-functions)
5. [Project Structure](#project-structure)
6. [Quick Start](#quick-start)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Interactive Demo](#interactive-demo)
10. [Configuration Reference](#configuration-reference)
11. [Testing](#testing)
12. [References](#references)

---

## Overview

**RLHF** (Reinforcement Learning from Human Feedback) is the technique used to
align language models with human preferences — most famously in InstructGPT,
ChatGPT, and Claude. This project implements the core **PPO-based RLHF loop**
from scratch with:

| Feature | Details |
|---|---|
| **Base model** | GPT-2 (any HF causal LM) |
| **RL algorithm** | Proximal Policy Optimisation (PPO) |
| **Reward design** | Composite: sentiment + toxicity + fluency + length |
| **KL control** | Adaptive β controller (Ziegler et al. 2019) |
| **Advantage estimation** | GAE (Schulman et al. 2016) |
| **Controllability** | Sentiment direction, detoxification, balanced |
| **Evaluation** | Reward metrics, KL, distinct-n diversity |
| **Logging** | Terminal + optional Weights & Biases |

---

## Theory: How RLHF-PPO Works

### 1. The RLHF Objective

We want to fine-tune a language model π_θ to maximise expected reward while
staying close to the original (reference) model π_ref:

```
max_θ  E_{s~D, a~π_θ(·|s)} [r(s, a)] − β · KL(π_θ(·|s) ‖ π_ref(·|s))
```

where:
- `s` is the prompt (state)
- `a` is the generated response (action)
- `r(s, a)` is the reward from our custom reward function
- `β` is the KL penalty coefficient (prevents the model from "gaming" the reward)

### 2. The KL Penalty

The KL term is crucial — without it, the model will quickly diverge from
sensible language and find degenerate ways to maximise the reward. We fold it
into per-token rewards:

```
r̂_t = r_env · 𝟙[t = T] − β · (log π_θ(a_t) − log π_ref(a_t))
```

The coefficient β is **adapted automatically** using:

```
if KL > target * 1.5:   β ← β * (1 + speed)
if KL < target / 1.5:   β ← β * (1 - speed)
```

### 3. Generalised Advantage Estimation (GAE)

We estimate the advantage function using GAE (λ = 0.95):

```
δ_t  = r_t + γ V(s_{t+1}) − V(s_t)
Â_t  = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

GAE smoothly interpolates between the high-variance Monte Carlo estimate (λ=1)
and the low-variance but biased TD estimate (λ=0).

### 4. PPO Update

For each rollout batch, we perform `ppo_epochs` passes with clipped surrogate loss:

```
ρ_t        = π_θ(a_t|s_t) / π_old(a_t|s_t)        # probability ratio

L_CLIP     = -E[ min(ρ_t Â_t,  clip(ρ_t, 1-ε, 1+ε) Â_t) ]

L_VALUE    = 0.5 · MSE(V_θ(s), R_t)                 # critic loss

L_ENTROPY  = -H[π_θ(·|s)]                           # exploration bonus

L_total    = L_CLIP + c_vf · L_VALUE − c_ent · L_ENTROPY
```

The clip ratio `ε` (default 0.2) prevents large policy updates that would
destabilise training.

### 5. The Actor-Critic Architecture

```
                    Prompt tokens
                         │
                   ┌─────▼──────┐
                   │ Transformer │  (shared backbone, updated by PPO)
                   │  Backbone  │
                   └──────┬─────┘
                          │ hidden states
               ┌──────────┴──────────┐
               ▼                     ▼
         LM Head (actor)       Value Head (critic)
         → token logits        → scalar V(s)
         → sample action a     → used for GAE
```

---

## Architecture

### Component Map

```
rlhf_ppo/
├── config.py                    # All hyperparameters (single source of truth)
│
├── models/
│   ├── policy_model.py          # PolicyModel (actor + value head)
│   │                            # ReferenceModel (frozen, for KL)
│   │                            # ValueHead (2-layer MLP)
│   └── __init__.py
│
├── rewards/
│   ├── reward_functions.py      # SentimentReward, ToxicityReward,
│   │                            # FluencyReward, LengthReward,
│   │                            # CompositeReward, RunningMeanStd
│   └── __init__.py
│
├── training/
│   ├── rollout_buffer.py        # RolloutBatch, compute_gae, whiten
│   ├── ppo_trainer.py           # PPOTrainer, AdaptiveKLController, PPOStats
│   ├── rlhf_trainer.py          # RLHFTrainer (full loop), RolloutCollector
│   └── __init__.py
│
├── data/
│   ├── dataset.py               # PromptDataset, SyntheticPromptDataset
│   └── __init__.py
│
├── train.py                     # CLI entry point for training
├── evaluate.py                  # Evaluation + comparison script
├── demo.py                      # Interactive generation demo
├── requirements.txt
└── tests/
    └── test_all.py              # 20+ unit tests across all components
```

---

## Reward Functions

### SentimentReward
Uses **DistilBERT fine-tuned on SST-2** to steer the model toward a target
polarity. The reward is the probability of the target class mapped to `[-1, +1]`.

```
r_sentiment = 2 * P(target_label) - 1
```

### ToxicityReward
Uses **toxic-bert** (or a keyword heuristic as fallback) to penalise harmful
content. Score is negated so higher toxicity → more negative reward.

```
r_toxicity = -min(toxicity_score / max_toxicity, 1.0)
```

### FluencyReward
Measures how surprised a **frozen GPT-2** is by the generated text (perplexity).
Lower perplexity → more natural language → higher reward.

```
r_fluency = max(0, 1 - log_ppl / log_ppl_max)
```

### LengthReward
Gaussian bell-curve centred on the ideal token count to discourage degenerate
very-short or very-long outputs.

```
r_length = 2 * exp(-0.5 * ((len - ideal) / sigma)²) - 1
```

### CompositeReward
Weighted sum with optional running-mean-std whitening across the rollout batch:

```
r_total = w_sent · r_sentiment
        + w_tox  · r_toxicity
        + w_flu  · r_fluency
        + w_len  · r_length
```

---

## Quick Start

### Installation

```bash
git clone <this-repo>
cd rlhf_ppo
pip install -r requirements.txt
```

### Minimal Training Run (offline, no internet required)

```bash
python train.py --synthetic --steps 50 --batch-size 8 --log-every 5
```

### Standard Training (downloads IMDb + models on first run)

```bash
python train.py --mode sentiment --steps 500
```

---

## Training

### CLI Options

```
python train.py [OPTIONS]

  --mode         {sentiment, detox, balanced}   Reward preset
  --model        STR    HuggingFace model name (default: gpt2)
  --steps        INT    Total training steps (default: 500)
  --batch-size   INT    Rollout batch size (default: 32)
  --lr           FLOAT  Learning rate (default: 1.4e-5)
  --output-dir   STR    Checkpoint directory (default: outputs/rlhf_ppo)
  --wandb        STR    W&B project name (optional)
  --synthetic    FLAG   Use offline synthetic prompts
  --seed         INT    Random seed (default: 42)
  --kl-coef      FLOAT  Initial KL penalty coefficient (default: 0.2)
  --clip-epsilon FLOAT  PPO clip ratio ε (default: 0.2)
  --ppo-epochs   INT    PPO optimisation epochs per step (default: 4)
```

### Example Commands

```bash
# Positive sentiment, GPT-2, 500 steps
python train.py --mode sentiment

# Detoxification with GPT-2-medium, W&B logging
python train.py --mode detox --model gpt2-medium --wandb my-rlhf-project

# Custom KL and clip settings
python train.py --kl-coef 0.1 --clip-epsilon 0.15 --ppo-epochs 2

# Fast test run with synthetic prompts
python train.py --synthetic --steps 20 --batch-size 4 --log-every 2
```

### Training Log Sample

```
[Step    10] reward=+0.312  policy_loss=0.0234  value_loss=0.1823
             approx_kl=0.0041  kl_coef=0.1983  clip_frac=0.021 (2.3 steps/s)
  Example: ' brilliant performances throughout, the cast delivers ...'

[Step    20] reward=+0.489  policy_loss=0.0189  value_loss=0.1241
             approx_kl=0.0073  kl_coef=0.1997  clip_frac=0.028 (2.5 steps/s)
```

---

## Evaluation

```bash
# Evaluate a checkpoint
python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final

# Compare with untuned base model
python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final --compare-base

# Save results
python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final \
                   --save-csv results.csv --save-json results.json
```

**Metrics reported:**

| Metric | Description |
|---|---|
| `reward_mean` | Mean total reward across eval samples |
| `reward_std` | Standard deviation of rewards |
| `kl_from_ref` | Approximate KL(policy ‖ reference) |
| `distinct_1` | Ratio of unique unigrams (diversity) |
| `distinct_2` | Ratio of unique bigrams (diversity) |
| `avg_length_words` | Mean response length in words |
| `component_*` | Per-component reward means |

---

## Interactive Demo

```bash
# Interactive REPL (enter prompts, see generated continuations + rewards)
python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final

# Single prompt mode
python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final \
               --prompt "The movie was absolutely" --n 5

# Compare with untuned base model
python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final --compare

# Run without a checkpoint (base model demo)
python demo.py --no-checkpoint
```

The demo displays colour-coded rewards (🟢 positive / 🔴 negative) with
a per-component breakdown for each generated continuation.

---

## Configuration Reference

### ModelConfig

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `gpt2` | HuggingFace model identifier |
| `ref_model_name` | same as model | Reference model (frozen) |
| `max_new_tokens` | `64` | Maximum tokens to generate |
| `temperature` | `1.0` | Sampling temperature |
| `top_k` | `50` | Top-k sampling |
| `top_p` | `0.95` | Nucleus sampling p |

### PPOConfig

| Parameter | Default | Description |
|---|---|---|
| `clip_epsilon` | `0.2` | PPO clip ratio ε |
| `vf_coef` | `0.5` | Value function loss weight |
| `entropy_coef` | `0.01` | Entropy bonus weight |
| `gamma` | `1.0` | Discount factor |
| `lam` | `0.95` | GAE λ |
| `ppo_epochs` | `4` | Optimisation epochs per rollout |
| `mini_batch_size` | `8` | Mini-batch size |
| `target_kl` | `0.1` | Target KL for adaptive β |
| `init_kl_coef` | `0.2` | Initial KL penalty β |
| `whiten_advantages` | `True` | Normalise advantages |

### RewardConfig

| Parameter | Default | Description |
|---|---|---|
| `sentiment_weight` | `1.0` | Sentiment reward weight |
| `toxicity_weight` | `0.5` | Toxicity penalty weight |
| `fluency_weight` | `0.3` | Fluency reward weight |
| `length_weight` | `0.1` | Length reward weight |
| `target_sentiment` | `POSITIVE` | Target polarity |
| `normalize_rewards` | `True` | Whiten rewards per batch |

---

## Testing

```bash
# Run the full test suite
python -m pytest tests/ -v

# Or directly
python tests/test_all.py
```

**20+ tests covering:**
- Config presets and auto-filling
- GAE computation (single-step and sequence)
- Advantage whitening
- Mini-batch sampling from `RolloutBatch`
- `ValueHead` output shapes
- Reward function shapes, values, and edge cases
- `RunningMeanStd` convergence
- `AdaptiveKLController` behaviour
- `SyntheticPromptDataset` and `collate_fn`
- `PPOStats` arithmetic and serialisation
- Diversity metrics (`distinct_n`, `average_length`)

---

## References

1. **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms*
   https://arxiv.org/abs/1707.06347

2. **Ouyang et al. (2022)** — *Training language models to follow instructions
   with human feedback* (InstructGPT)
   https://arxiv.org/abs/2203.02155

3. **Ziegler et al. (2019)** — *Fine-Tuning Language Models from Human Preferences*
   https://arxiv.org/abs/1909.08593

4. **Schulman et al. (2016)** — *High-Dimensional Continuous Control Using
   Generalised Advantage Estimation*
   https://arxiv.org/abs/1506.02438

5. **Stiennon et al. (2020)** — *Learning to summarize with human feedback*
   https://arxiv.org/abs/2009.01325

6. **TRL Library** (HuggingFace) — practical PPO implementation reference
   https://github.com/huggingface/trl
