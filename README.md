<div align="center">

<img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/W&B-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=black"/>
<img src="https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge"/>

<br/><br/>

# 🤖 RLHF-PPO
### Controllable Text Generation via Reinforcement Learning from Human Feedback

<br/>

> A from-scratch implementation of the **PPO-based RLHF loop** —  
> the same technique powering **InstructGPT**, **ChatGPT**, and **Claude**.

<br/>

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [🎯 Reward Design](#-reward-functions) · [📊 Evaluation](#-evaluation) · [📎 References](#-references)

<br/>



</div>

## ✨ Features at a Glance

<div align="center">

| Component | Details |
|:---|:---|
| 🧠 **Base model** | GPT-2 (any HuggingFace causal LM) |
| 🔁 **RL algorithm** | Proximal Policy Optimisation (PPO) |
| 🎯 **Reward design** | Composite: sentiment + toxicity + fluency + length |
| 🎛️ **KL control** | Adaptive β controller (Ziegler et al. 2019) |
| 📈 **Advantage estimation** | GAE with λ = 0.95 (Schulman et al. 2016) |
| 🕹️ **Controllability** | Sentiment direction · detoxification · balanced |
| 📊 **Evaluation** | Reward metrics · KL divergence · distinct-n diversity |
| 📝 **Logging** | Terminal output + optional Weights & Biases |

</div>

<br/>



## 🏗️ Architecture

```
              Prompt tokens
                   │
             ┌─────▼──────┐
             │ Transformer │   ← shared backbone, updated by PPO
             │  Backbone   │
             └──────┬──────┘
                    │  hidden states
         ┌──────────┴──────────┐
         ▼                     ▼
   LM Head  (actor)      Value Head  (critic)
   → token logits        → scalar  V(s)
   → sample action a     → used for GAE / advantage
```

<br/>



## 🎯 Reward Functions

Four differentiable reward signals are combined into a single weighted composite score.

<br/>

### 😊 SentimentReward
Uses **DistilBERT fine-tuned on SST-2** to steer outputs toward a target polarity:

```
r_sentiment = 2 * P(target_label) - 1      # in [-1, +1]
```

### ☣️ ToxicityReward
Uses **toxic-bert** (keyword heuristic fallback) to penalise harmful content:

```
r_toxicity = -min(toxicity_score / max_toxicity, 1.0)
```

### 📝 FluencyReward
Measures perplexity under a **frozen GPT-2** — lower perplexity means more natural text:

```
r_fluency = max(0, 1 - log_ppl / log_ppl_max)
```

### 📏 LengthReward
Gaussian bell-curve centred on the ideal token count, discouraging degenerate outputs:

```
r_length = 2 * exp(-0.5 * ((len - ideal) / sigma)^2) - 1
```

### 🧮 CompositeReward

```
r_total = w_sent · r_sentiment
        + w_tox  · r_toxicity
        + w_flu  · r_fluency
        + w_len  · r_length
```

Optional running-mean-std **reward whitening** is applied across each rollout batch.

<br/>



## 🚀 Quick Start

### Installation

```bash
git clone <this-repo>
cd rlhf_ppo
pip install -r requirements.txt
```

### Minimal Run — offline, no internet required

```bash
python train.py --synthetic --steps 50 --batch-size 8 --log-every 5
```

### Standard Training — downloads IMDb + models on first run

```bash
python train.py --mode sentiment --steps 500
```

<br/>



## 🏋️ Training

### CLI Reference

```
python train.py [OPTIONS]

  --mode         {sentiment, detox, balanced}   Reward preset
  --model        STR    HuggingFace model name        (default: gpt2)
  --steps        INT    Total training steps           (default: 500)
  --batch-size   INT    Rollout batch size             (default: 32)
  --lr           FLOAT  Learning rate                  (default: 1.4e-5)
  --output-dir   STR    Checkpoint directory           (default: outputs/rlhf_ppo)
  --wandb        STR    W&B project name               (optional)
  --synthetic    FLAG   Use offline synthetic prompts
  --seed         INT    Random seed                    (default: 42)
  --kl-coef      FLOAT  Initial KL penalty coefficient (default: 0.2)
  --clip-epsilon FLOAT  PPO clip ratio ε               (default: 0.2)
  --ppo-epochs   INT    PPO epochs per rollout step    (default: 4)
```

### Example Commands

```bash
# Positive sentiment steering with GPT-2
python train.py --mode sentiment

# Detoxification with GPT-2-medium + W&B logging
python train.py --mode detox --model gpt2-medium --wandb my-rlhf-project

# Custom KL and clip settings
python train.py --kl-coef 0.1 --clip-epsilon 0.15 --ppo-epochs 2

# Fast offline test run
python train.py --synthetic --steps 20 --batch-size 4 --log-every 2
```

### Training Log Sample

```
[Step    10] reward=+0.312  policy_loss=0.0234  value_loss=0.1823
             approx_kl=0.0041  kl_coef=0.1983  clip_frac=0.021  (2.3 steps/s)
  ↳ ' brilliant performances throughout, the cast delivers ...'

[Step    20] reward=+0.489  policy_loss=0.0189  value_loss=0.1241
             approx_kl=0.0073  kl_coef=0.1997  clip_frac=0.028  (2.5 steps/s)
```

<br/>


## 📊 Evaluation

```bash
# Evaluate a checkpoint
python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final

# Compare with untuned base model
python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final --compare-base

# Export results
python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final \
                   --save-csv results.csv --save-json results.json
```

**Metrics reported:**

| Metric | Description |
|:---|:---|
| `reward_mean` | Mean total reward across eval samples |
| `reward_std` | Standard deviation of rewards |
| `kl_from_ref` | Approximate KL(policy ‖ reference) |
| `distinct_1` | Unique unigram ratio (lexical diversity) |
| `distinct_2` | Unique bigram ratio (lexical diversity) |
| `avg_length_words` | Mean response length in words |
| `component_*` | Per-component reward breakdown |

<br/>



## 🎮 Demo

```bash
# Interactive REPL — type prompts, see completions + reward scores
python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final

# Single prompt, 5 samples
python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final \
               --prompt "The movie was absolutely" --n 5

# Side-by-side comparison with untuned base model
python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final --compare

# Base model only (no checkpoint required)
python demo.py --no-checkpoint
```

<br/>



## ⚙️ Configuration Reference

### ModelConfig

| Parameter | Default | Description |
|:---|:---:|:---|
| `model_name` | `gpt2` | HuggingFace model identifier |
| `ref_model_name` | *(same)* | Reference model (frozen) |
| `max_new_tokens` | `64` | Maximum tokens to generate |
| `temperature` | `1.0` | Sampling temperature |
| `top_k` | `50` | Top-k sampling |
| `top_p` | `0.95` | Nucleus sampling p |

### PPOConfig

| Parameter | Default | Description |
|:---|:---:|:---|
| `clip_epsilon` | `0.2` | PPO clip ratio ε |
| `vf_coef` | `0.5` | Value function loss weight |
| `entropy_coef` | `0.01` | Entropy bonus weight |
| `gamma` | `1.0` | Discount factor |
| `lam` | `0.95` | GAE λ |
| `ppo_epochs` | `4` | Optimisation epochs per rollout |
| `mini_batch_size` | `8` | Mini-batch size for gradient steps |
| `target_kl` | `0.1` | Target KL for adaptive β |
| `init_kl_coef` | `0.2` | Initial KL penalty β |
| `whiten_advantages` | `True` | Normalise advantages per batch |

### RewardConfig

| Parameter | Default | Description |
|:---|:---:|:---|
| `sentiment_weight` | `1.0` | Sentiment reward weight |
| `toxicity_weight` | `0.5` | Toxicity penalty weight |
| `fluency_weight` | `0.3` | Fluency reward weight |
| `length_weight` | `0.1` | Length reward weight |
| `target_sentiment` | `POSITIVE` | Target polarity |
| `normalize_rewards` | `True` | Whiten rewards per batch |

<br/>



## 🧪 Testing

```bash
# Full test suite
python -m pytest tests/ -v

# Or directly
python tests/test_all.py
```

**20+ tests covering:**

- ✅ Config presets and auto-filling
- ✅ GAE computation (single-step and multi-step sequences)
- ✅ Advantage whitening
- ✅ Mini-batch sampling from `RolloutBatch`
- ✅ `ValueHead` output shapes
- ✅ Reward function shapes, values, and edge cases
- ✅ `RunningMeanStd` convergence
- ✅ `AdaptiveKLController` behaviour
- ✅ `SyntheticPromptDataset` and `collate_fn`
- ✅ `PPOStats` arithmetic and serialisation
- ✅ Diversity metrics (`distinct_n`, `average_length`)

<br/>



## 📎 References

1. **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms*  
   https://arxiv.org/abs/1707.06347

2. **Ouyang et al. (2022)** — *Training language models to follow instructions with human feedback* (InstructGPT)  
   https://arxiv.org/abs/2203.02155

3. **Ziegler et al. (2019)** — *Fine-Tuning Language Models from Human Preferences*  
   https://arxiv.org/abs/1909.08593

4. **Schulman et al. (2016)** — *High-Dimensional Continuous Control Using Generalised Advantage Estimation*  
   https://arxiv.org/abs/1506.02438

5. **Stiennon et al. (2020)** — *Learning to summarize with human feedback*  
   https://arxiv.org/abs/2009.01325

6. **TRL Library** (HuggingFace) — practical PPO implementation reference  
   https://github.com/huggingface/trl

<br/>


<div align="center">


</div>
