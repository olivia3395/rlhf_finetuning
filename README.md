



# RLHF-Custom: Reinforcement Learning from Human Feedback on Custom Text Dataset

This repository demonstrates a **method-driven pipeline** for aligning a pretrained language model with human-defined preferences through **Reinforcement Learning from Human Feedback (RLHF)**.  
Unlike standard fine-tuning, this approach uses an explicit **reward model or heuristic reward function** to iteratively improve model behavior based on qualitative criteria.

---

## Motivation

Modern large language models (LLMs) often generate fluent but misaligned text—responses that are verbose, off-topic, or stylistically inconsistent with user preferences.  
To address this, **RLHF** offers a principled framework to incorporate human-like evaluation signals into model training.

This project provides a minimal yet extensible implementation for:
- Applying **RLHF on a custom dataset** of prompts and responses;
- Integrating **custom reward functions** (semantic, stylistic, or rule-based);
- Supporting **any Hugging Face causal language model** (e.g., GPT-2, StarCoder, Falcon).

---

## Methodology Overview

### 1. **Supervised Initialization**
We start from a pretrained model (`AutoModelForCausalLM`) such as `distilgpt2`.  
A small dataset of prompt–response pairs is loaded from a local `.parquet` file or defined manually.

### 2. **Reward Function Design**
A reward function quantifies response quality.  
This can be:
- **Heuristic-based:** e.g., brevity, politeness, factuality.
- **Model-based:** e.g., a reward model trained for preference comparison.

```python
def compute_reward(query, response):
    # Example: penalize long answers
    return -len(response.split())
````

### 3. **PPO Fine-Tuning**

We apply **Proximal Policy Optimization (PPO)** via the [TRL library](https://github.com/huggingface/trl) to adjust model outputs toward higher reward responses.

```python
from trl import PPOTrainer, PPOConfig
config = PPOConfig(model_name="distilgpt2", learning_rate=1e-5, batch_size=2)
ppo_trainer = PPOTrainer(config=config, model=model, tokenizer=tokenizer, dataset=dataset)
```

At each iteration:

1. The model generates a response.
2. The reward function scores it.
3. PPO updates model parameters to increase expected reward while maintaining stability via KL regularization.

### 4. **Evaluation & Sampling**

After training, the model is tested on new prompts to examine alignment improvements.

---

## Implementation Highlights

| Component              | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| **Model Backbone**     | `distilgpt2` (lightweight GPT-2 variant; CPU-friendly) |
| **Trainer**            | `PPOTrainer` from TRL (Proximal Policy Optimization)   |
| **Reward Mechanism**   | Custom function or learned reward model                |
| **Training Objective** | Maximize reward – β × KL divergence                    |

---

## Setup

```bash
pip install transformers trl datasets accelerate pandas
```

> To run on GPU:
> Go to *Runtime → Change runtime type → GPU* 


---

## Mathematical Formulation

Given model parameters $ \theta $, reward $ r(x, y) $, and baseline policy $ \pi_{\text{ref}}$:

$$
\max_\theta \mathbb{E}_{x \sim D, y \sim \pi_\theta}
\left[ r(x, y) - \beta \, \mathrm{KL}(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)) \right]
$$

- $ r(x, y) $: reward function defined by user  
- $ \beta $: regularization coefficient balancing alignment and stability


## Example Use Cases

* Fine-tuning a conversational model for concise and polite answers
* Adapting a code model (e.g., StarCoder) for cleaner, PEP8-compliant code generation
* Reinforcement-based summarization optimization
* Domain-specific response alignment (e.g., clinical dialogue, financial Q&A)

---

## References


* Hugging Face TRL Documentation: [https://huggingface.co/docs/trl](https://huggingface.co/docs/trl)


---


