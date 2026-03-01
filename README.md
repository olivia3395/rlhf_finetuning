
# RLHF-Custom: RLHF on a Custom Text Dataset (Reward-Driven Alignment)

We implement a **clean, method-first RLHF pipeline** for aligning a pretrained causal LM to **human-defined preferences** using an explicit reward signal—either a **lightweight heuristic** (rules, style constraints) or a **learned reward model**.  
Compared to standard supervised fine-tuning, RLHF optimizes *behavior* directly by iteratively reinforcing responses that score higher under your preference metric.


## Why RLHF?

LLMs can be fluent yet misaligned: overly long, inconsistent in tone, or ignoring formatting rules. RLHF provides a practical loop:

- **Define what “good” means** (reward)
- **Sample outputs from the model**
- **Update the model to increase expected reward**
- **Keep it stable** via KL-regularization against a reference policy

This project is meant to stay minimal but extensible:
- works with **any Hugging Face causal LM** (`AutoModelForCausalLM`)
- supports **custom datasets** (e.g., `.parquet`)
- supports **reward functions** that are heuristic, learned, or hybrid
- uses **PPO** from Hugging Face **TRL**


## Pipeline Overview (End-to-End)

### 1) Supervised Initialization (Optional but recommended)
Start from a pretrained model (e.g., `distilgpt2`) and optionally run a short **SFT warm-start** on prompt–response pairs to reduce exploration cost.

**Data format (typical):**
- `prompt`: user query / instruction
- `response`: preferred response (for SFT), or empty if PPO-only




### 2) Reward Design (Heuristic, Model-Based, or Hybrid)

The reward function is the core of alignment. You can start simple and iterate:
- **Style/format rewards** (length, politeness markers, Markdown structure, banned phrases)
- **Semantic rewards** (topic relevance, entailment, embedding similarity)
- **Safety & constraint rewards** (PII avoidance, refusal compliance)
- **Learned reward model** (pairwise preferences: “A better than B”)

A **hybrid** reward often works best: combine multiple signals with weights.

```python
import re

def compute_reward(prompt: str, response: str) -> float:
    # Example: encourage concise + structured answers
    length_penalty = -0.01 * len(response.split())
    has_bullets = 0.2 if re.search(r"^\s*[-*]\s+", response, flags=re.M) else 0.0
    no_wall_of_text = 0.2 if response.count("\n") >= 2 else -0.1
    return length_penalty + has_bullets + no_wall_of_text
````

**Practical tips**

* Normalize rewards to a stable range (e.g., `[-1, 1]`)
* Avoid sparse rewards (all zeros) early on
* Log each reward component so you can debug what the policy is learning


### 3) PPO Fine-Tuning (TRL)

We use **Proximal Policy Optimization (PPO)** to update the policy while controlling drift from a **reference model** (KL penalty). At each PPO step:

1. Sample a response `y ~ πθ(·|x)`
2. Score it with reward `r(x, y)`
3. Update θ to increase reward while discouraging large distribution shifts

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

config = PPOConfig(
    model_name=model_name,
    learning_rate=1e-5,
    batch_size=2,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
)

# dataset should yield dicts like {"query": "..."} or {"prompt": "..."}
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
)
```

**Common knobs you’ll likely tune**

* `β` / KL coefficient (or adaptive KL): too high → no learning; too low → collapse/drift
* generation params (max_new_tokens, temperature, top_p): affects exploration
* reward scaling/clipping: stabilizes training


### 4) Evaluation & Sampling

After PPO, evaluation should be both:

* **automatic** (reward, length stats, rule violation rate)
* **human** (side-by-side comparison on a fixed prompt set)

Recommended eval artifacts:

* a small `eval_prompts.jsonl`
* before/after samples saved in `outputs/`
* summary metrics: mean reward, KL, length, diversity, refusal rate (if relevant)



## Implementation Highlights

| Component      | What it does                                                   |
| -------------- | -------------------------------------------------------------- |
| **Backbone**   | Any HF causal LM (default: `distilgpt2` for CPU-friendly runs) |
| **RL Trainer** | TRL `PPOTrainer`                                               |
| **Reward**     | User-defined heuristic, learned RM, or a weighted combination  |
| **Stability**  | KL regularization against a frozen reference policy            |
| **Data**       | Local `.parquet` or `datasets.Dataset`                         |


## Setup

```bash
pip install transformers trl datasets accelerate pandas
```

**GPU tip (Colab):** `Runtime → Change runtime type → GPU`



## Mathematical Objective (KL-Regularized RL)

Given prompts $x \sim D$, model policy $\pi_\theta(y \mid x)$, reference policy $\pi_{\mathrm{ref}}(y \mid x)$, and reward $r(x,y)$, PPO approximately optimizes:

$$
\max_{\theta};\mathbb{E}*{x \sim D,; y \sim \pi*{\theta}(\cdot \mid x)}
\Big[
r(x,y) - \beta
\mathrm{KL}\left(\pi_{\theta}(\cdot \mid x),|,\pi_{\mathrm{ref}}(\cdot \mid x)\right)
\Big].
$$

Intuition:

* maximize responses that score higher under your preference signal
* penalize drifting too far from the reference distribution



## Example Use Cases

* Train a chat model to be **concise, structured, and polite**
* Align a code model (e.g., StarCoder) toward **PEP8 + lint-clean outputs**
* RL-based summarization: optimize **coverage vs. brevity**
* Domain alignment (finance/clinical): enforce **terminology + refusal rules**


## References

* Hugging Face TRL docs: [https://huggingface.co/docs/trl](https://huggingface.co/docs/trl)
* TRL GitHub: [https://github.com/huggingface/trl](https://github.com/huggingface/trl)


