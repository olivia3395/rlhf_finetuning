"""
evaluate.py — Comprehensive evaluation of a trained RLHF policy.

Usage
─────
  # Evaluate the final checkpoint
  python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final

  # Compare with the base (non-finetuned) model
  python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final --compare-base

  # Save detailed results to CSV
  python evaluate.py --checkpoint outputs/rlhf_ppo/checkpoint-final --save-csv results.csv

Metrics reported
────────────────
  • Mean / std / min / max reward
  • Mean per-component reward (sentiment, toxicity, fluency, length)
  • Approximate KL divergence from the reference model
  • Diversity: distinct-1 and distinct-2 (ratio of unique n-grams)
  • 10 qualitative sample outputs
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import positive_sentiment_config, Config
from models.policy_model import PolicyModel, ReferenceModel, load_tokenizer
from rewards.reward_functions import CompositeReward
from data.dataset import SyntheticPromptDataset, SYNTHETIC_PROMPTS

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def distinct_n(texts: List[str], n: int) -> float:
    """
    Distinct-n: fraction of unique n-grams across all generated texts.
    Higher is better (more diverse vocabulary usage).
    """
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def average_length(texts: List[str]) -> float:
    return np.mean([len(t.split()) for t in texts])


# ---------------------------------------------------------------------------
# KL estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_kl(
    policy: PolicyModel,
    ref_model: ReferenceModel,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    device: torch.device,
) -> float:
    """Estimate mean KL(π_θ ‖ π_ref) over a batch of generated responses."""
    pol_lp = policy.compute_log_probs(
        prompt_ids.to(device), prompt_mask.to(device), response_ids.to(device)
    )
    ref_lp = ref_model.compute_log_probs(
        prompt_ids.to(device), prompt_mask.to(device), response_ids.to(device)
    )
    kl = (pol_lp - ref_lp).sum(dim=1)   # sum over response tokens
    return kl.mean().item()


# ---------------------------------------------------------------------------
# Load policy from checkpoint
# ---------------------------------------------------------------------------

def load_policy_from_checkpoint(
    checkpoint_dir: str, cfg: Config, device: torch.device
) -> PolicyModel:
    from transformers import AutoModelForCausalLM

    ckpt_dir = Path(checkpoint_dir)
    policy = PolicyModel(cfg.model).to(device)

    backbone_path = ckpt_dir / "backbone"
    if backbone_path.exists():
        policy.backbone = AutoModelForCausalLM.from_pretrained(
            str(backbone_path)
        ).to(device)
        logger.info(f"Loaded backbone from {backbone_path}")
    else:
        logger.warning("No backbone checkpoint found, using pre-trained weights")

    value_head_path = ckpt_dir / "value_head.pt"
    if value_head_path.exists():
        policy.value_head.load_state_dict(
            torch.load(value_head_path, map_location=device)
        )
        logger.info(f"Loaded value head from {value_head_path}")

    return policy


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    policy: PolicyModel,
    ref_model: Optional[ReferenceModel],
    tokenizer,
    reward_fn: CompositeReward,
    cfg: Config,
    device: torch.device,
    n_samples: int = 200,
    label: str = "policy",
) -> Dict:
    """Full evaluation pass — returns a metrics dict."""
    policy.eval()

    # Build eval prompts
    dataset = SyntheticPromptDataset(tokenizer, n_samples=n_samples, seed=0)
    batch = dataset.collate_fn([dataset[i] for i in range(n_samples)])
    prompt_ids = batch["input_ids"].to(device)
    prompt_mask = batch["attention_mask"].to(device)

    # Generate
    with torch.no_grad():
        gen = policy.generate(
            prompt_ids, prompt_mask,
            tokenizer=tokenizer,
            cfg=cfg.model,
        )
    response_ids = gen["response_ids"]
    texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    # Reward
    rewards, components = reward_fn(texts, return_components=True)
    rewards_np = rewards.numpy()

    # KL
    kl = 0.0
    if ref_model is not None:
        kl = estimate_kl(
            policy, ref_model, prompt_ids, prompt_mask, response_ids, device
        )

    metrics = {
        "label": label,
        "n_samples": n_samples,
        "reward_mean": float(rewards_np.mean()),
        "reward_std": float(rewards_np.std()),
        "reward_min": float(rewards_np.min()),
        "reward_max": float(rewards_np.max()),
        "kl_from_ref": kl,
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "avg_length_words": average_length(texts),
    }
    for name, r in components.items():
        metrics[f"component_{name}"] = float(r.mean().numpy())

    metrics["sample_texts"] = texts[:10]
    return metrics


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_metrics(m: Dict):
    print(f"\n{'═'*60}")
    print(f"  Results for: {m['label']}")
    print(f"{'═'*60}")
    print(f"  Samples evaluated : {m['n_samples']}")
    print(f"  Reward (mean±std) : {m['reward_mean']:+.4f} ± {m['reward_std']:.4f}")
    print(f"  Reward [min, max] : [{m['reward_min']:+.4f}, {m['reward_max']:+.4f}]")
    print(f"  KL from reference : {m['kl_from_ref']:.4f}")
    print(f"  Distinct-1        : {m['distinct_1']:.4f}")
    print(f"  Distinct-2        : {m['distinct_2']:.4f}")
    print(f"  Avg length (words): {m['avg_length_words']:.1f}")
    for k, v in m.items():
        if k.startswith("component_"):
            name = k[len("component_"):]
            print(f"  reward/{name:<18} : {v:+.4f}")
    print(f"\n  Sample outputs:")
    for i, t in enumerate(m.get("sample_texts", [])[:5]):
        print(f"  [{i+1}] {t[:140]!r}")
    print(f"{'═'*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an RLHF-PPO checkpoint")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--compare-base", action="store_true",
        help="Also evaluate the untuned base model for comparison"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of evaluation samples"
    )
    parser.add_argument("--save-csv", default=None, help="Save results to CSV file")
    parser.add_argument("--save-json", default=None, help="Save results to JSON file")
    parser.add_argument(
        "--mode", choices=["sentiment", "detox", "balanced"], default="sentiment"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Config
    from config import positive_sentiment_config, detoxification_config, balanced_config
    preset_map = {
        "sentiment": positive_sentiment_config,
        "detox": detoxification_config,
        "balanced": balanced_config,
    }
    cfg = preset_map[args.mode]()

    # Tokenizer + reward
    tokenizer = load_tokenizer(cfg.model.model_name)
    reward_fn = CompositeReward(cfg.reward, device=str(device))
    ref_model = ReferenceModel(cfg.model.ref_model_name).to(device)

    all_results = []

    # ── Evaluate RLHF policy ─────────────────────────────────────────────
    policy = load_policy_from_checkpoint(args.checkpoint, cfg, device)
    results_policy = evaluate(
        policy, ref_model, tokenizer, reward_fn, cfg, device,
        n_samples=args.n_samples, label="rlhf_policy",
    )
    print_metrics(results_policy)
    all_results.append(results_policy)

    # ── Optionally compare with base model ───────────────────────────────
    if args.compare_base:
        base_policy = PolicyModel(cfg.model).to(device)
        results_base = evaluate(
            base_policy, ref_model, tokenizer, reward_fn, cfg, device,
            n_samples=args.n_samples, label="base_model",
        )
        print_metrics(results_base)
        all_results.append(results_base)

        # Delta
        delta_r = results_policy["reward_mean"] - results_base["reward_mean"]
        print(f"  Δ reward (RLHF - base) = {delta_r:+.4f}")

    # ── Save results ─────────────────────────────────────────────────────
    if args.save_json:
        # Remove non-serialisable tensors
        for r in all_results:
            r.pop("sample_texts", None)
        with open(args.save_json, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved → {args.save_json}")

    if args.save_csv:
        fieldnames = [k for k in all_results[0] if k != "sample_texts"]
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_results:
                row = {k: v for k, v in r.items() if k != "sample_texts"}
                writer.writerow(row)
        logger.info(f"Results saved → {args.save_csv}")


if __name__ == "__main__":
    main()
