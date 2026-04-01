"""
demo.py — Interactive demo for the trained RLHF policy.

Usage
─────
  # Interactive REPL
  python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final

  # Single prompt, non-interactive
  python demo.py --checkpoint outputs/rlhf_ppo/checkpoint-final \
                 --prompt "The movie was absolutely" \
                 --n 5

  # Run without a checkpoint (uses the untuned GPT-2 base model)
  python demo.py --no-checkpoint

Features
────────
  • Generate N continuations for a given prompt
  • Show per-generation reward breakdown
  • Compare side-by-side with the base (untuned) model
  • Colour-coded reward display (green = high, red = low)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import positive_sentiment_config
from models.policy_model import PolicyModel, load_tokenizer
from rewards.reward_functions import CompositeReward


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

RESET = "\033[0m"
BOLD  = "\033[1m"
RED   = "\033[31m"
GRN   = "\033[32m"
YEL   = "\033[33m"
CYN   = "\033[36m"
GRY   = "\033[90m"


def colour_reward(r: float) -> str:
    if r > 0.5:
        return f"{GRN}{r:+.3f}{RESET}"
    elif r > 0.0:
        return f"{YEL}{r:+.3f}{RESET}"
    else:
        return f"{RED}{r:+.3f}{RESET}"


# ---------------------------------------------------------------------------
# Generation helper
# ---------------------------------------------------------------------------

def generate(
    policy: PolicyModel,
    tokenizer,
    prompt: str,
    cfg,
    device: torch.device,
    n: int = 5,
) -> List[str]:
    enc = tokenizer(
        [prompt] * n,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        gen = policy.generate(input_ids, attention_mask, tokenizer, cfg)

    texts = tokenizer.batch_decode(gen["response_ids"], skip_special_tokens=True)
    return texts


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_results(
    prompt: str,
    texts: List[str],
    reward_fn: CompositeReward,
    label: str = "RLHF Policy",
):
    print(f"\n{BOLD}{CYN}{'─'*60}{RESET}")
    print(f"  {BOLD}{label}{RESET}")
    print(f"  Prompt: {GRY}{prompt!r}{RESET}")
    print(f"{CYN}{'─'*60}{RESET}\n")

    details = reward_fn.describe(texts)
    for i, (text, detail) in enumerate(zip(texts, details)):
        total = detail["total"]
        print(f"  {BOLD}[{i+1}]{RESET} {colour_reward(total)}  {text[:120]!r}")
        parts = "  ".join(
            f"{k}={colour_reward(v)}"
            for k, v in detail.items()
            if k != "total"
        )
        print(f"        {GRY}↳ {parts}{RESET}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive RLHF-PPO demo")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Run with the base model (no RLHF)")
    parser.add_argument("--prompt", default=None,
                        help="Prompt text (non-interactive mode)")
    parser.add_argument("-n", "--n-samples", type=int, default=5,
                        help="Number of continuations to generate")
    parser.add_argument("--compare", action="store_true",
                        help="Also show base model outputs for comparison")
    parser.add_argument("--mode", choices=["sentiment", "detox", "balanced"],
                        default="sentiment")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config + models
    from config import positive_sentiment_config, detoxification_config, balanced_config
    cfg = {"sentiment": positive_sentiment_config,
           "detox": detoxification_config,
           "balanced": balanced_config}[args.mode]()

    tokenizer = load_tokenizer(cfg.model.model_name)
    reward_fn = CompositeReward(cfg.reward, device=str(device))

    # Load RLHF policy
    if args.no_checkpoint or args.checkpoint is None:
        print(f"{YEL}⚠  No checkpoint supplied — using untuned base model{RESET}")
        policy = PolicyModel(cfg.model).to(device)
    else:
        from evaluate import load_policy_from_checkpoint
        policy = load_policy_from_checkpoint(args.checkpoint, cfg, device)

    policy.eval()

    # Optionally load base model for comparison
    base_policy = None
    if args.compare:
        print(f"{GRY}Loading base model for comparison …{RESET}")
        base_policy = PolicyModel(cfg.model).to(device)
        base_policy.eval()

    print(f"\n{BOLD}{'═'*60}")
    print(f"  RLHF-PPO Interactive Demo")
    print(f"  Mode: {args.mode}   |   Model: {cfg.model.model_name}")
    print(f"{'═'*60}{RESET}")
    print(f"  Type a prompt and press Enter to generate continuations.")
    print(f"  Commands: 'quit' | 'q' to exit, 'clear' to reset.")
    print()

    # ── Non-interactive mode ─────────────────────────────────────────────
    if args.prompt:
        texts = generate(policy, tokenizer, args.prompt, cfg.model, device, args.n_samples)
        display_results(args.prompt, texts, reward_fn, label="RLHF Policy")
        if base_policy:
            base_texts = generate(
                base_policy, tokenizer, args.prompt, cfg.model, device, args.n_samples
            )
            display_results(args.prompt, base_texts, reward_fn, label="Base Model")
        return

    # ── Interactive REPL ─────────────────────────────────────────────────
    while True:
        try:
            prompt = input(f"{BOLD}{CYN}Prompt >{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue
        if prompt.lower() in {"quit", "q", "exit"}:
            print("Bye!")
            break
        if prompt.lower() == "clear":
            print("\033[2J\033[H")
            continue

        texts = generate(policy, tokenizer, prompt, cfg.model, device, args.n_samples)
        display_results(prompt, texts, reward_fn, label="RLHF Policy")

        if base_policy:
            base_texts = generate(
                base_policy, tokenizer, prompt, cfg.model, device, args.n_samples
            )
            display_results(prompt, base_texts, reward_fn, label="Base Model (no RLHF)")


if __name__ == "__main__":
    main()
