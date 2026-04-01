"""
data/dataset.py — Dataset utilities for RLHF-PPO prompt sampling.

We treat training as a prompt-continuation task:
  • Sample text snippets from a dataset (e.g. IMDb reviews, OpenWebText)
  • Truncate to max_prompt_length tokens
  • The policy then generates a continuation

The reward function evaluates only the *generated* continuation, not the
prompt.  This mirrors the InstructGPT / TRL setup.

Supported datasets (auto-downloaded from HuggingFace Hub):
  • "imdb"       — movie reviews; naturally has positive/negative labels
  • "sst2"       — sentiment classification
  • "openwebtext"— general web text (for detoxification tasks)
  • Any other HF text dataset with a "text" or "review" column
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Prompt Dataset
# ---------------------------------------------------------------------------

class PromptDataset(Dataset):
    """
    Wraps a HuggingFace dataset and tokenises text into prompt tensors.

    Each sample is a truncated prefix of the source text.  During training
    the model is asked to continue from this prefix; the continuation is
    scored by the reward function.
    """

    TEXT_KEYS = ("text", "review", "sentence", "content", "article")

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_prompt_length: int = 64,
        min_prompt_length: int = 16,
        seed: int = 42,
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.min_prompt_length = min_prompt_length
        self.rng = random.Random(seed)

        # ── Load dataset ─────────────────────────────────────────────────
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Could not load dataset '{dataset_name}'. "
                f"Make sure it exists on HuggingFace Hub.\n{e}"
            )

        # Find the text column
        text_key = None
        for key in self.TEXT_KEYS:
            if key in ds.column_names:
                text_key = key
                break
        if text_key is None:
            raise ValueError(
                f"No text column found in dataset '{dataset_name}'. "
                f"Expected one of {self.TEXT_KEYS}. "
                f"Got: {ds.column_names}"
            )

        texts = ds[text_key]
        if max_samples:
            texts = texts[:max_samples]

        # ── Tokenise and filter ───────────────────────────────────────────
        self.prompts: List[torch.Tensor] = []
        for text in texts:
            tokens = tokenizer.encode(
                text, add_special_tokens=True, truncation=False
            )
            if len(tokens) < min_prompt_length + 4:
                continue
            # Random prefix length for variety
            max_start = max(
                min_prompt_length,
                min(len(tokens) - 4, max_prompt_length),
            )
            if max_start < min_prompt_length:
                continue
            prompt_len = self.rng.randint(min_prompt_length, max_start)
            prompt_ids = torch.tensor(tokens[:prompt_len], dtype=torch.long)
            self.prompts.append(prompt_ids)

        if len(self.prompts) == 0:
            raise RuntimeError(
                "No valid prompts found after filtering. "
                "Try a larger dataset or smaller min_prompt_length."
            )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.prompts[idx]

    # ------------------------------------------------------------------
    # Collation
    # ------------------------------------------------------------------

    def collate_fn(self, batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Left-pad sequences to the same length (causal LMs generate from the
        right, so left-padding keeps the prompt prefix aligned at the end).
        """
        max_len = max(t.size(0) for t in batch)
        pad_id = self.tokenizer.pad_token_id or 0

        input_ids = torch.full(
            (len(batch), max_len), pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

        for i, t in enumerate(batch):
            offset = max_len - t.size(0)
            input_ids[i, offset:] = t
            attention_mask[i, offset:] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # ------------------------------------------------------------------
    # Evaluation prompt helper
    # ------------------------------------------------------------------

    def get_eval_prompts(self, n: int) -> Dict[str, torch.Tensor]:
        """Return a fixed subset of n prompts for evaluation."""
        indices = list(range(min(n, len(self.prompts))))
        batch = [self.prompts[i] for i in indices]
        return self.collate_fn(batch)


# ---------------------------------------------------------------------------
# Synthetic fallback (for unit-testing without an internet connection)
# ---------------------------------------------------------------------------

SYNTHETIC_PROMPTS = [
    "The movie was absolutely",
    "I couldn't believe how",
    "This restaurant serves the most",
    "The weather today is",
    "Scientists have recently discovered",
    "In a surprising turn of events,",
    "The new album by the band is",
    "After reading the book, I felt",
    "The team's performance was",
    "Experts say that the economy",
]


class SyntheticPromptDataset(Dataset):
    """Minimal dataset that works offline, useful for unit tests."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        n_samples: int = 200,
        seed: int = 42,
    ):
        rng = random.Random(seed)
        self.tokenizer = tokenizer
        self.prompts: List[torch.Tensor] = []
        for _ in range(n_samples):
            text = rng.choice(SYNTHETIC_PROMPTS)
            ids = tokenizer.encode(text, add_special_tokens=True)
            self.prompts.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

    def collate_fn(self, batch):
        max_len = max(t.size(0) for t in batch)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, t in enumerate(batch):
            offset = max_len - t.size(0)
            input_ids[i, offset:] = t
            attention_mask[i, offset:] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def get_eval_prompts(self, n: int):
        batch = [self.prompts[i] for i in range(min(n, len(self.prompts)))]
        return self.collate_fn(batch)
