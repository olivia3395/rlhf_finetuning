"""
rewards/reward_functions.py — Custom reward functions for controllable generation.

Reward design
─────────────
We decompose the per-sample scalar reward r(s, a) into four interpretable
components, each targeting a distinct generation quality:

  r_total = w_sent · r_sentiment
          + w_tox  · r_toxicity          (penalty, already negative)
          + w_flu  · r_fluency
          + w_len  · r_length

All component values are normalised to roughly [-1, +1] before weighting.
The composite reward can optionally be whitened (mean/std normalisation)
over each rollout batch to stabilise training.

Reward components
─────────────────
  SentimentReward   – targets a desired polarity label using a classifier
  ToxicityReward    – penalises toxic text detected by a lightweight model
  FluencyReward     – rewards natural language via a frozen LM's perplexity
  LengthReward      – soft penalty for deviating from the ideal length
  CompositeReward   – weighted sum + optional whitening
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from config import RewardConfig


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseReward(ABC):
    """All reward functions share this interface."""

    @abstractmethod
    def __call__(self, texts: List[str]) -> torch.Tensor:
        """
        Args:
            texts: list of N generated strings
        Returns:
            rewards: float tensor of shape (N,)
        """

    def to(self, device):
        return self


# ---------------------------------------------------------------------------
# 1. Sentiment reward
# ---------------------------------------------------------------------------

class SentimentReward(BaseReward):
    """
    Uses a pre-trained sentiment classifier to reward outputs that match
    the desired polarity.

    Score:
        +1  if the top predicted label == target_label
        -1  otherwise

    Continuous variant: maps the probability of the target class to [-1, +1].
    """

    # Lightweight distilled model — fast inference on CPU
    DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(
        self,
        target_label: str = "POSITIVE",
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        continuous: bool = True,
    ):
        self.target_label = target_label.upper()
        self.continuous = continuous
        self.device = device

        self._pipe = pipeline(
            "text-classification",
            model=model_name,
            device=0 if device == "cuda" else -1,
            truncation=True,
            max_length=512,
            top_k=None,       # return all labels
        )

    def __call__(self, texts: List[str]) -> torch.Tensor:
        results = self._pipe(texts, batch_size=32)
        rewards = []
        for result in results:
            # result: list of {"label": str, "score": float}
            label_scores = {r["label"].upper(): r["score"] for r in result}
            if self.continuous:
                # Map P(target) to [-1, +1]
                p_target = label_scores.get(self.target_label, 0.5)
                reward = 2 * p_target - 1.0
            else:
                top_label = max(label_scores, key=label_scores.get)
                reward = 1.0 if top_label == self.target_label else -1.0
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 2. Toxicity reward (penalty)
# ---------------------------------------------------------------------------

class ToxicityReward(BaseReward):
    """
    Penalises text that a toxicity classifier flags as harmful.

    Score:
        r = -(toxicity_score)   →  in [-1, 0]

    Clamps at -1 for robustness.
    """

    DEFAULT_MODEL = "unitary/toxic-bert"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "cpu",
        max_toxicity: float = 0.5,
    ):
        """
        max_toxicity: scores above this get a penalty of -1.
        """
        self.max_toxicity = max_toxicity
        self.device = device

        try:
            self._pipe = pipeline(
                "text-classification",
                model=model_name,
                device=0 if device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
        except Exception:
            # Fall back to a keyword-based heuristic if model unavailable
            self._pipe = None
            self._toxic_words = {
                "hate", "kill", "die", "stupid", "idiot", "moron",
                "damn", "hell", "crap", "awful", "terrible", "worst",
            }

    def _heuristic(self, text: str) -> float:
        words = set(text.lower().split())
        hits = words & self._toxic_words
        return min(len(hits) * 0.15, 1.0)

    def __call__(self, texts: List[str]) -> torch.Tensor:
        if self._pipe is None:
            scores = [self._heuristic(t) for t in texts]
        else:
            results = self._pipe(texts, batch_size=32)
            scores = []
            for r in results:
                label = r["label"].upper()
                score = r["score"]
                # Model predicts "TOXIC" label
                tox_score = score if "TOXIC" in label else 1 - score
                scores.append(tox_score)

        # Normalise and negate
        rewards = [
            -min(s / self.max_toxicity, 1.0)
            for s in scores
        ]
        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. Fluency reward
# ---------------------------------------------------------------------------

class FluencyReward(BaseReward):
    """
    Rewards natural, fluent text using a frozen reference LM's perplexity.

    r = clip(-log PPL / log_ppl_max, -1, 0) + 1    →   in [0, 1]

    Lower PPL → higher fluency → higher reward.
    """

    DEFAULT_MODEL = "gpt2"
    LOG_PPL_MAX = 6.0   # log(403) ≈ 6 → very surprised reference model

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu"):
        self.device = device
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self._model.eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        for p in self._model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, texts: List[str]) -> torch.Tensor:
        rewards = []
        for text in texts:
            if not text.strip():
                rewards.append(0.0)
                continue
            enc = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)
            ids = enc["input_ids"]
            if ids.size(1) < 2:
                rewards.append(0.0)
                continue
            output = self._model(ids, labels=ids)
            log_ppl = output.loss.item()  # cross-entropy = log PPL per token
            # Map to [0, 1]: low log_ppl → reward ≈ 1
            reward = max(0.0, 1.0 - log_ppl / self.LOG_PPL_MAX)
            rewards.append(reward)
        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Length reward
# ---------------------------------------------------------------------------

class LengthReward(BaseReward):
    """
    Soft Gaussian penalty around an ideal token-count target.

    r = exp(-0.5 · ((len - ideal) / sigma)^2)   →   in (0, 1]
    Then rescaled to [-1, +1]: r = 2 * raw - 1

    A response of exactly `ideal` tokens gets +1; very short / very long
    responses approach -1.
    """

    def __init__(self, ideal_length: int = 48, sigma: float = 16.0):
        self.ideal = ideal_length
        self.sigma = sigma

    def __call__(self, texts: List[str]) -> torch.Tensor:
        rewards = []
        for text in texts:
            n = len(text.split())
            raw = math.exp(-0.5 * ((n - self.ideal) / self.sigma) ** 2)
            rewards.append(2 * raw - 1.0)
        return torch.tensor(rewards, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 5. Composite reward (the one used by the RLHF trainer)
# ---------------------------------------------------------------------------

class RunningMeanStd:
    """Welford online algorithm for reward normalisation."""

    def __init__(self, momentum: float = 0.01, epsilon: float = 1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0
        self.momentum = momentum
        self.epsilon = epsilon

    def update(self, values: np.ndarray):
        batch_mean = values.mean()
        batch_var = values.var()
        self.mean = (1 - self.momentum) * self.mean + self.momentum * batch_mean
        self.var = (1 - self.momentum) * self.var + self.momentum * batch_var

    def normalise(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / (math.sqrt(self.var) + self.epsilon)


class CompositeReward(BaseReward):
    """
    Weighted combination of the four reward components.

    Usage:
        reward_fn = CompositeReward(cfg.reward)
        rewards = reward_fn(generated_texts)   # (N,)
    """

    def __init__(self, cfg: RewardConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device

        # ── instantiate components ──────────────────────────────────────
        self.components: Dict[str, BaseReward] = {}

        if cfg.sentiment_weight > 0 and cfg.target_sentiment:
            self.components["sentiment"] = SentimentReward(
                target_label=cfg.target_sentiment,
                device=device,
            )

        if cfg.toxicity_weight > 0:
            self.components["toxicity"] = ToxicityReward(
                device=device,
                max_toxicity=cfg.max_toxicity,
            )

        if cfg.fluency_weight > 0:
            self.components["fluency"] = FluencyReward(device=device)

        if cfg.length_weight > 0:
            self.components["length"] = LengthReward(
                ideal_length=cfg.ideal_length,
            )

        self.weights = {
            "sentiment": cfg.sentiment_weight,
            "toxicity": cfg.toxicity_weight,
            "fluency": cfg.fluency_weight,
            "length": cfg.length_weight,
        }

        # Reward normalisation
        self._rms = RunningMeanStd(momentum=cfg.reward_norm_momentum)

    # ------------------------------------------------------------------

    def __call__(
        self,
        texts: List[str],
        return_components: bool = False,
    ) -> Any:
        """
        Compute the composite reward.

        Args:
            texts: list of N generated strings
            return_components: if True, also return dict of component tensors

        Returns:
            rewards: (N,) float tensor
            components (optional): dict of component name → (N,) tensor
        """
        component_rewards: Dict[str, torch.Tensor] = {}
        total = torch.zeros(len(texts))

        for name, fn in self.components.items():
            r = fn(texts).cpu()
            component_rewards[name] = r
            total += self.weights[name] * r

        # Optional whitening across the rollout batch
        if self.cfg.normalize_rewards:
            arr = total.numpy()
            self._rms.update(arr)
            arr = self._rms.normalise(arr)
            total = torch.from_numpy(arr).float()

        if return_components:
            return total, component_rewards
        return total

    def describe(self, texts: List[str]) -> List[Dict[str, float]]:
        """Return per-sample reward breakdown as a list of dicts."""
        total, components = self(texts, return_components=True)
        results = []
        for i in range(len(texts)):
            d: Dict[str, float] = {"total": total[i].item()}
            for name, r in components.items():
                d[name] = r[i].item()
            results.append(d)
        return results
