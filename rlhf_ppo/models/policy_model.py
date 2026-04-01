"""
models/policy_model.py — Actor-Critic policy model for RLHF-PPO.

Architecture
------------
  ┌──────────────────────────────────────────────┐
  │  Pre-trained Causal LM  (e.g. GPT-2)         │
  │  ┌─────────────────────────────────────────┐ │
  │  │  Transformer backbone  (shared weights) │ │
  │  └──────────────────────┬──────────────────┘ │
  │                         │ hidden states        │
  │          ┌──────────────┴──────────────┐      │
  │          ▼                             ▼      │
  │   LM head (actor)           Value head        │
  │   → token logits            → scalar V(s)     │
  └──────────────────────────────────────────────┘

The value head is a small 2-layer MLP on top of the last hidden state of the
*last generated token* (following the convention in TRL / InstructGPT).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from typing import Optional, Tuple, Dict, Any

from config import ModelConfig


class ValueHead(nn.Module):
    """Small MLP that maps a hidden state to a scalar value estimate."""

    def __init__(self, hidden_size: int, inner_size: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_size, 1),
        )
        # Initialise output layer near-zero so early training is stable
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (batch, hidden_size)  — last-token hidden state
        Returns:
            value: (batch,)
        """
        return self.net(hidden).squeeze(-1)


class PolicyModel(nn.Module):
    """
    Actor-Critic wrapper around a pre-trained causal LM.

    The backbone's weights are updated during PPO; the value head
    is trained from scratch.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # ── backbone ────────────────────────────────────────────────────────
        self.backbone: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            output_hidden_states=True,
        )
        hidden_size = self.backbone.config.hidden_size

        # ── value head ──────────────────────────────────────────────────────
        self.value_head = ValueHead(
            hidden_size=hidden_size,
            inner_size=cfg.value_head_hidden_size,
            dropout=cfg.value_head_dropout,
        )

    # ------------------------------------------------------------------
    # Forwarding helpers
    # ------------------------------------------------------------------

    def _last_token_hidden(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the hidden state of the last *non-padding* token.

        hidden_states : tuple of (batch, seq, hidden) across layers
        attention_mask: (batch, seq) with 1 for real tokens, 0 for padding
        """
        last_layer: torch.Tensor = hidden_states[-1]          # (B, T, H)
        # Index of the last real token per sample
        lengths = attention_mask.sum(dim=1) - 1               # (B,)
        batch_idx = torch.arange(last_layer.size(0), device=last_layer.device)
        return last_layer[batch_idx, lengths]                  # (B, H)

    # ------------------------------------------------------------------
    # Core forward pass (used during PPO update)
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full actor-critic forward pass over a *complete* sequence
        (prompt + response concatenated).

        Returns dict with:
            logits  : (B, T, V)   token logits from the LM head
            values  : (B,)        scalar value estimate per sample
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        logits: torch.Tensor = outputs.logits                  # (B, T, V)
        hidden = self._last_token_hidden(
            outputs.hidden_states, attention_mask
        )                                                       # (B, H)
        values = self.value_head(hidden)                        # (B,)
        return {"logits": logits, "values": values}

    # ------------------------------------------------------------------
    # Generation (used during rollout)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        cfg: ModelConfig,
    ) -> Dict[str, torch.Tensor]:
        """
        Auto-regressive generation that also returns per-token log-probs
        and value estimates for each generated step.

        Returns dict:
            response_ids        : (B, T_gen)
            response_log_probs  : (B, T_gen)   log π(a|s) for each token
            values              : (B,)          V(s) at last prompt token
            full_ids            : (B, T_prompt + T_gen)
            full_mask           : (B, T_prompt + T_gen)
        """
        B = input_ids.size(0)
        device = input_ids.device

        # Generate response tokens
        gen_out = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            min_new_tokens=cfg.min_new_tokens,
            do_sample=cfg.do_sample,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            temperature=cfg.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,         # per-step logit tensors
        )

        # gen_out.sequences : (B, T_prompt + T_gen)
        full_ids = gen_out.sequences
        T_prompt = input_ids.size(1)
        response_ids = full_ids[:, T_prompt:]                  # (B, T_gen)
        T_gen = response_ids.size(1)

        # Build attention mask for full sequence
        full_mask = torch.ones_like(full_ids, dtype=torch.long)
        # Preserve original padding in prompt
        full_mask[:, :T_prompt] = attention_mask

        # ── per-token log-probs of the *taken* action ────────────────────
        # gen_out.scores is a tuple of T_gen tensors, each (B, V)
        stacked_scores = torch.stack(gen_out.scores, dim=1)    # (B, T_gen, V)
        log_probs_all = torch.log_softmax(stacked_scores, dim=-1)
        # Gather log-prob of the actually selected token
        response_log_probs = log_probs_all.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)                                           # (B, T_gen)

        # ── value estimate at the end of the prompt ─────────────────────
        with torch.enable_grad():
            fwd = self.forward(input_ids, attention_mask)
        values = fwd["values"].detach()                        # (B,)

        return {
            "response_ids": response_ids,
            "response_log_probs": response_log_probs,
            "values": values,
            "full_ids": full_ids,
            "full_mask": full_mask,
        }

    # ------------------------------------------------------------------
    # Log-prob computation on an existing sequence
    # ------------------------------------------------------------------

    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Re-compute log π(response | prompt) under the *current* policy.

        Used inside the PPO update to get the updated log-probs μ_θ.

        Returns:
            log_probs: (B, T_gen)
        """
        T_prompt = input_ids.size(1)
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.ones_like(full_ids, dtype=torch.long)
        full_mask[:, :T_prompt] = attention_mask

        outputs = self.backbone(
            input_ids=full_ids,
            attention_mask=full_mask,
            output_hidden_states=True,
        )
        # Shift so that token i predicts token i+1
        logits = outputs.logits[:, T_prompt - 1 : -1, :]      # (B, T_gen, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        gathered = log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)                                          # (B, T_gen)
        return gathered

    def compute_values(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return V(s) for a batch of (prompt) sequences — (B,)."""
        fwd = self.forward(input_ids, attention_mask)
        return fwd["values"]


# ---------------------------------------------------------------------------
# Reference model (frozen copy of the initial policy)
# ---------------------------------------------------------------------------

class ReferenceModel(nn.Module):
    """
    Frozen copy of the initial policy used to compute KL divergence:

        KL(π_θ ‖ π_ref) = log π_θ(a|s) - log π_ref(a|s)

    Weights are *never* updated.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        # Freeze all parameters
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def compute_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log π_ref(response | prompt).

        Returns: (B, T_gen)
        """
        T_prompt = input_ids.size(1)
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        full_mask = torch.ones_like(full_ids, dtype=torch.long)
        full_mask[:, :T_prompt] = attention_mask

        outputs = self.backbone(
            input_ids=full_ids,
            attention_mask=full_mask,
        )
        logits = outputs.logits[:, T_prompt - 1 : -1, :]      # (B, T_gen, V)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.gather(
            2, response_ids.unsqueeze(-1)
        ).squeeze(-1)                                          # (B, T_gen)


# ---------------------------------------------------------------------------
# Tokenizer factory
# ---------------------------------------------------------------------------

def load_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"   # For causal LMs, left-pad prompts
    return tok
