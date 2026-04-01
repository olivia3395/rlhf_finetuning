"""
tests/test_all.py — Unit tests for all RLHF-PPO components.

Run with:
    python -m pytest tests/ -v
or directly:
    python tests/test_all.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import torch
import torch.nn as nn
import numpy as np

from config import Config, PPOConfig, ModelConfig, RewardConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = Config()
        self.assertEqual(cfg.model.model_name, "gpt2")
        self.assertEqual(cfg.model.ref_model_name, "gpt2")  # auto-filled
        self.assertIsInstance(cfg.ppo.clip_epsilon, float)

    def test_preset_configs(self):
        from config import (
            positive_sentiment_config,
            detoxification_config,
            balanced_config,
        )
        cfg_s = positive_sentiment_config()
        self.assertEqual(cfg_s.reward.target_sentiment, "POSITIVE")

        cfg_d = detoxification_config()
        self.assertIsNone(cfg_d.reward.target_sentiment)
        self.assertGreater(cfg_d.reward.toxicity_weight, 1.0)

        cfg_b = balanced_config()
        self.assertGreater(cfg_b.reward.length_weight, 0)

    def test_summary(self):
        cfg = Config()
        s = cfg.summary()
        self.assertIn("MODEL", s)
        self.assertIn("PPO", s)
        self.assertIn("REWARD", s)


# ---------------------------------------------------------------------------
# Rollout buffer / GAE tests
# ---------------------------------------------------------------------------

class TestRolloutBuffer(unittest.TestCase):

    def test_gae_single_step(self):
        from training.rollout_buffer import compute_gae
        rewards = torch.tensor([1.0, -1.0, 0.5])
        values = torch.tensor([0.3, 0.7, 0.2])
        adv, ret = compute_gae(rewards, values)
        # With no next_values: adv = rewards - values
        expected_adv = rewards - values
        self.assertTrue(torch.allclose(adv, expected_adv, atol=1e-5))
        expected_ret = adv + values
        self.assertTrue(torch.allclose(ret, expected_ret, atol=1e-5))

    def test_gae_sequence(self):
        from training.rollout_buffer import compute_gae_sequence
        B, T = 2, 4
        rewards = torch.zeros(B, T)
        rewards[:, -1] = 1.0  # terminal reward only
        values = torch.zeros(B, T)
        adv, ret = compute_gae_sequence(rewards, values)
        self.assertEqual(adv.shape, (B, T))
        self.assertEqual(ret.shape, (B, T))
        # With all zeros values and reward only at end, last adv should be 1
        self.assertAlmostEqual(adv[0, -1].item(), 1.0, places=4)

    def test_whiten(self):
        from training.rollout_buffer import whiten
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        w = whiten(t)
        self.assertAlmostEqual(w.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(w.std().item(), 1.0, places=3)

    def test_mini_batches(self):
        from training.rollout_buffer import RolloutBatch
        B, T_p, T_r = 16, 10, 20
        batch = RolloutBatch(
            prompt_ids=torch.zeros(B, T_p, dtype=torch.long),
            prompt_mask=torch.ones(B, T_p, dtype=torch.long),
            response_ids=torch.zeros(B, T_r, dtype=torch.long),
            old_log_probs=torch.zeros(B, T_r),
            ref_log_probs=torch.zeros(B, T_r),
            rewards=torch.rand(B),
            values=torch.rand(B),
            advantages=torch.rand(B),
            returns=torch.rand(B),
        )
        mbs = list(batch.mini_batches(mini_batch_size=4))
        self.assertEqual(len(mbs), 4)
        for mb in mbs:
            self.assertEqual(mb.rewards.shape[0], 4)


# ---------------------------------------------------------------------------
# Value head tests
# ---------------------------------------------------------------------------

class TestValueHead(unittest.TestCase):
    def test_output_shape(self):
        from models.policy_model import ValueHead
        vh = ValueHead(hidden_size=768, inner_size=256)
        x = torch.randn(8, 768)
        out = vh(x)
        self.assertEqual(out.shape, (8,))

    def test_output_is_scalar(self):
        from models.policy_model import ValueHead
        vh = ValueHead(hidden_size=128, inner_size=64)
        x = torch.randn(1, 128)
        out = vh(x)
        self.assertEqual(out.dim(), 1)


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------

class TestRewardFunctions(unittest.TestCase):

    def test_length_reward(self):
        from rewards.reward_functions import LengthReward
        fn = LengthReward(ideal_length=10, sigma=5)
        texts = ["word " * 10, "w", "word " * 50]
        rewards = fn(texts)
        self.assertEqual(rewards.shape[0], 3)
        # Reward for ideal-length text should be highest
        self.assertGreater(rewards[0].item(), rewards[1].item())
        self.assertGreater(rewards[0].item(), rewards[2].item())
        # All values in [-1, 1]
        self.assertTrue((rewards >= -1.0).all())
        self.assertTrue((rewards <= 1.0).all())

    def test_running_mean_std(self):
        from rewards.reward_functions import RunningMeanStd
        rms = RunningMeanStd(momentum=0.1)
        for _ in range(100):
            rms.update(np.random.randn(32))
        # After many updates, mean should be near 0, var near 1
        self.assertAlmostEqual(rms.mean, 0.0, delta=0.5)
        self.assertAlmostEqual(rms.var, 1.0, delta=0.5)

    def test_composite_reward_shapes(self):
        from rewards.reward_functions import CompositeReward
        # Disable heavy models for unit test
        cfg = RewardConfig(
            sentiment_weight=0.0,   # skip classifier
            toxicity_weight=0.0,    # skip classifier
            fluency_weight=0.0,     # skip LM
            length_weight=1.0,      # only length
            normalize_rewards=False,
        )
        fn = CompositeReward(cfg, device="cpu")
        texts = ["Hello world this is a test", "Short", "A longer sentence with many words to test"]
        rewards = fn(texts)
        self.assertEqual(rewards.shape[0], 3)

    def test_composite_reward_describe(self):
        from rewards.reward_functions import CompositeReward
        cfg = RewardConfig(
            sentiment_weight=0.0,
            toxicity_weight=0.0,
            fluency_weight=0.0,
            length_weight=1.0,
            normalize_rewards=False,
        )
        fn = CompositeReward(cfg, device="cpu")
        texts = ["test text here"]
        desc = fn.describe(texts)
        self.assertEqual(len(desc), 1)
        self.assertIn("total", desc[0])
        self.assertIn("length", desc[0])


# ---------------------------------------------------------------------------
# PPO stats tests
# ---------------------------------------------------------------------------

class TestPPOStats(unittest.TestCase):
    def test_addition(self):
        from training.ppo_trainer import PPOStats
        s1 = PPOStats(policy_loss=0.5, value_loss=0.3, n_updates=1)
        s2 = PPOStats(policy_loss=0.1, value_loss=0.7, n_updates=1)
        s3 = s1 + s2
        self.assertAlmostEqual(s3.policy_loss, 0.6)
        self.assertAlmostEqual(s3.value_loss, 1.0)
        self.assertEqual(s3.n_updates, 2)

    def test_mean(self):
        from training.ppo_trainer import PPOStats
        s = PPOStats(policy_loss=1.0, value_loss=2.0, n_updates=2)
        m = s.mean()
        self.assertAlmostEqual(m.policy_loss, 0.5)
        self.assertAlmostEqual(m.value_loss, 1.0)

    def test_to_dict(self):
        from training.ppo_trainer import PPOStats
        s = PPOStats()
        d = s.to_dict()
        self.assertIn("train/policy_loss", d)
        self.assertIn("train/approx_kl", d)


# ---------------------------------------------------------------------------
# Dataset tests
# ---------------------------------------------------------------------------

class TestDataset(unittest.TestCase):

    def _make_tokenizer(self):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return tok

    def test_synthetic_dataset(self):
        from data.dataset import SyntheticPromptDataset
        tok = self._make_tokenizer()
        ds = SyntheticPromptDataset(tok, n_samples=20, seed=0)
        self.assertEqual(len(ds), 20)
        item = ds[0]
        self.assertIsInstance(item, torch.Tensor)
        self.assertEqual(item.dim(), 1)

    def test_collate_fn_left_pads(self):
        from data.dataset import SyntheticPromptDataset
        tok = self._make_tokenizer()
        ds = SyntheticPromptDataset(tok, n_samples=10, seed=42)
        items = [ds[i] for i in range(4)]
        batch = ds.collate_fn(items)
        # All sequences should have the same length
        self.assertEqual(
            batch["input_ids"].shape[0], 4
        )
        # Attention mask should sum to actual token counts
        for i, item in enumerate(items):
            mask_sum = batch["attention_mask"][i].sum().item()
            self.assertEqual(mask_sum, item.size(0))

    def test_get_eval_prompts(self):
        from data.dataset import SyntheticPromptDataset
        tok = self._make_tokenizer()
        ds = SyntheticPromptDataset(tok, n_samples=20, seed=0)
        ep = ds.get_eval_prompts(n=8)
        self.assertIn("input_ids", ep)
        self.assertIn("attention_mask", ep)
        self.assertEqual(ep["input_ids"].shape[0], 8)


# ---------------------------------------------------------------------------
# Adaptive KL controller tests
# ---------------------------------------------------------------------------

class TestAdaptiveKL(unittest.TestCase):
    def test_increases_on_high_kl(self):
        from training.ppo_trainer import AdaptiveKLController
        ctl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.1, speed=0.1)
        ctl.update(observed_kl=5.0)   # Way above target
        self.assertGreater(ctl.value, 0.2)

    def test_decreases_on_low_kl(self):
        from training.ppo_trainer import AdaptiveKLController
        ctl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.1, speed=0.1)
        ctl.update(observed_kl=0.001)  # Way below target
        self.assertLess(ctl.value, 0.2)

    def test_no_change_on_target(self):
        from training.ppo_trainer import AdaptiveKLController
        ctl = AdaptiveKLController(init_kl_coef=0.2, target_kl=0.1, speed=0.1)
        initial = ctl.value
        ctl.update(observed_kl=0.1)   # Exactly on target
        self.assertAlmostEqual(ctl.value, initial, places=5)

    def test_disabled_when_no_target(self):
        from training.ppo_trainer import AdaptiveKLController
        ctl = AdaptiveKLController(init_kl_coef=0.5, target_kl=None)
        ctl.update(observed_kl=99.0)
        self.assertAlmostEqual(ctl.value, 0.5, places=5)


# ---------------------------------------------------------------------------
# Distinct-n diversity metric tests
# ---------------------------------------------------------------------------

class TestDiversityMetrics(unittest.TestCase):
    def test_distinct_1_perfect(self):
        from evaluate import distinct_n
        texts = ["alpha beta gamma", "delta epsilon zeta"]
        score = distinct_n(texts, n=1)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_distinct_1_repetitive(self):
        from evaluate import distinct_n
        texts = ["the the the", "the the the"]
        score = distinct_n(texts, n=1)
        self.assertAlmostEqual(score, 1 / 6, places=5)

    def test_average_length(self):
        from evaluate import average_length
        texts = ["one two three", "a b"]
        avg = average_length(texts)
        self.assertAlmostEqual(avg, 2.5, places=3)


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
