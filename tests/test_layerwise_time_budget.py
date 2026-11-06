"""CPU regression checks for PI goal-and-time conditioning; no VLM is loaded."""

import unittest

from omegaconf import OmegaConf
import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import (
    LayerwiseFlowmatchingActionHead,
    TimeBudgetEncoder,
)


_OMITTED = object()


def tiny_config(use_time_budget=False, target_pose_dim=14):
    action = {
        "action_dim": 14,
        "state_dim": 14,
        "target_pose_dim": target_pose_dim,
        "action_horizon": 2,
        "num_inference_timesteps": 2,
        "num_target_vision_tokens": 2,
        "add_pos_embed": True,
        "max_seq_len": 32,
        "noise_beta_alpha": 1.5,
        "noise_beta_beta": 1.0,
        "noise_s": 0.999,
        "num_timestep_buckets": 1000,
        "diffusion_model_cfg": {
            "num_layers": 2,
            "input_embedding_dim": 32,
            "cross_attention_dim": 32,
            "num_attention_heads": 2,
            "attention_head_dim": 16,
            "dropout": 0.0,
            "final_dropout": False,
            "norm_type": "ada_norm",
            "positional_embeddings": None,
            "interleave_self_attention": True,
        },
    }
    if use_time_budget is not _OMITTED:
        action["use_time_budget"] = use_time_budget
    return OmegaConf.create({"framework": {"action_model": action}})


class LayerwiseTimeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def setUp(self):
        torch.manual_seed(120)
        self.vl = [torch.randn(2, 3, 32) for _ in range(2)]
        self.actions = torch.randn(2, 2, 14)
        self.state = torch.randn(2, 1, 14)
        self.goal = torch.randn(2, 1, 14)
        self.seconds = torch.tensor([[[2.0]], [[5.0]]])

    def head(self, enabled=True, target_pose_dim=14):
        torch.manual_seed(321)
        result = LayerwiseFlowmatchingActionHead(tiny_config(enabled, target_pose_dim))
        self.assertTrue(all(parameter.device.type == "cpu" for parameter in result.parameters()))
        return result

    def call(self, head, method, budget):
        kwargs = {"target_pose": self.goal, "time_budget": budget}
        if method == "forward":
            return head(self.vl, self.actions, self.state, **kwargs)
        return head.predict_action(self.vl, self.state, **kwargs)

    def test_encoder_returns_one_time_token(self):
        encoder = TimeBudgetEncoder(hidden_size=32)
        result = encoder(self.seconds)
        self.assertEqual(tuple(result.shape), (2, 1, 32))
        self.assertTrue(torch.isfinite(result).all())

    def test_time_encoder_receives_gradient_from_actual_action_loss(self):
        head = self.head()
        loss = self.call(head, "forward", self.seconds)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        gradients = [parameter.grad for parameter in head.time_budget_encoder.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.0)

    def test_fixed_sampling_noise_changing_only_time_changes_actions(self):
        head = self.head().eval()
        torch.manual_seed(2027)
        first = self.call(head, "predict", self.seconds)
        torch.manual_seed(2027)
        changed = self.call(head, "predict", self.seconds + 4.0)
        self.assertEqual(tuple(first.shape), (2, 2, 14))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(torch.isfinite(changed).all())
        self.assertGreater(float((first - changed).abs().max()), 1e-7)
        # Repeating the original input/noise reproduces the original output.
        torch.manual_seed(2027)
        again = self.call(head, "predict", self.seconds)
        torch.testing.assert_close(first, again, rtol=0, atol=0)

    def test_time_on_preserves_seeded_shared_goal_model_weights(self):
        goal_only = self.head(enabled=False)
        goal_and_time = self.head(enabled=True)
        old = goal_only.state_dict()
        new = goal_and_time.state_dict()
        extra = set(new) - set(old)
        self.assertTrue(extra)
        self.assertTrue(all(name.startswith("time_budget_encoder.") for name in extra))
        self.assertFalse(any(name.startswith("time_budget_encoder.") for name in old))
        for name, value in old.items():
            with self.subTest(parameter=name):
                self.assertTrue(torch.equal(value, new[name]))

    def test_omitted_flag_and_explicit_time_off_load_legacy_goal_weights(self):
        # An old goal-only configuration/checkpoint has no time flag or weights.
        legacy_config = self.head(enabled=_OMITTED).eval()
        explicit_off = self.head(enabled=False).eval()
        self.assertIsNone(legacy_config.time_budget_encoder)
        self.assertIsNone(explicit_off.time_budget_encoder)
        status = explicit_off.load_state_dict(legacy_config.state_dict(), strict=True)
        self.assertEqual(status.missing_keys, [])
        self.assertEqual(status.unexpected_keys, [])
        torch.manual_seed(44)
        old_loss = legacy_config(self.vl, self.actions, self.state, target_pose=self.goal)
        torch.manual_seed(44)
        new_loss = explicit_off(self.vl, self.actions, self.state, target_pose=self.goal)
        torch.testing.assert_close(old_loss, new_loss, rtol=0, atol=0)
        token_lengths = []
        handle = explicit_off.model.register_forward_pre_hook(
            lambda _module, _args, kwargs: token_lengths.append(kwargs["hidden_states"].shape[1]),
            with_kwargs=True,
        )
        try:
            torch.manual_seed(45)
            original = legacy_config.predict_action(self.vl, self.state, target_pose=self.goal)
            torch.manual_seed(45)
            current = explicit_off.predict_action(self.vl, self.state, target_pose=self.goal)
        finally:
            handle.remove()
        torch.testing.assert_close(original, current, rtol=0, atol=0)
        # State + goal + two future tokens + two action tokens; no time token.
        self.assertEqual(token_lengths, [6, 6])

    def test_enabled_time_requires_budget_in_both_paths(self):
        head = self.head()
        for method in ("forward", "predict"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "time_budget is required"):
                self.call(head, method, None)

    def test_wrong_shapes_rejected_in_both_paths(self):
        head = self.head()
        invalid = (
            torch.ones(()), torch.ones(2), torch.ones(2, 1),
            torch.ones(1, 1, 1), torch.ones(2, 2, 1), torch.ones(2, 1, 2),
        )
        for value in invalid:
            for method in ("forward", "predict"):
                with self.subTest(shape=tuple(value.shape), method=method):
                    with self.assertRaisesRegex(ValueError, "Expected time_budget shape"):
                        self.call(head, method, value)

    def test_nonfinite_or_nonpositive_seconds_rejected_in_both_paths(self):
        head = self.head()
        for bad in (float("nan"), float("inf"), float("-inf"), 0.0, -0.5):
            budget = self.seconds.clone()
            budget[0, 0, 0] = bad
            for method in ("forward", "predict"):
                with self.subTest(value=bad, method=method):
                    with self.assertRaisesRegex(ValueError, "positive finite seconds"):
                        self.call(head, method, budget)

    def test_time_off_rejects_supplied_time_in_both_paths(self):
        head = self.head(enabled=False)
        for method in ("forward", "predict"):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, "use_time_budget=True"):
                self.call(head, method, self.seconds)

    def test_time_conditioning_requires_goal_encoder(self):
        with self.assertRaisesRegex(ValueError, "requires target_pose_dim"):
            self.head(enabled=True, target_pose_dim=0)

    def test_time_flag_must_be_boolean(self):
        for value in ("true", "false", 1, 0, None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                self.head(enabled=value)


if __name__ == "__main__":
    unittest.main()
