"""CPU-only checks of QwenPI's real condition transport, without loading a VLM."""

import ast
import contextlib
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple
import unittest
from unittest.mock import patch

import numpy as np
import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "starVLA/model/framework/VLM4A/QwenPI.py"
)


def extract_methods():
    """Compile only the actual forward/predict bodies and inference decorator."""
    tree = ast.parse(SOURCE.read_text())
    framework = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen_PI"
    )
    methods = [
        copy.deepcopy(node) for node in framework.body
        if isinstance(node, ast.FunctionDef) and node.name in {"forward", "predict_action"}
    ]
    if len(methods) != 2:
        raise AssertionError("Could not find QwenPI's forward and predict_action")
    module = ast.Module(body=methods, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "torch": torch,
        "List": List,
        "Tuple": Tuple,
        # Image preparation is outside the condition-transport contract tested here.
        "to_pil_preserve": lambda images: images,
        "resize_images": lambda images, target_size: images,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


def snapshot(vl_features, state, mask, target_pose, time_budget, actions=None):
    def clone(value):
        return None if value is None else value.detach().clone()

    return {
        "vl_features": [clone(value) for value in vl_features],
        "state": clone(state),
        "mask": clone(mask),
        "target_pose": clone(target_pose),
        "time_budget": clone(time_budget),
        "actions": clone(actions),
    }


class CaptureHead(torch.nn.Module):
    def __init__(self, target_pose_dim=14, use_time_budget=True, prediction_dtype=torch.float32):
        super().__init__()
        self.target_pose_dim = target_pose_dim
        self.use_time_budget = use_time_budget
        self.prediction_dtype = prediction_dtype
        self.scale = torch.nn.Parameter(torch.tensor(1.0, device="cpu"))
        self.training_inputs = None
        self.prediction_inputs = None
        self.prediction_grad_enabled = None

    def forward(
        self, vl_features, actions, state=None, encoder_attention_mask=None,
        target_pose=None, time_budget=None,
    ):
        self.training_inputs = snapshot(
            vl_features, state, encoder_attention_mask, target_pose, time_budget, actions
        )
        return self.scale * actions.mean()

    def predict_action(
        self, vl_features, state=None, encoder_attention_mask=None,
        target_pose=None, time_budget=None,
    ):
        self.prediction_inputs = snapshot(
            vl_features, state, encoder_attention_mask, target_pose, time_budget
        )
        self.prediction_grad_enabled = torch.is_grad_enabled()
        batch_size = vl_features[0].shape[0]
        return torch.full(
            (batch_size, 16, 14), 0.25, dtype=self.prediction_dtype, device="cpu"
        )


class StubVLMFramework:
    def __init__(self, head):
        self.action_model = head
        self.action_horizon = 16
        self.config = SimpleNamespace(
            framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 8}),
            datasets=SimpleNamespace(vla_data=SimpleNamespace(obs_image_size=None)),
        )
        self.encoded_features = None
        self.encoded_mask = None

    def _encode_vl_hidden_states(self, images, instructions):
        batch_size = len(images)
        ids = torch.arange(1, batch_size + 1, dtype=torch.float32, device="cpu")
        self.encoded_features = [
            (ids[:, None, None] + layer * 10).expand(-1, 3, 4).clone()
            for layer in range(2)
        ]
        self.encoded_mask = torch.tensor(
            [[1, 1, 0] if i % 2 == 0 else [1, 0, 0] for i in range(batch_size)],
            dtype=torch.int64, device="cpu",
        )
        return self.encoded_features, self.encoded_mask


class NoTimeRead(dict):
    """Make accidental budget access fail even when a malformed key is present."""
    def __getitem__(self, key):
        if key == "time_budget":
            raise AssertionError("Disabled time conditioning read time_budget")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "time_budget":
            raise AssertionError("Disabled time conditioning read time_budget")
        return super().get(key, default)


def examples(include_goal=True, include_time=True, include_state=True):
    result = []
    for i in range(2):
        sample = {
            "image": [f"head-view-{i}"],
            "lang": f"task-{i}",
            "action": np.full((16, 14), (i + 1) * 1000, dtype=np.float32),
        }
        if include_state:
            sample["state"] = np.full((1, 14), (i + 1) * 100, dtype=np.float32)
        if include_goal:
            sample["target_pose"] = np.full((1, 14), (i + 1) * 10, dtype=np.float32)
        if include_time:
            sample["time_budget"] = np.array([[4.0 if i == 0 else 2.25]], dtype=np.float32)
        result.append(sample)
    return result


class QwenPITimeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.methods = extract_methods()

    def invoke(self, method, framework, batch):
        # The production methods select CUDA autocast. Suppress only that
        # context for this CPU transport test; all tensor operations stay real.
        with patch.object(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext()):
            return self.methods[method](framework, examples=batch)

    def test_training_repeats_time_state_goal_actions_and_vl_in_matching_order(self):
        head = CaptureHead()
        framework = StubVLMFramework(head)
        batch = examples()
        result = self.invoke("forward", framework, batch)
        result["action_loss"].backward()
        self.assertIsNotNone(head.scale.grad)
        self.assertGreater(float(head.scale.grad), 0)
        actual = head.training_inputs
        # Current PI intentionally overrides the configured 8 with fixed repeat=2.
        for field, shape in (
            ("actions", (4, 16, 14)), ("state", (4, 1, 14)),
            ("target_pose", (4, 1, 14)), ("time_budget", (4, 1, 1)),
        ):
            key = "action" if field == "actions" else field
            expected = torch.from_numpy(np.stack([sample[key] for sample in batch]))
            torch.testing.assert_close(actual[field], expected.repeat(2, 1, 1))
            self.assertEqual(tuple(actual[field].shape), shape)
            self.assertEqual(actual[field].device.type, "cpu")
            self.assertEqual(actual[field].dtype, torch.float32)
        self.assertEqual(actual["time_budget"][:, 0, 0].tolist(), [4.0, 2.25, 4.0, 2.25])
        for actual_layer, original in zip(actual["vl_features"], framework.encoded_features):
            torch.testing.assert_close(actual_layer, original.repeat(2, 1, 1))
        torch.testing.assert_close(actual["mask"], framework.encoded_mask.repeat(2, 1).bool())

    def test_prediction_does_not_repeat_and_retains_state_with_bf16_output(self):
        head = CaptureHead(prediction_dtype=torch.bfloat16)
        framework = StubVLMFramework(head)
        batch = examples()
        result = self.invoke("predict_action", framework, batch)
        actual = head.prediction_inputs
        for field in ("state", "target_pose", "time_budget"):
            expected = torch.from_numpy(np.stack([sample[field] for sample in batch]))
            torch.testing.assert_close(actual[field], expected)
            self.assertEqual(actual[field].shape[0], 2)
        self.assertEqual(tuple(actual["state"].shape), (2, 1, 14))
        self.assertFalse(head.prediction_grad_enabled)
        for actual_layer, original in zip(actual["vl_features"], framework.encoded_features):
            torch.testing.assert_close(actual_layer, original)
        torch.testing.assert_close(actual["mask"], framework.encoded_mask.bool())
        prediction = result["normalized_actions"]
        self.assertEqual(prediction.shape, (2, 16, 14))
        self.assertEqual(prediction.dtype, np.float32)
        np.testing.assert_array_equal(prediction, np.full((2, 16, 14), 0.25, np.float32))

    def test_disabled_time_is_not_read_for_goal_only_models(self):
        batch = [NoTimeRead(sample) for sample in examples(include_time=False)]
        for sample in batch:
            sample["time_budget"] = object()
        for method in ("forward", "predict_action"):
            with self.subTest(method=method):
                head = CaptureHead(use_time_budget=False)
                self.invoke(method, StubVLMFramework(head), batch)
                captured = head.training_inputs if method == "forward" else head.prediction_inputs
                self.assertIsNone(captured["time_budget"])
                self.assertIsNotNone(captured["target_pose"])
                self.assertIsNotNone(captured["state"])

    def test_legacy_examples_without_time_or_goal_keep_optional_state_contract(self):
        for include_state in (True, False):
            batch = examples(include_goal=False, include_time=False, include_state=include_state)
            for method in ("forward", "predict_action"):
                with self.subTest(method=method, state=include_state):
                    head = CaptureHead(target_pose_dim=0, use_time_budget=False)
                    self.invoke(method, StubVLMFramework(head), batch)
                    captured = head.training_inputs if method == "forward" else head.prediction_inputs
                    self.assertIsNone(captured["target_pose"])
                    self.assertIsNone(captured["time_budget"])
                    self.assertEqual(captured["state"] is not None, include_state)

    def test_enabled_time_requires_budget_in_every_example(self):
        batch = examples()
        del batch[1]["time_budget"]
        for method in ("forward", "predict_action"):
            with self.subTest(method=method):
                head = CaptureHead()
                with self.assertRaisesRegex(KeyError, "time_budget"):
                    self.invoke(method, StubVLMFramework(head), batch)
                self.assertIsNone(head.training_inputs)
                self.assertIsNone(head.prediction_inputs)

    def test_prediction_accepts_one_legacy_example_without_action_labels(self):
        sample = examples(include_goal=False, include_time=False)[0]
        del sample["action"]
        head = CaptureHead(target_pose_dim=0, use_time_budget=False)
        result = self.invoke("predict_action", StubVLMFramework(head), sample)
        self.assertEqual(result["normalized_actions"].shape, (1, 16, 14))
        self.assertEqual(tuple(head.prediction_inputs["state"].shape), (1, 1, 14))


if __name__ == "__main__":
    unittest.main()
