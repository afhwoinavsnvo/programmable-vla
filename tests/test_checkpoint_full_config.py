"""CPU regression tests for restoring complete checkpoint observation contracts."""

import json
from pathlib import Path
import tempfile
import unittest

import yaml

from starVLA.model.framework.share_tools import read_mode_config


def complete_config():
    return {
        "framework": {
            "name": "QwenPI",
            "qwenvl": {"vl_hidden_dim": 32},
            "action_model": {
                "action_horizon": 16,
                "action_dim": 14,
                "state_dim": 14,
                "target_pose_dim": 14,
                "use_time_budget": True,
                "hidden_size": 32,
                "noise_s": 0.999,
                "repeated_diffusion_steps": 2,
                "diffusion_model_cfg": {},
            },
        },
        "datasets": {
            "vla_data": {
                "include_state": True,
                "include_target_pose": True,
                "include_time_budget": True,
                "state_action_dtype": "float32",
                "image_size": [192, 256],
                "obs_image_size": [192, 256],
                "ablation_manifest": "/training/prepared/manifest.json",
                "train_start_frames": 5,
            }
        },
        "trainer": {"max_train_steps": 20},
    }


class CheckpointFullConfigTests(unittest.TestCase):
    def make_checkpoint(self, full=None, partial=None, suffix=".pt"):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        run = Path(temporary.name)
        checkpoint = run / "final_model" / ("model" + suffix)
        checkpoint.parent.mkdir()
        checkpoint.touch()  # The config reader must not deserialize model weights.
        stats = {"robotwin": {"action": {"min": [-1.0] * 14, "max": [1.0] * 14}}}
        (run / "dataset_statistics.json").write_text(json.dumps(stats))
        if full is not None:
            (run / "config.full.yaml").write_text(json.dumps(full))
        if partial is not None:
            (run / "config.yaml").write_text(json.dumps(partial))
        return checkpoint, run, stats

    def test_full_snapshot_restores_state_preprocessing_and_manifest_without_writes(self):
        full = complete_config()
        partial = {
            "framework": {"name": "QwenPI", "action_model": {"action_horizon": 16}},
            "datasets": {"vla_data": {"include_target_pose": True}},
        }
        for suffix in (".pt", ".safetensors"):
            with self.subTest(suffix=suffix):
                checkpoint, run, expected_stats = self.make_checkpoint(full, partial, suffix)
                before = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
                cfg, stats = read_mode_config(checkpoint)
                data = cfg["datasets"]["vla_data"]
                self.assertIs(data["include_state"], True)
                self.assertIs(data["include_time_budget"], True)
                self.assertEqual(data["state_action_dtype"], "float32")
                self.assertEqual(data["image_size"], [192, 256])
                self.assertEqual(data["obs_image_size"], [192, 256])
                self.assertEqual(data["ablation_manifest"], "/training/prepared/manifest.json")
                self.assertEqual(data["train_start_frames"], 5)
                self.assertIs(cfg["framework"]["action_model"]["use_time_budget"], True)
                self.assertEqual(stats, expected_stats)
                after = {path.name: path.read_bytes() for path in run.iterdir() if path.is_file()}
                self.assertEqual(before, after)

    def test_accessed_snapshot_overrides_full_values_including_explicit_false(self):
        partial = {
            "framework": {
                "action_model": {"action_horizon": 8, "target_pose_dim": 0, "use_time_budget": False}
            },
            "datasets": {
                "vla_data": {
                    "include_state": False,
                    "include_target_pose": False,
                    "include_time_budget": False,
                    "state_action_dtype": "float16",
                }
            },
            "trainer": {"max_train_steps": 30},
        }
        checkpoint, _, _ = self.make_checkpoint(complete_config(), partial)
        cfg, _ = read_mode_config(checkpoint)
        data = cfg["datasets"]["vla_data"]
        for flag in ("include_state", "include_target_pose", "include_time_budget"):
            self.assertIs(data[flag], False)
        action = cfg["framework"]["action_model"]
        self.assertIs(action["use_time_budget"], False)
        self.assertEqual(action["target_pose_dim"], 0)
        self.assertEqual(action["action_horizon"], 8)
        self.assertEqual(action["future_action_window_size"], 7)
        self.assertEqual(data["state_action_dtype"], "float16")
        self.assertEqual(data["obs_image_size"], [192, 256])
        self.assertEqual(cfg["trainer"]["max_train_steps"], 30)

    def test_legacy_accessed_only_checkpoint_keeps_existing_behavior(self):
        partial = complete_config()
        partial["datasets"]["vla_data"]["include_state"] = False
        del partial["framework"]["action_model"]["use_time_budget"]
        del partial["datasets"]["vla_data"]["include_time_budget"]
        checkpoint, run, expected_stats = self.make_checkpoint(partial=partial)
        self.assertFalse((run / "config.full.yaml").exists())
        cfg, stats = read_mode_config(checkpoint)
        self.assertIs(cfg["datasets"]["vla_data"]["include_state"], False)
        self.assertNotIn("use_time_budget", cfg["framework"]["action_model"])
        self.assertNotIn("include_time_budget", cfg["datasets"]["vla_data"])
        self.assertEqual(stats, expected_stats)

    def test_bad_full_snapshot_fails_instead_of_falling_back_to_partial(self):
        checkpoint, run, _ = self.make_checkpoint(complete_config(), complete_config())
        (run / "config.full.yaml").write_text("datasets: [unterminated\n")
        with self.assertRaises(yaml.YAMLError):
            read_mode_config(checkpoint)

    def test_bad_partial_snapshot_fails_instead_of_ignoring_its_overrides(self):
        checkpoint, run, _ = self.make_checkpoint(complete_config(), complete_config())
        (run / "config.yaml").write_text("datasets: [unterminated\n")
        with self.assertRaises(yaml.YAMLError):
            read_mode_config(checkpoint)

    def test_bad_normalization_statistics_still_fail(self):
        checkpoint, run, _ = self.make_checkpoint(complete_config(), complete_config())
        (run / "dataset_statistics.json").write_text("{invalid json")
        with self.assertRaises(json.JSONDecodeError):
            read_mode_config(checkpoint)


if __name__ == "__main__":
    unittest.main()
