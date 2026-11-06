"""CPU checks for the future-goal time condition in RoboTwin data samples."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from test_robotwin_ablation_tools import make_dataset


class TimeBudgetDataTests(unittest.TestCase):
    CONFIG = {
        "lerobot_version": "v2.0",
        "video_backend": "torchvision_av",
        "action_type": "abs_qpos",
        "action_mode": "abs",
        "include_state": True,
        "include_target_pose": True,
        "include_time_budget": True,
        "action_horizon": 16,
        "sampling_action_horizon": 16,
        "state_action_dtype": "float32",
        "image_size": 224,
    }

    @classmethod
    def setUpClass(cls):
        from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

        cls.temp = tempfile.TemporaryDirectory(prefix="robotwin-time-budget-test-")
        cls.source = make_dataset(Path(cls.temp.name) / "source", episode_count=1, length=139)
        parquet = cls.source / "data/chunk-000/episode_000000.parquet"
        table = pd.read_parquet(parquet)
        # Keep a concrete 126-frame / 15 Hz example independent of random goals.
        target_frames = table["observation.target_frame"].tolist()
        target_frames[4] = np.array([130], dtype=np.int64)
        table["observation.target_frame"] = target_frames
        target_poses = table["observation.target_pose"].tolist()
        target_poses[4] = table.at[130, "observation.ee_pose"].copy()
        table["observation.target_pose"] = target_poses
        table.to_parquet(parquet, index=False)
        cls.original_rows = len(table)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cls.dataset = make_LeRobotSingleDataset(
                cls.source.parent,
                cls.source.name,
                "robotwin",
                data_cfg=OmegaConf.create(cls.CONFIG),
            )
        # This fixture exercises numeric conditions and normalization, not codecs.
        # All actual get_step_data/transforms/_pack_sample implementations remain.
        cls.dataset._modality_keys["video"] = []

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.dataset.data_cfg = OmegaConf.create(self.CONFIG)
        self.dataset._lerobot_info_meta["fps"] = 15

    def _sample(self, frame=4):
        raw = self.dataset.get_step_data(0, frame)
        return self.dataset._pack_sample(self.dataset.transforms(raw))

    def test_future_frame_interval_is_float32_seconds(self):
        for frame in range(5):
            with self.subTest(frame=frame):
                raw = self.dataset.get_step_data(0, frame)
                expected = np.array(
                    [[(raw["target_frame"] - raw["frame_index"]) / 15]], dtype=np.float32
                )
                self.assertEqual(raw["time_budget"].shape, (1, 1))
                self.assertEqual(raw["time_budget"].dtype, np.dtype("float32"))
                np.testing.assert_array_equal(raw["time_budget"], expected)
                sample = self.dataset._pack_sample(self.dataset.transforms(raw))
                np.testing.assert_array_equal(sample["time_budget"], expected)
                self.assertEqual(sample["time_budget"].dtype, np.dtype("float32"))
                self.assertEqual(sample["state"].shape, (1, 14))
                self.assertEqual(sample["action"].shape, (16, 14))
                self.assertEqual(sample["target_pose"].shape, (1, 14))
        self.assertEqual((sample["frame_index"], sample["target_frame"]), (4, 130))
        np.testing.assert_array_equal(sample["time_budget"], np.array([[8.4]], dtype=np.float32))

    def test_disabled_time_preserves_state_and_action(self):
        enabled = self._sample()
        self.dataset.data_cfg.include_time_budget = False
        raw = self.dataset.get_step_data(0, 4)
        disabled = self.dataset._pack_sample(self.dataset.transforms(raw))
        self.assertNotIn("time_budget", raw)
        self.assertNotIn("time_budget", disabled)
        self.assertEqual(set(enabled) - set(disabled), {"time_budget"})
        for key in ("state", "action", "target_pose"):
            np.testing.assert_array_equal(disabled[key], enabled[key])
        self.assertEqual(disabled["target_frame"], enabled["target_frame"])
        self.assertEqual(disabled["frame_index"], enabled["frame_index"])

    def test_time_does_not_follow_joint_output_dtype(self):
        self.dataset.data_cfg.state_action_dtype = "float16"
        sample = self._sample()
        self.assertEqual(sample["action"].dtype, np.dtype("float16"))
        self.assertEqual(sample["state"].dtype, np.dtype("float16"))
        self.assertEqual(sample["time_budget"].dtype, np.dtype("float32"))
        np.testing.assert_array_equal(sample["time_budget"], np.array([[8.4]], dtype=np.float32))

    def test_metadata_frequency_controls_duration(self):
        self.dataset._lerobot_info_meta["fps"] = 30
        sample = self._sample()
        np.testing.assert_array_equal(sample["time_budget"], np.array([[4.2]], dtype=np.float32))

    def test_self_goal_is_rejected_without_truncating_trajectory(self):
        last = self.original_rows - 1
        with self.assertRaisesRegex(ValueError, "strictly future"):
            self.dataset.get_step_data(0, last)
        self.assertEqual(int(self.dataset.trajectory_lengths[0]), self.original_rows)
        self.assertEqual(len(self.dataset.get_trajectory_data(0)), self.original_rows)
        self.dataset.data_cfg.include_time_budget = False
        raw = self.dataset.get_step_data(0, last)
        self.assertEqual(raw["target_frame"], raw["frame_index"])
        self.assertNotIn("time_budget", raw)

    def test_time_requires_enabled_goal_input(self):
        self.dataset.data_cfg.include_target_pose = False
        with self.assertRaisesRegex(ValueError, "requires include_target_pose=True"):
            self.dataset.get_step_data(0, 0)
        with self.assertRaisesRegex(ValueError, "requires include_target_pose=True"):
            self.dataset._pack_sample({})

    def test_non_boolean_time_flag_is_rejected(self):
        for value in ("False", "true", 1, None):
            with self.subTest(value=value):
                self.dataset.data_cfg.include_time_budget = value
                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    self.dataset.get_step_data(0, 0)

    def test_invalid_metadata_frequency_is_rejected(self):
        for fps in (None, 0, -15, float("nan"), float("inf"), True, "invalid"):
            with self.subTest(fps=fps):
                self.dataset._lerobot_info_meta["fps"] = fps
                with self.assertRaisesRegex(ValueError, "positive finite dataset metadata fps"):
                    self.dataset.get_step_data(0, 0)

    def test_missing_or_invalid_packed_time_is_rejected(self):
        data = self.dataset.transforms(self.dataset.get_step_data(0, 4))
        missing = {key: value for key, value in data.items() if key != "time_budget"}
        with self.assertRaisesRegex(ValueError, "raw sample has no time_budget"):
            self.dataset._pack_sample(missing)
        for value in (
            np.array([8.4], dtype=np.float32),
            np.array([[0.0]], dtype=np.float32),
            np.array([[float("nan")]], dtype=np.float32),
            np.array([[float("inf")]], dtype=np.float32),
        ):
            with self.subTest(value=value):
                invalid = {**data, "time_budget": value}
                with self.assertRaisesRegex(ValueError, "positive finite seconds"):
                    self.dataset._pack_sample(invalid)


if __name__ == "__main__":
    unittest.main()
