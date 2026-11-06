"""CPU regression checks for episode splits, horizon alignment, and error metrics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from examples.simBenchmarks.Robotwin.train_files.ablation_data import prepare_splits, load_manifest
from examples.simBenchmarks.Robotwin.eval_files.evaluate_action_errors import (
    ErrorAccumulator,
    action_layout,
    anchor_noise_seed,
)


def make_dataset(root, episode_count=2, length=24):
    """A small numeric fixture; video placeholders are never decoded."""
    root = Path(root)
    (root / "meta").mkdir(parents=True)
    features = {
        name: {"dtype": "float32", "shape": [14], "names": [f"d{i}" for i in range(14)]}
        for name in ("observation.state", "action", "observation.ee_pose", "observation.target_pose")
    }
    features["observation.target_frame"] = {"dtype": "int64", "shape": [1], "names": ["frame_index"]}
    videos = [f"observation.images.{name}" for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")]
    for name in videos:
        features[name] = {
            "dtype": "video", "shape": [3, 240, 320], "names": ["channels", "height", "width"],
            "info": {"video.channels": 3, "video.fps": 15},
        }
    for name in ("episode_index", "frame_index", "index", "task_index"):
        features[name] = {"dtype": "int64", "shape": [1], "names": None}
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    info = {
        "codebase_version": "v2.1", "fps": 15, "chunks_size": 1000,
        "total_episodes": episode_count, "total_frames": episode_count * length,
        "total_videos": episode_count * len(videos), "total_tasks": 1,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    modality = {"state": {}, "action": {}, "video": {}, "annotation": {
        "human.action.task_description": {"original_key": "task_index"}
    }}
    for group, original in (("state", "observation.state"), ("action", "action")):
        for name, start, end in (("left_joints", 0, 6), ("left_gripper", 6, 7),
                                 ("right_joints", 7, 13), ("right_gripper", 13, 14)):
            modality[group][name] = {"original_key": original, "start": start, "end": end}
    for key in videos:
        modality["video"][key.rsplit(".", 1)[1]] = {"original_key": key}
    episodes = []
    for eid in range(episode_count):
        values = (np.arange(length, dtype=np.float32)[:, None] + eid * 100)
        poses = np.repeat(values, 14, axis=1)
        targets = np.maximum(int(.8 * length), np.arange(length) + 1).clip(max=length - 1)
        state = poses.copy()
        state[:, [6, 13]] = (np.arange(length) % 2)[:, None]
        action = state + np.float32(.25)
        table = pd.DataFrame({
            "observation.state": list(state), "action": list(action),
            "observation.ee_pose": list(poses), "observation.target_pose": list(poses[targets]),
            "observation.target_frame": [np.array([g], dtype=np.int64) for g in targets],
            "episode_index": np.full(length, eid, dtype=np.int64),
            "frame_index": np.arange(length, dtype=np.int64),
            "index": np.arange(length, dtype=np.int64) + eid * length,
            "task_index": np.zeros(length, dtype=np.int64),
            "timestamp": np.arange(length, dtype=np.float32) / 15,
        })
        path = root / info["data_path"].format(episode_chunk=0, episode_index=eid)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        for video_key in videos:
            path = root / info["video_path"].format(episode_chunk=0, episode_index=eid, video_key=video_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"numeric-test-video-placeholder")
        episodes.append({"episode_index": eid, "length": length, "tasks": ["test instruction"]})
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/modality.json").write_text(json.dumps(modality))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "test instruction"}) + "\n")
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in episodes))
    return root


class AblationDataTests(unittest.TestCase):
    def test_episode_holdout_and_idempotence(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_dataset(Path(temp) / "source")
            dest = Path(temp) / "prepared"
            path = prepare_splits(source, dest, first_k=3, max_horizon=16)
            m = load_manifest(path)
            self.assertEqual(m["split_mode"], "episode_holdout")
            self.assertFalse(set(m["train_episodes"]) & set(m["eval_episodes"]))
            self.assertEqual(len(m["train_anchors"]), 3)
            self.assertEqual(len(m["eval_anchors"]), 3)
            for subset, ids in (("train", m["train_episodes"]), ("eval", m["eval_episodes"])):
                files = list((dest / subset / "data").rglob("*.parquet"))
                self.assertEqual(len(files), 1)
                self.assertTrue(files[0].is_symlink())
                self.assertEqual(int(pd.read_parquet(files[0])["episode_index"].iloc[0]), ids[0])
                self.assertFalse((dest / subset / "meta/stats_gr00t.json").exists())
            self.assertEqual(path, prepare_splits(source, dest, first_k=3, max_horizon=16))
            with self.assertRaises(FileExistsError):
                prepare_splits(source, dest, first_k=4, max_horizon=16)

    def test_one_episode_requires_explicit_smoke(self):
        with tempfile.TemporaryDirectory() as temp:
            source = make_dataset(Path(temp) / "source", episode_count=1)
            with self.assertRaises(ValueError):
                prepare_splits(source, Path(temp) / "formal", first_k=3)
            m = load_manifest(prepare_splits(source, Path(temp) / "smoke", first_k=3, smoke=True))
            self.assertEqual(m["split_mode"], "train_replay_smoke")
            self.assertEqual(m["train_episodes"], m["eval_episodes"])

    def test_short_horizon_uses_prefix_and_shared_training_starts(self):
        from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
        from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset
        from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP

        with tempfile.TemporaryDirectory() as temp:
            source = make_dataset(Path(temp) / "source")
            m = load_manifest(prepare_splits(source, Path(temp) / "split", first_k=None))
            cfg = {
                "lerobot_version": "v2.0", "video_backend": "torchvision_av", "action_mode": "abs",
                "include_state": True, "include_target_pose": True, "action_horizon": 2,
                "sampling_action_horizon": 16, "drop_incomplete_action_chunks": True,
                "state_action_dtype": "float32",
            }
            ds = make_LeRobotSingleDataset(Path(m["prepared_root"]), "train", "robotwin", data_cfg=cfg)
            ds._modality_keys["video"] = []  # This test checks numeric labels, not video decoding.
            eid = int(ds.trajectory_ids[0])
            raw = ds.get_step_data(eid, 0)
            actual = np.concatenate([raw[k] for k in ds.modality_keys["action"]], axis=1)
            table = ds.get_trajectory_data(eid)
            expected = np.stack(table["action"].iloc[:2])[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13]]
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(ROBOT_TYPE_CONFIG_MAP["robotwin"].action_indices, list(range(16)))
            mix = LeRobotMixtureDataset([(ds, 1.0)], mode="train", data_cfg=cfg,
                                       balance_dataset_weights=False, balance_trajectory_weights=False)
            self.assertEqual(mix.dataset_lengths.tolist(), [9])
            self.assertTrue(all(0 <= int(mix.sample_step(i)[2]) < 9 for i in range(100)))
            expected_mean = eid * 100 + 11.5
            self.assertAlmostEqual(float(ds.metadata.statistics.state["left_joints"].mean[0]), expected_mean)


class ActionMetricTests(unittest.TestCase):
    def test_mean_errors_and_units(self):
        keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
        layout = action_layout(keys, dict(zip(keys, (6, 6, 1, 1))))
        metric = ErrorAccumulator(2, layout)
        reference = np.zeros((2, 14))
        norm_pred = np.ones((2, 14))
        norm_pred[1] = 3
        physical_pred = np.full((2, 14), 2.)
        physical_pred[:, 12:] = 3
        metric.add(norm_pred, reference, physical_pred, reference)
        result = metric.summary()
        for key, expected in (("normalized_mse", 5), ("normalized_mae", 2),
                              ("first_step_normalized_mse", 1), ("physical_joint_mse", 4),
                              ("physical_joint_mae", 2), ("physical_gripper_mse", 9)):
            self.assertAlmostEqual(result[key], expected)
        self.assertEqual([row["normalized_mse"] for row in metric.per_step()], [1, 9])
        self.assertEqual(len(metric.per_dimension()), 14)
        self.assertEqual(anchor_noise_seed(42, 1, 2), anchor_noise_seed(42, 1, 2))

    def test_metrics_reject_broadcasting_and_nonfinite_values(self):
        keys = ["action.left_joints", "action.right_joints", "action.left_gripper", "action.right_gripper"]
        metric = ErrorAccumulator(2, action_layout(keys, dict(zip(keys, (6, 6, 1, 1)))))
        reference = np.zeros((2, 14))
        with self.assertRaises(ValueError):
            metric.add(np.zeros((1, 14)), reference, reference, reference)
        prediction = reference.copy()
        prediction[0, 0] = np.nan
        with self.assertRaises(ValueError):
            metric.add(prediction, reference, reference, reference)


if __name__ == "__main__":
    unittest.main()
