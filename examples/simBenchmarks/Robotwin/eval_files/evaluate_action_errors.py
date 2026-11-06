#!/usr/bin/env python3
"""Evaluate PI action predictions on the fixed anchors of an ablation manifest.

The checkpoint's training transforms normalize observations and labels. Physical
references are captured before those transforms; predictions are inverted with
the checkpoint's transforms. This is an offline action prediction evaluation,
not a closed-loop RoboTwin success-rate evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from typing import Any


def action_layout(keys: list[str], dims: dict[str, int]) -> dict[str, Any]:
    """Describe the model's ordered qpos dimensions without hard-coded slices."""
    joint_indices, gripper_indices, names, units = [], [], [], []
    cursor = 0
    for key in keys:
        width = int(dims[key])
        if width < 1:
            raise ValueError(f"Invalid action dimension for {key!r}: {width}")
        is_gripper = "gripper" in key.lower()
        indices = list(range(cursor, cursor + width))
        (gripper_indices if is_gripper else joint_indices).extend(indices)
        names.extend(f"{key}[{i}]" for i in range(width))
        units.extend(["unitless" if is_gripper else "rad"] * width)
        cursor += width
    if cursor != 14 or len(joint_indices) != 12 or len(gripper_indices) != 2:
        raise ValueError(
            "Expected RoboTwin's 12 joint + 2 gripper dimensions; "
            f"got keys={keys}, dims={dims}"
        )
    return {
        "joint_indices": joint_indices,
        "gripper_indices": gripper_indices,
        "names": names,
        "units": units,
    }


class ErrorAccumulator:
    """Accumulate elementwise errors in float64, including short last batches."""

    def __init__(self, horizon: int, layout: dict[str, Any]):
        import numpy as np

        self.horizon = int(horizon)
        if self.horizon < 1:
            raise ValueError("Action horizon must be positive")
        self.layout = layout
        self.n_samples = 0
        shape = (self.horizon, len(layout["names"]))
        self.norm_squared = np.zeros(shape, dtype=np.float64)
        self.norm_absolute = np.zeros(shape, dtype=np.float64)
        self.phys_squared = np.zeros(shape, dtype=np.float64)
        self.phys_absolute = np.zeros(shape, dtype=np.float64)

    def add(self, pred_norm, reference_norm, pred_physical, reference_physical):
        import numpy as np

        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (pred_norm, reference_norm, pred_physical, reference_physical)
        ]
        for value in arrays:
            if value.shape != self.norm_squared.shape:
                raise ValueError(
                    f"Expected action shape {self.norm_squared.shape}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError("Non-finite action predictions or references")
        norm_error = arrays[0] - arrays[1]
        physical_error = arrays[2] - arrays[3]
        self.norm_squared += np.square(norm_error)
        self.norm_absolute += np.abs(norm_error)
        self.phys_squared += np.square(physical_error)
        self.phys_absolute += np.abs(physical_error)
        self.n_samples += 1
        return norm_error, physical_error

    def summary(self) -> dict[str, Any]:
        if not self.n_samples:
            raise ValueError("Cannot report metrics without evaluated samples")
        n = self.n_samples
        joints = self.layout["joint_indices"]
        grippers = self.layout["gripper_indices"]
        return {
            "n_samples": n,
            "normalized_mse": float(self.norm_squared.mean() / n),
            "normalized_mae": float(self.norm_absolute.mean() / n),
            "first_step_normalized_mse": float(self.norm_squared[0].mean() / n),
            "first_step_normalized_mae": float(self.norm_absolute[0].mean() / n),
            "physical_joint_mse": float(self.phys_squared[:, joints].mean() / n),
            "physical_joint_mae": float(self.phys_absolute[:, joints].mean() / n),
            "physical_gripper_mse": float(self.phys_squared[:, grippers].mean() / n),
            "physical_gripper_mae": float(self.phys_absolute[:, grippers].mean() / n),
        }

    def per_dimension(self) -> list[dict[str, Any]]:
        if not self.n_samples:
            raise ValueError("No samples")
        count = self.n_samples * self.horizon
        return [
            {
                "dimension": i,
                "name": name,
                "physical_unit": self.layout["units"][i],
                "normalized_mse": float(self.norm_squared[:, i].sum() / count),
                "normalized_mae": float(self.norm_absolute[:, i].sum() / count),
                "physical_mse": float(self.phys_squared[:, i].sum() / count),
                "physical_mae": float(self.phys_absolute[:, i].sum() / count),
            }
            for i, name in enumerate(self.layout["names"])
        ]

    def per_step(self) -> list[dict[str, Any]]:
        if not self.n_samples:
            raise ValueError("No samples")
        joints = self.layout["joint_indices"]
        grippers = self.layout["gripper_indices"]
        n = self.n_samples
        return [
            {
                "action_step": step + 1,
                "step_offset": step,
                "n_samples": n,
                "normalized_mse": float(self.norm_squared[step].mean() / n),
                "normalized_mae": float(self.norm_absolute[step].mean() / n),
                "physical_joint_mse": float(self.phys_squared[step, joints].mean() / n),
                "physical_joint_mae": float(self.phys_absolute[step, joints].mean() / n),
                "physical_gripper_mse": float(self.phys_squared[step, grippers].mean() / n),
                "physical_gripper_mae": float(self.phys_absolute[step, grippers].mean() / n),
            }
            for step in range(self.horizon)
        ]


def anchor_noise_seed(base_seed: int, episode_index: int, frame_index: int) -> int:
    """Use the same deterministic per-anchor RNG seed across all model variants."""
    token = f"{base_seed}:{episode_index}:{frame_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % (2**63)


def require_matching_manifests(training: dict[str, Any], evaluation: dict[str, Any]) -> None:
    """Reject a different split/suite before loading the checkpoint model."""
    if training != evaluation:
        changed = sorted(
            key for key in set(training) | set(evaluation)
            if training.get(key) != evaluation.get(key)
        )
        raise ValueError(
            "Evaluation manifest differs from the checkpoint's training suite "
            f"(different fields: {changed}). Use the same prepared manifest; "
            "cross-dataset evaluation is not supported by this script."
        )


def validate_time_budget(value, frame_index: int, target_frame: int, fps: float) -> float:
    """Check the physical time hint passed to the model, without renormalizing it."""
    import numpy as np

    budget = np.asarray(value)
    if budget.shape != (1, 1) or budget.dtype != np.float32:
        raise ValueError(
            "time_budget must be a float32 array of shape (1, 1); "
            f"got shape={budget.shape}, dtype={budget.dtype}"
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Time-budget fps must be positive and finite")
    seconds = float(budget[0, 0])
    expected = (target_frame - frame_index) / fps
    if not np.isfinite(seconds) or seconds <= 0 or expected <= 0:
        raise ValueError("time_budget must be finite and positive")
    if not np.isclose(seconds, expected, rtol=1e-6, atol=1e-6):
        raise ValueError(
            f"time_budget={seconds} disagrees with the manifest frame gap: {expected} seconds"
        )
    return seconds


def _seed_runtime(seed: int, torch_module) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Summary JSON path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates all manifest anchors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=123)
    return parser


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Import model dependencies and perform inference only when explicitly run."""
    if args.max_samples < 0:
        raise ValueError("--max-samples must be nonnegative")
    checkpoint = args.checkpoint.resolve()
    manifest_path = args.manifest.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    repo_root = Path(__file__).resolve().parents[4]
    train_files = Path(__file__).resolve().parents[1] / "train_files"
    for directory in (repo_root, train_files):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from ablation_data import load_manifest, register_mixture
    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
    from starVLA.model.framework.base_framework import baseframework
    from starVLA.model.framework.share_tools import read_mode_config

    manifest = load_manifest(manifest_path)
    split_mode = manifest["split_mode"]
    if split_mode not in {"episode_holdout", "train_replay_smoke"}:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")
    if split_mode == "episode_holdout" and set(manifest["train_episodes"]) & set(manifest["eval_episodes"]):
        raise ValueError("Heldout evaluation episodes overlap with training episodes")
    anchors = list(manifest["eval_anchors"])
    if args.max_samples:
        anchors = anchors[: args.max_samples]
    if not anchors:
        raise ValueError("Manifest has no evaluation anchors")

    cfg_dict, _ = read_mode_config(checkpoint)
    cfg = OmegaConf.create(cfg_dict)
    training_manifest = register_mixture(cfg)
    require_matching_manifests(training_manifest, manifest)
    action_cfg = cfg.framework.action_model
    horizon = int(action_cfg.action_horizon)
    if not 1 <= horizon <= int(manifest["max_horizon"]):
        raise ValueError(f"Checkpoint horizon {horizon} exceeds the manifest's valid horizon")
    if int(action_cfg.action_dim) != 14:
        raise ValueError("This evaluator expects 14-dimensional RoboTwin qpos actions")
    target_pose_dim = int(action_cfg.get("target_pose_dim", 0))
    if target_pose_dim not in (0, 14):
        raise ValueError(f"Unsupported checkpoint target_pose_dim={target_pose_dim}")
    use_time_budget = bool(action_cfg.get("use_time_budget", False))
    if use_time_budget and target_pose_dim != 14:
        raise ValueError("Time conditioning requires target_pose_dim=14")
    if bool(cfg.datasets.vla_data.get("include_time_budget", False)) != use_time_budget:
        raise ValueError("Checkpoint dataset and action head disagree on time conditioning")
    use_state = bool(cfg.datasets.vla_data.get("include_state", False))
    if use_state and int(action_cfg.state_dim) != 14:
        raise ValueError("Checkpoint proprioceptive state must have 14 dimensions")

    proc = PolicyNormProcessor(str(checkpoint))
    layout = action_layout(proc.action_keys, proc.action_key_dims)
    eval_dataset = Path(manifest["eval_dataset"])
    if not eval_dataset.is_absolute():
        raise ValueError("Manifest eval_dataset must be an absolute path")
    fps = float(manifest["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Manifest fps must be positive and finite")
    eval_info = json.loads((eval_dataset / "meta" / "info.json").read_text())
    if float(eval_info["fps"]) != fps:
        raise ValueError("Manifest fps disagrees with the evaluation dataset metadata")
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    for key, value in {
        "include_state": True,
        "include_target_pose": True,
        "include_time_budget": use_time_budget,
        "action_horizon": horizon,
        "lerobot_version": "v2.0",
        "video_backend": "torchvision_av",
        "state_action_dtype": "float32",
        "num_workers": 0,
    }.items():
        eval_cfg[key] = value
    ds = make_LeRobotSingleDataset(
        eval_dataset.parent, eval_dataset.name, "robotwin", data_cfg=eval_cfg
    )
    if list(ds.modality_configs["action"].delta_indices) != list(range(horizon)):
        raise RuntimeError("Dataset factory must honor action_horizon with offsets range(H)")
    if list(ds.modality_keys["action"]) != list(proc.action_keys):
        raise ValueError("Evaluation and checkpoint action key orders differ")
    if list(ds.modality_keys.get("state", [])) != list(proc.state_keys):
        raise ValueError("Evaluation and checkpoint state key orders differ")
    # Construction may cache heldout statistics, but those statistics are NEVER
    # used for model inputs or normalized references: replace the bound transform.
    ds.transforms = proc.transform

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu explicitly")
    _seed_runtime(args.seed, torch)
    model = baseframework.from_pretrained(str(checkpoint)).to(
        device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    ).eval()
    if int(model.action_model.target_pose_dim) != target_pose_dim:
        raise ValueError("Loaded model goal conditioning differs from its saved checkpoint config")
    if bool(getattr(model.action_model, "use_time_budget", False)) != use_time_budget:
        raise ValueError("Loaded model time conditioning differs from its saved checkpoint config")

    output = args.output.resolve()
    if output in {checkpoint, manifest_path}:
        raise ValueError("--output must not overwrite the checkpoint or manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_path = output.with_name(output.stem + "_per_episode.json")
    step_path = output.with_name(output.stem + "_per_step.jsonl")
    total = ErrorAccumulator(horizon, layout)
    per_episode: dict[int, ErrorAccumulator] = {}
    time_budgets: list[float] = []
    step_temp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{step_path.name}.", delete=False
        ) as detail_file:
            step_temp = Path(detail_file.name)
            for index, anchor in enumerate(anchors):
                episode = int(anchor["episode_index"])
                frame = int(anchor["frame_index"])
                target_frame = int(anchor["target_frame"])
                trajectory_index = ds.get_trajectory_index(episode)
                length = int(ds.trajectory_lengths[trajectory_index])
                if frame < 0 or frame + horizon > length or not frame < target_frame < length:
                    raise ValueError(f"Invalid evaluation anchor: {anchor}; trajectory length={length}")
                raw = ds.get_step_data(episode, frame)
                if int(raw["frame_index"]) != frame or int(raw["target_frame"]) != target_frame:
                    raise ValueError(f"Dataset target/current frame does not match manifest anchor {anchor}")
                reference_physical = np.concatenate(
                    [np.asarray(raw[key]) for key in proc.action_keys], axis=-1
                ).astype(np.float64, copy=True)
                sample = ds._pack_sample(proc.transform(raw))
                reference_normalized = np.asarray(sample["action"], dtype=np.float64)
                inputs = {key: sample[key] for key in ("image", "lang")}
                if use_state:
                    inputs["state"] = sample["state"]
                if target_pose_dim:
                    inputs["target_pose"] = sample["target_pose"]
                time_budget_seconds = None
                if use_time_budget:
                    time_budget_seconds = validate_time_budget(
                        sample["time_budget"], frame, target_frame, fps
                    )
                    inputs["time_budget"] = sample["time_budget"]
                    time_budgets.append(time_budget_seconds)
                noise_seed = anchor_noise_seed(args.noise_seed, episode, frame)
                _seed_runtime(noise_seed, torch)
                with torch.inference_mode():
                    prediction = model.predict_action(examples=[inputs])["normalized_actions"]
                prediction = np.asarray(prediction)
                if prediction.shape != (1, horizon, 14):
                    raise ValueError(f"Expected prediction shape {(1, horizon, 14)}, got {prediction.shape}")
                pred_normalized = prediction[0].astype(np.float64)
                pred_physical = proc.unapply_actions(pred_normalized.astype(np.float32))
                norm_error, physical_error = total.add(
                    pred_normalized, reference_normalized, pred_physical, reference_physical
                )
                episode_acc = per_episode.setdefault(episode, ErrorAccumulator(horizon, layout))
                episode_acc.add(pred_normalized, reference_normalized, pred_physical, reference_physical)
                sample_acc = ErrorAccumulator(horizon, layout)
                sample_acc.add(pred_normalized, reference_normalized, pred_physical, reference_physical)
                for row in sample_acc.per_step():
                    offset = row["step_offset"]
                    row.update({
                        "sample_index": index,
                        "episode_index": episode,
                        "frame_index": frame,
                        "target_frame": target_frame,
                        "time_budget_seconds": time_budget_seconds,
                        "noise_seed": noise_seed,
                        "normalized_error": norm_error[offset].tolist(),
                        "physical_error": physical_error[offset].tolist(),
                    })
                    detail_file.write(json.dumps(row, allow_nan=False) + "\n")
        os.replace(step_temp, step_path)
        step_temp = None
    finally:
        if step_temp is not None:
            step_temp.unlink(missing_ok=True)

    episode_rows = [
        {"episode_index": episode, **acc.summary()}
        for episode, acc in sorted(per_episode.items())
    ]
    _write_json(episode_path, episode_rows)
    report = {
        **total.summary(),
        "n_episodes": len(per_episode),
        "action_horizon": horizon,
        "fps": fps,
        "action_horizon_seconds": horizon / fps,
        "target_pose_dim": target_pose_dim,
        "goal_conditioning": bool(target_pose_dim),
        "state_conditioning": use_state,
        "time_conditioning": use_time_budget,
        "time_definition": "(target_frame-frame_index)/fps",
        "time_units": "seconds",
        "time_source": "expert_frame_gap_seconds",
        "time_is_oracle": use_time_budget,
        "time_budget_min_seconds": min(time_budgets) if time_budgets else None,
        "time_budget_max_seconds": max(time_budgets) if time_budgets else None,
        "time_budget_mean_seconds": float(np.mean(time_budgets)) if time_budgets else None,
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_signature": manifest["source_signature"],
        "training_manifest_matches": True,
        "eval_dataset": str(eval_dataset),
        "split_mode": split_mode,
        "is_heldout": split_mode == "episode_holdout",
        "goal_source": manifest["goal_source"],
        "goal_is_oracle": bool(target_pose_dim),
        "norm_source": "train",
        "norm_key": proc.unnorm_key,
        "seed": args.seed,
        "noise_seed": args.noise_seed,
        "noise_seed_strategy": "sha256(base_seed, episode_index, frame_index)",
        "units": {"physical_joint_mse": "rad^2", "physical_joint_mae": "rad",
                  "physical_gripper_mse": "unitless", "physical_gripper_mae": "unitless"},
        "reference": "raw physical actions; normalized references use checkpoint training transforms",
        "metric_scope": "offline action prediction, not closed-loop task success",
        "per_dimension": total.per_dimension(),
        "per_step": total.per_step(),
        "per_episode_file": str(episode_path),
        "per_step_file": str(step_path),
    }
    if split_mode == "train_replay_smoke":
        report["evaluation_note"] = "Training-episode replay smoke test; this is not heldout validation."
    _write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate(args)
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, (list, dict))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
