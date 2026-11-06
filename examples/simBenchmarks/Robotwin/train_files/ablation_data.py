"""Prepare paired RoboTwin ablations without sharing held-out normalization data.

The subsets keep complete episodes, original episode IDs and original relative
Parquet/video paths.  Only their eligible *input anchors* are restricted.  Future
poses are expert-provided goal conditions, not predicted goals or time budgets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MANIFEST_VERSION = 1
MIXTURE_NAME = "robotwin_ablation"
META_FILES = ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json")
POSE_KEYS = ("observation.ee_pose", "observation.target_pose")
TARGET_FRAME_KEY = "observation.target_frame"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON objects in {path}")
    return records


def _integer(value: Any, name: str) -> int:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain exactly one integer, got {value!r}")
    return int(array.item())


def _positive_integer(value: Any, name: str) -> int:
    result = _integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _relative_path(root: Path, pattern: str, **values: Any) -> Path:
    relative = Path(pattern.format(**values))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Dataset path must stay inside its root: {relative}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return relative


def _source_metadata(source: Path) -> tuple[dict, list[dict], list[dict], dict]:
    for name in META_FILES:
        if not (source / "meta" / name).is_file():
            raise FileNotFoundError(source / "meta" / name)
    info = json.loads((source / "meta/info.json").read_text())
    if not str(info.get("codebase_version", "")).startswith("v2."):
        raise ValueError("This preparer requires a converted LeRobot v2.x dataset")
    episodes = _read_jsonl(source / "meta/episodes.jsonl")
    tasks = _read_jsonl(source / "meta/tasks.jsonl")
    modality = json.loads((source / "meta/modality.json").read_text())
    if not episodes or not tasks:
        raise ValueError("The source must have episodes and language tasks")
    ids = [_integer(row["episode_index"], "episode_index") for row in episodes]
    if len(set(ids)) != len(ids) or any(index < 0 for index in ids):
        raise ValueError("Source episode IDs must be unique nonnegative integers")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError("info.json total_episodes disagrees with episodes.jsonl")
    features = info.get("features", {})
    for key in POSE_KEYS:
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or tuple(feature.get("shape", ())) != (14,):
            raise ValueError(f"{key} must declare float32 shape (14,), ordered left pose then right pose")
    feature = features.get(TARGET_FRAME_KEY, {})
    if feature.get("dtype") != "int64" or tuple(feature.get("shape", ())) != (1,):
        raise ValueError(f"{TARGET_FRAME_KEY} must declare int64 shape (1,)")
    fps = info.get("fps", 0)
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Source fps must be a positive finite number")
    _positive_integer(info.get("chunks_size"), "chunks_size")
    if not isinstance(info.get("data_path"), str):
        raise ValueError("info.json must provide its Parquet data_path pattern")
    return info, sorted(episodes, key=lambda row: row["episode_index"]), tasks, modality


def _episode_paths(source: Path, info: dict, episode_index: int) -> list[Path]:
    values = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // int(info["chunks_size"]),
    }
    paths = [_relative_path(source, info["data_path"], **values)]
    video_keys = [key for key, feature in info["features"].items() if feature.get("dtype") == "video"]
    if video_keys and not isinstance(info.get("video_path"), str):
        raise ValueError("Video features require a video_path pattern")
    for video_key in video_keys:
        paths.append(_relative_path(source, info["video_path"], video_key=video_key, **values))
    return paths


def _source_signature(source: Path, info: dict, episodes: list[dict]) -> str:
    digest = hashlib.sha256()
    for name in META_FILES:
        digest.update(name.encode())
        digest.update((source / "meta" / name).read_bytes())
    for episode in episodes:
        for relative in _episode_paths(source, info, int(episode["episode_index"])):
            stat = (source / relative).stat()
            digest.update(f"{relative.as_posix()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _episode_anchors(source: Path, info: dict, episode: dict, max_horizon: int, first_k: int | None) -> list[dict]:
    episode_index = _integer(episode["episode_index"], "episode_index")
    length = _positive_integer(episode["length"], f"episode {episode_index} length")
    parquet = source / _episode_paths(source, info, episode_index)[0]
    columns = ["episode_index", "frame_index", *POSE_KEYS, TARGET_FRAME_KEY]
    data = pd.read_parquet(parquet, columns=columns)
    if len(data) != length:
        raise ValueError(f"Episode {episode_index}: metadata length {length} differs from {len(data)} rows")
    episode_ids = np.array([_integer(value, "episode_index") for value in data["episode_index"]])
    frame_ids = np.array([_integer(value, "frame_index") for value in data["frame_index"]])
    if not np.all(episode_ids == episode_index) or not np.array_equal(frame_ids, np.arange(length)):
        raise ValueError(f"Episode {episode_index}: Parquet episode/frame IDs do not match its metadata")
    poses = {}
    for key in POSE_KEYS:
        arrays = [np.asarray(value, dtype=np.float32) for value in data[key]]
        if any(array.shape != (14,) for array in arrays):
            raise ValueError(f"Episode {episode_index}: {key} must contain complete left/right 7D poses")
        poses[key] = np.stack(arrays)
        if not np.isfinite(poses[key]).all():
            raise ValueError(f"Episode {episode_index}: non-finite {key}")
    targets = np.array([_integer(value, TARGET_FRAME_KEY) for value in data[TARGET_FRAME_KEY]])
    last_twenty_percent = int(0.8 * length)
    for frame, target in enumerate(targets):
        if not last_twenty_percent <= target < length:
            raise ValueError(f"Episode {episode_index}, frame {frame}: target {target} is outside the final 20%")
        if target <= frame and not (frame == length - 1 and target == frame):
            raise ValueError(f"Episode {episode_index}, frame {frame}: target must be strictly in the future")
        if not np.allclose(poses[POSE_KEYS[1]][frame], poses[POSE_KEYS[0]][target], rtol=1e-5, atol=1e-6):
            raise ValueError(f"Episode {episode_index}, frame {frame}: target pose disagrees with ee_pose[{target}]")
    stop = max(0, length - max_horizon + 1)
    if first_k is not None:
        stop = min(first_k, stop)
    anchors = [
        {"episode_index": episode_index, "frame_index": frame, "target_frame": int(targets[frame])}
        for frame in range(stop)
        if targets[frame] > frame
    ]
    if not anchors:
        raise ValueError(f"Episode {episode_index} has no valid anchors for max_horizon={max_horizon}, first_k={first_k}")
    return anchors


def _write_subset(source: Path, destination: Path, info: dict, episodes: list[dict], tasks: list[dict], modality: dict) -> None:
    meta = destination / "meta"
    meta.mkdir(parents=True)
    subset_info = copy.deepcopy(info)
    video_count = sum(feature.get("dtype") == "video" for feature in info["features"].values())
    subset_info.update({
        "total_episodes": len(episodes),
        "total_frames": sum(int(row["length"]) for row in episodes),
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * video_count,
        "total_chunks": len({int(row["episode_index"]) // int(info["chunks_size"]) for row in episodes}),
        "splits": {"train": f"0:{len(episodes)}"},
    })
    for name, value in (("info.json", subset_info), ("modality.json", modality)):
        (meta / name).write_text(json.dumps(value, indent=2) + "\n")
    for name, rows in (("episodes.jsonl", episodes), ("tasks.jsonl", tasks)):
        (meta / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    # Intentionally do not copy stats.json, episode statistics, or loader caches.
    for episode in episodes:
        for relative in _episode_paths(source, info, int(episode["episode_index"])):
            link = destination / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(source / relative)


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path /= "manifest.json"
    manifest = json.loads(path.read_text())
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"Unsupported ablation manifest version in {path}")
    root = Path(manifest["prepared_root"])
    if not root.is_absolute() or root.resolve() != path.parent:
        raise ValueError("Manifest prepared_root must be its own absolute parent directory")
    if not Path(manifest["source_dataset"]).is_absolute():
        raise ValueError("Manifest source_dataset must be absolute")
    if manifest.get("goal_source") != "expert_future_pose":
        raise ValueError("Manifest must identify expert-provided future goal poses")
    max_horizon = _positive_integer(manifest["max_horizon"], "max_horizon")
    if manifest["first_k"] is not None:
        _positive_integer(manifest["first_k"], "first_k")
    groups = {}
    for split in ("train", "eval"):
        ids = manifest[f"{split}_episodes"]
        if not ids or any(type(index) is not int or index < 0 for index in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid {split} episode IDs")
        groups[split] = set(ids)
        expected = root / split
        if Path(manifest[f"{split}_dataset"]) != expected or not (expected / "meta/episodes.jsonl").is_file():
            raise ValueError(f"Manifest {split} dataset is missing or has an unexpected path")
        episode_records = _read_jsonl(expected / "meta/episodes.jsonl")
        episode_lengths = {
            _integer(row["episode_index"], "episode_index"): _positive_integer(row["length"], "episode length")
            for row in episode_records
        }
        if len(episode_lengths) != len(episode_records) or set(episode_lengths) != groups[split]:
            raise ValueError(f"Manifest {split} episode IDs disagree with the physical subset metadata")
        subset_info = json.loads((expected / "meta/info.json").read_text())
        if subset_info["total_episodes"] != len(episode_lengths) or subset_info["total_frames"] != sum(episode_lengths.values()):
            raise ValueError(f"Manifest {split} counts disagree with the physical subset metadata")
        anchors = manifest[f"{split}_anchors"]
        if not anchors:
            raise ValueError(f"Manifest has no {split} anchors")
        seen = set()
        for anchor in anchors:
            episode, frame, target = (anchor[key] for key in ("episode_index", "frame_index", "target_frame"))
            if any(type(value) is not int for value in (episode, frame, target)):
                raise ValueError("Anchor IDs must be integers")
            if episode not in groups[split] or frame < 0 or target <= frame or (episode, frame) in seen:
                raise ValueError(f"Invalid or duplicate {split} anchor: {anchor}")
            length = episode_lengths[episode]
            if frame + max_horizon > length or not int(0.8 * length) <= target < length:
                raise ValueError(f"Anchor has an incomplete action window or invalid future target: {anchor}")
            if manifest["first_k"] is not None and frame >= manifest["first_k"]:
                raise ValueError("Anchor is outside first_k")
            seen.add((episode, frame))
    mode = manifest.get("split_mode")
    if mode == "episode_holdout":
        if groups["train"] & groups["eval"]:
            raise ValueError("Held-out episode IDs overlap with training")
    elif mode == "train_replay_smoke":
        if groups["train"] != groups["eval"] or len(groups["train"]) != 1:
            raise ValueError("Replay smoke mode must use the same single episode")
    else:
        raise ValueError(f"Unknown split_mode: {mode!r}")
    return manifest


def prepare_splits(
    source: Path,
    dest: Path,
    *,
    seed: int = 42,
    eval_fraction: float = 0.2,
    max_horizon: int = 16,
    first_k: int | None = 5,
    smoke: bool = False,
) -> Path:
    source = Path(source).expanduser().resolve(strict=True)
    dest = Path(dest).expanduser().resolve()
    if source == dest or source in dest.parents or dest in source.parents:
        raise ValueError("Source and prepared dataset roots must be separate, non-nested directories")
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    max_horizon = _positive_integer(max_horizon, "max_horizon")
    if first_k is not None:
        first_k = _positive_integer(first_k, "first_k")
    if isinstance(eval_fraction, bool) or not math.isfinite(eval_fraction) or not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between zero and one")
    info, episodes, tasks, modality = _source_metadata(source)
    if len(episodes) < 2 and not smoke:
        raise ValueError("Formal held-out evaluation requires at least two converted episodes; use smoke=True only for replay validation")
    signature = _source_signature(source, info, episodes)
    parameters = {"seed": seed, "eval_fraction": float(eval_fraction), "max_horizon": max_horizon, "first_k": first_k, "smoke": bool(smoke)}
    manifest_path = dest / "manifest.json"
    if dest.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"Refusing to overwrite existing destination without a manifest: {dest}")
        existing = load_manifest(manifest_path)
        if existing["source_dataset"] != str(source) or existing.get("source_signature") != signature or existing.get("preparation_parameters") != parameters:
            raise FileExistsError(f"Destination was prepared from different data or parameters: {dest}")
        return manifest_path
    anchors_by_episode = {
        int(episode["episode_index"]): _episode_anchors(source, info, episode, max_horizon, first_k)
        for episode in episodes
    }
    if len(episodes) == 1:
        train_episodes = eval_episodes = episodes
        split_mode = "train_replay_smoke"
    else:
        ids = [int(episode["episode_index"]) for episode in episodes]
        shuffled = np.random.default_rng(seed).permutation(ids)
        eval_count = max(1, min(len(ids) - 1, math.ceil(len(ids) * eval_fraction)))
        eval_ids = set(int(value) for value in shuffled[:eval_count])
        train_episodes = [episode for episode in episodes if int(episode["episode_index"]) not in eval_ids]
        eval_episodes = [episode for episode in episodes if int(episode["episode_index"]) in eval_ids]
        split_mode = "episode_holdout"
    manifest = {
        "version": MANIFEST_VERSION,
        "source_dataset": str(source),
        "source_signature": signature,
        "prepared_root": str(dest),
        "split_mode": split_mode,
        "seed": seed,
        "max_horizon": max_horizon,
        "first_k": first_k,
        "fps": info["fps"],
        "goal_source": "expert_future_pose",
        "preparation_parameters": parameters,
        "train_episodes": [int(episode["episode_index"]) for episode in train_episodes],
        "eval_episodes": [int(episode["episode_index"]) for episode in eval_episodes],
        "train_anchors": [anchor for episode in train_episodes for anchor in anchors_by_episode[int(episode["episode_index"])]],
        "eval_anchors": [anchor for episode in eval_episodes for anchor in anchors_by_episode[int(episode["episode_index"])]],
        "train_dataset": str(dest / "train"),
        "eval_dataset": str(dest / "eval"),
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dest.name}.preparing-", dir=dest.parent))
    try:
        _write_subset(source, staging / "train", info, train_episodes, tasks, modality)
        _write_subset(source, staging / "eval", info, eval_episodes, tasks, modality)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if dest.exists():
            raise FileExistsError(f"Destination appeared during preparation: {dest}")
        staging.rename(dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    load_manifest(manifest_path)
    return manifest_path


def register_mixture(cfg: Any) -> dict[str, Any]:
    """Register only the training subset; evaluation must use train normalization."""
    data_cfg = cfg.datasets.vla_data
    root = Path(data_cfg.data_root_dir).expanduser().resolve()
    manifest_path = root / "manifest.json"
    recorded = data_cfg.get("ablation_manifest", None)
    if recorded is not None and Path(recorded).expanduser().resolve() != manifest_path:
        raise ValueError("ablation_manifest disagrees with data_root_dir/manifest.json")
    manifest = load_manifest(manifest_path)
    if data_cfg.data_mix != MIXTURE_NAME:
        raise ValueError(f"Prepared ablation configs must use data_mix={MIXTURE_NAME!r}")
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

    DATASET_NAMED_MIXTURES[data_cfg.data_mix] = [("train", 1.0, "robotwin")]
    return manifest


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Register prepared data and invoke the original StarVLA trainer")
    parser.add_argument("--config_yaml", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--dest", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument("--first-k", type=int, default=5, help="Input prefix size; use 0 for every complete anchor")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.train:
        if args.config_yaml is None:
            parser.error("--train requires --config_yaml")
        config_path = args.config_yaml.expanduser().resolve(strict=True)
        repo_root = Path(__file__).resolve().parents[4]
        os.chdir(repo_root)
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        register_mixture(cfg)
        # Set the seed before the original entry point constructs any modules.
        from accelerate.utils import set_seed

        set_seed(int(cfg.get("seed", 42)))
        entry = repo_root / "starVLA/training/train_starvla.py"
        previous_argv = sys.argv
        try:
            sys.argv = [str(entry), "--config_yaml", str(config_path)]
            runpy.run_path(str(entry), run_name="__main__")
        finally:
            sys.argv = previous_argv
        return
    if args.source is None or args.dest is None:
        parser.error("Preparation requires --source and --dest")
    print(prepare_splits(args.source, args.dest, seed=args.seed, eval_fraction=args.eval_fraction, max_horizon=args.max_horizon, first_k=None if args.first_k == 0 else args.first_k, smoke=args.smoke))


if __name__ == "__main__":
    _main()
