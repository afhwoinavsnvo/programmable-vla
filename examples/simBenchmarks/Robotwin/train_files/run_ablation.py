#!/usr/bin/env python3
"""Prepare and optionally run matched RoboTwin PI action-error experiments.

The default invocation only prepares data views/configs and prints commands.
Use --execute to train and evaluate serially. Smoke results on one episode are
training-replay diagnostics, not held-out performance or rollout success.
Conditions are baseline PI, PI with a goal pose, and PI with a goal pose plus
the demonstrated time to that goal. Time is independent of output horizon.
--resume-suite skips completed stages; it does not resume optimizer state from
an interrupted training run. Use a different --run-root for different settings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
HELPER = HERE / "ablation_data.py"
EVALUATOR = HERE.parent / "eval_files/evaluate_action_errors.py"
DEFAULT_CONFIG = HERE / "starvla_cotrain_robotwin_abs.yaml"
TIME_DEFINITION = "(target_frame-frame_index)/fps"
METRICS = (
    "normalized_mse", "normalized_mae", "first_step_normalized_mse",
    "first_step_normalized_mae", "physical_joint_mse", "physical_joint_mae",
    "physical_gripper_mse", "physical_gripper_mae", "n_samples", "n_episodes",
)


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", choices=("smoke", "formal"), default="smoke")
    p.add_argument("--source-dataset", type=Path, default=Path("playground/Datasets/RoboTwin/adjust_bottle_probe"))
    p.add_argument("--run-root", type=Path, default=Path("/var/tmp/jliu3375/starvla_robotwin_ablations"))
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Existing training YAML to use as the template.")
    p.add_argument("--gpus", default="0,2,3,7", help="Physical GPU indices, comma-separated; all runs execute serially.")
    p.add_argument("--horizons", nargs="+", type=positive_int, default=[1, 2, 4, 8, 16])
    p.add_argument("--conditions", nargs="+", choices=("baseline", "goal", "goal_time"),
                   default=["baseline", "goal", "goal_time"],
                   help="Input conditions to compare; --horizons 16 prepares the three main methods.")
    p.add_argument("--steps", type=positive_int, help="Defaults: smoke=50, formal=150000 optimizer updates.")
    p.add_argument("--warmup", type=int, help="Defaults: smoke=5, formal=5000 updates.")
    p.add_argument("--batch-size", type=positive_int, help="Per GPU; defaults: smoke=1, formal=8 (global32 on four GPUs).")
    p.add_argument("--first-k", type=positive_int, default=5, help="Train/evaluate on the first K eligible input frames per episode.")
    p.add_argument("--all-frames", action="store_true", help="Use all eligible input frames instead of --first-k.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-fraction", type=float, default=0.2, help="Held-out episode fraction when at least two episodes are available.")
    p.add_argument("--eval-samples", type=positive_int, help="Defaults: smoke=50, formal=200; same deterministic subset for every run.")
    p.add_argument("--matched-baselines", action="store_true", help="Train a no-goal PI baseline at every requested horizon.")
    p.add_argument("--only", nargs="+", help="Run names to select, e.g. baseline_h16 goal_h16 goal_time_h16; other suite settings must match on resume.")
    p.add_argument("--execute", action="store_true", help="Actually launch training and offline evaluation; default is preparation/dry-run only.")
    p.add_argument("--resume-suite", action="store_true", help="Explicitly skip completed stages of an identical suite; partial training is never overwritten.")
    return p


def absolute(path: Path) -> Path:
    return (path if path.is_absolute() else REPO / path).resolve()


def save_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def code_digest() -> str:
    digest = hashlib.sha256()
    for relative in (
        "starVLA/model/framework/VLM4A/QwenPI.py",
        "starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py",
        "starVLA/dataloader/lerobot_datasets.py",
        "starVLA/dataloader/gr00t_lerobot/datasets.py",
        "starVLA/training/train_starvla.py",
        str(Path(__file__).resolve().relative_to(REPO)),
        str(HELPER.relative_to(REPO)), str(EVALUATOR.relative_to(REPO)),
    ):
        path = REPO / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation is missing: {path}")
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configurations(args, manifest: dict, runs: list[dict]) -> dict[str, Path]:
    from omegaconf import OmegaConf

    paths = {}
    for run in runs:
        cfg = OmegaConf.load(args.config)
        cfg.seed = args.seed
        cfg.run_root_dir = str(args.run_root / "runs")
        cfg.run_id = run["name"]
        cfg.is_debug = False
        cfg.framework.name = "QwenPI"
        cfg.framework.qwenvl.base_vlm = str(absolute(Path(str(cfg.framework.qwenvl.base_vlm))))
        head = cfg.framework.action_model
        head.action_model_type = "LayerwiseFM"
        head.action_dim = head.state_dim = 14
        head.target_pose_dim = 14 if run["goal"] else 0
        head.use_time_budget = run["time"]
        head.action_horizon = run["horizon"]
        head.repeated_diffusion_steps = 2
        head.diffusion_model_cfg.interleave_self_attention = True
        for stale in ("num_layers", "output_dim"):
            if stale in head.diffusion_model_cfg:
                del head.diffusion_model_cfg[stale]
        data = cfg.datasets.vla_data
        for key, value in {
            "dataset_py": "lerobot_datasets", "lerobot_version": "v2.0",
            "data_root_dir": manifest["prepared_root"], "data_mix": "robotwin_ablation",
            "ablation_manifest": str(args.run_root / "prepared/manifest.json"),
            "include_state": True, "include_target_pose": True,
            "include_time_budget": run["time"],
            "action_horizon": run["horizon"], "sampling_action_horizon": max(args.horizons),
            "train_start_frames": args.first_k, "drop_incomplete_action_chunks": True,
            "action_type": "abs_qpos", "action_mode": "abs", "num_workers": 0,
            "per_device_batch_size": args.batch_size, "state_action_dtype": "float32",
            "obs_image_size": [224, 224], "image_size": 224, "video_backend": "torchvision_av",
            "balance_dataset_weights": False, "balance_trajectory_weights": False,
        }.items():
            data[key] = value
        cfg.trainer.max_train_steps = args.steps
        cfg.trainer.num_warmup_steps = args.warmup
        cfg.trainer.logging_frequency = 1 if args.profile == "smoke" else 100
        cfg.trainer.eval_interval = cfg.trainer.save_interval = args.steps + 1
        cfg.trainer.pretrained_checkpoint = None
        cfg.trainer.is_resume = False
        cfg.trainer.save_format = "pt"
        cfg.trainer.gradient_accumulation_steps = 1
        path = args.run_root / "configs" / f"{run['name']}.yaml"
        text = OmegaConf.to_yaml(cfg, resolve=True)
        if path.exists() and path.read_text() != text:
            raise ValueError(f"Refusing to replace a different generated configuration: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(text)
        paths[run["name"]] = path
    return paths


def commands(args, run: dict, config: Path) -> tuple[list[str], list[str]]:
    train = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
             str(REPO / "starVLA/config/deepseeds/deepspeed_zero2.yaml"),
             "--num_processes", str(len(args.gpu_ids)), str(HELPER),
             "--train", "--config_yaml", str(config)]
    evaluate = [sys.executable, str(EVALUATOR), "--checkpoint", str(run["checkpoint"]),
                "--manifest", str(args.run_root / "prepared/manifest.json"),
                "--output", str(run["metrics"]), "--device", "cuda:0",
                "--max-samples", str(args.eval_samples), "--seed", str(args.seed)]
    return train, evaluate


def environment(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("NCCL_SOCKET_IFNAME", "NCCL_IB_HCA"):
        env.pop(key, None)
    env.update(CUDA_VISIBLE_DEVICES=gpus, WANDB_MODE="disabled", TOKENIZERS_PARALLELISM="false",
               NO_ALBUMENTATIONS_UPDATE="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO), env.get("PYTHONPATH"))))
    return env


def display(command: list[str], env: dict[str, str]) -> str:
    prefix = ["env", "-u", "NCCL_SOCKET_IFNAME", "-u", "NCCL_IB_HCA"]
    prefix += [f"{key}={env[key]}" for key in ("CUDA_VISIBLE_DEVICES", "WANDB_MODE", "PYTHONPATH")]
    return shlex.join(prefix + command)


def validate_weights(config: Path) -> None:
    from omegaconf import OmegaConf

    directory = Path(str(OmegaConf.load(config).framework.qwenvl.base_vlm))
    if not (directory / "config.json").is_file():
        raise FileNotFoundError(f"Base VLM is not ready: {directory}. Download the configured model before --execute.")
    indices = list(directory.glob("*.safetensors.index.json")) + list(directory.glob("*.bin.index.json"))
    required = set()
    for index in indices:
        required.update(json.loads(index.read_text())["weight_map"].values())
    files = [directory / name for name in required] if required else list(directory.glob("*.safetensors")) + list(directory.glob("*.bin"))
    if not files or any(not file.is_file() or file.stat().st_size == 0 for file in files):
        raise FileNotFoundError(f"Base VLM weight files are missing/incomplete: {directory}")


def read_metrics(run: dict) -> dict:
    value = json.loads(run["metrics"].read_text())
    for key in ("normalized_mse", "normalized_mae"):
        if key not in value or isinstance(value[key], bool) or not math.isfinite(float(value[key])):
            raise ValueError(f"Missing/nonfinite {key}: {run['metrics']}")
    for key, expected in run["expected_metrics"].items():
        if value.get(key) != expected:
            raise ValueError(f"Metrics {key} does not match this suite's {run['name']}: {run['metrics']}")
    for key, expected in (("checkpoint", run["checkpoint"]), ("manifest", run["manifest"])):
        if not value.get(key) or Path(value[key]).resolve() != expected:
            raise ValueError(f"Metrics {key} path does not match this suite: {run['metrics']}")
    if int(value.get("n_samples", 0)) <= 0:
        raise ValueError(f"Evaluation contains no samples: {run['metrics']}")
    return value


def check_outputs(args, runs: list[dict]) -> None:
    for run in runs:
        directory = run["checkpoint"].parents[1]
        if run["metrics"].exists():
            if not args.resume_suite:
                raise FileExistsError(f"Evaluation already exists: {run['metrics']}; use --resume-suite or a new --run-root.")
            read_metrics(run)
        if directory.exists() and any(directory.iterdir()):
            if not args.resume_suite:
                raise FileExistsError(f"Training output already exists: {directory}; use --resume-suite or a new --run-root.")
            if not run["checkpoint"].is_file() and not run["metrics"].is_file():
                raise ValueError(f"Interrupted training found in {directory}. Optimizer resume is not supported; choose a new --run-root.")
            if run["checkpoint"].is_file() and any(not (directory / key).is_file() for key in ("config.yaml", "dataset_statistics.json")):
                raise FileNotFoundError(f"Checkpoint metadata is incomplete in {directory}")


def launch(command: list[str], env: dict[str, str], log: Path) -> float:
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with log.open("a") as output:
        output.write("\n$ " + display(command, env) + "\n")
        with subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1,
                              start_new_session=True) as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    output.write(line)
                    output.flush()
                status = process.wait()
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                raise
        if status:
            raise RuntimeError(f"Command failed with exit code {status}; see {log}")
    return time.monotonic() - start


def summarize(args, runs: list[dict], manifest: dict) -> None:
    fields = ["run", "condition", "status", "action_horizon", "goal_conditioning", "time_conditioning",
              "time_definition", "split_mode", "goal_source",
              "training_steps", "seed", "global_batch", "train_anchor_count", "eval_anchor_count",
              "supervised_action_positions", *METRICS, "checkpoint", "metrics_file"]
    path = args.run_root / "summary.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = dict(run=run["name"], condition=run["condition"], status="prepared", action_horizon=run["horizon"],
                       goal_conditioning=run["goal"], time_conditioning=run["time"],
                       time_definition=TIME_DEFINITION, split_mode=manifest["split_mode"],
                       goal_source=manifest["goal_source"], training_steps=args.steps, seed=args.seed,
                       global_batch=args.batch_size * len(args.gpu_ids),
                       train_anchor_count=len(manifest["train_anchors"]),
                       eval_anchor_count=len(manifest["eval_anchors"]),
                       supervised_action_positions=args.steps * args.batch_size * len(args.gpu_ids) * run["horizon"],
                       checkpoint=str(run["checkpoint"]),
                       metrics_file=str(run["metrics"]))
            if run["metrics"].is_file():
                result = read_metrics(run)
                row.update({key: result.get(key, "") for key in METRICS})
                row["status"] = "evaluated"
            elif run["checkpoint"].is_file():
                row["status"] = "trained"
            writer.writerow(row)


def main() -> int:
    args = parser().parse_args()
    args.source_dataset, args.run_root, args.config = map(absolute, (args.source_dataset, args.run_root, args.config))
    args.horizons = sorted(set(args.horizons))
    args.conditions = [name for name in ("baseline", "goal", "goal_time") if name in args.conditions]
    args.gpu_ids = [part.strip() for part in args.gpus.split(",")]
    if not args.gpu_ids or any(not part.isdigit() for part in args.gpu_ids) or len(set(args.gpu_ids)) != len(args.gpu_ids):
        raise ValueError("--gpus must contain distinct physical GPU indices, e.g. 0,2,3,7")
    args.gpus = ",".join(args.gpu_ids)
    formal = args.profile == "formal"
    args.steps = args.steps if args.steps is not None else (150000 if formal else 50)
    args.warmup = args.warmup if args.warmup is not None else (5000 if formal else 5)
    args.batch_size = args.batch_size if args.batch_size is not None else (8 if formal else 1)
    args.eval_samples = args.eval_samples if args.eval_samples is not None else (200 if formal else 50)
    args.first_k = None if args.all_frames else args.first_k
    if args.seed < 0:
        raise ValueError("--seed must be nonnegative")
    if not 0 <= args.warmup < args.steps or not 0 < args.eval_fraction < 1:
        raise ValueError("Require 0 <= warmup < steps and 0 < eval-fraction < 1; set --warmup explicitly for very short tests.")
    names = []
    for condition in args.conditions:
        horizons = args.horizons if condition != "baseline" or args.matched_baselines else [max(args.horizons)]
        names.extend((f"{condition}_h{h}", h, condition) for h in horizons)
    runs = [dict(name=name, horizon=h, condition=condition,
                 goal=condition != "baseline", time=condition == "goal_time",
                 checkpoint=args.run_root / "runs" / name / "final_model/pytorch_model.pt",
                 metrics=args.run_root / "metrics" / f"{name}.json") for name, h, condition in names]
    unknown = set(args.only or ()) - {run["name"] for run in runs}
    if unknown:
        raise ValueError(f"Unknown --only names: {sorted(unknown)}; available: {[r['name'] for r in runs]}")
    selected = [run for run in runs if not args.only or run["name"] in args.only]
    settings = {key: getattr(args, key) for key in ("profile", "steps", "warmup", "batch_size", "first_k", "seed", "eval_fraction", "eval_samples", "horizons", "conditions", "matched_baselines", "gpus")}
    settings.update(source_dataset=str(args.source_dataset), config=str(args.config),
                    config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), code_sha256=code_digest())
    suite_file = args.run_root / "suite.json"
    if suite_file.exists():
        if json.loads(suite_file.read_text()) != settings:
            raise ValueError(f"Suite settings/code differ from {suite_file}; choose a new --run-root.")
    elif args.run_root.exists() and any(args.run_root.iterdir()):
        raise ValueError(f"Nonempty run root has no suite manifest: {args.run_root}; choose a new directory.")
    else:
        save_json(suite_file, settings)
    from ablation_data import load_manifest, prepare_splits

    manifest_path = Path(prepare_splits(args.source_dataset, args.run_root / "prepared", seed=args.seed,
                                      eval_fraction=args.eval_fraction, max_horizon=max(args.horizons),
                                      first_k=args.first_k, smoke=not formal))
    manifest = load_manifest(manifest_path)
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    for run in runs:
        run["manifest"] = manifest_path
        run["expected_metrics"] = {
            "action_horizon": run["horizon"], "goal_conditioning": run["goal"],
            "time_conditioning": run["time"],
            "time_definition": TIME_DEFINITION, "time_units": "seconds",
            "time_source": "expert_frame_gap_seconds",
            "target_pose_dim": 14 if run["goal"] else 0, "state_conditioning": True,
            "manifest_sha256": manifest_sha, "seed": args.seed, "noise_seed": 123,
            "split_mode": manifest["split_mode"], "goal_source": manifest["goal_source"],
            "norm_source": "train", "n_samples": min(args.eval_samples, len(manifest["eval_anchors"])),
        }
    configs = configurations(args, manifest, runs)
    train_env, eval_env = environment(args.gpus), environment(args.gpu_ids[0])
    plan = []
    print(f"Split: {manifest['split_mode']}; global observation batch={args.batch_size * len(args.gpu_ids)}; FM repeat=2.")
    print(f"Eligible input anchors: train={len(manifest['train_anchors'])}, eval={len(manifest['eval_anchors'])}.")
    if manifest["split_mode"] == "train_replay_smoke":
        print("SMOKE ONLY: evaluation reuses training data; these are not held-out results.")
    for run in selected:
        train, evaluate = commands(args, run, configs[run["name"]])
        print(f"\n[{run['name']}]\nTRAIN: {display(train, train_env)}\nEVAL:  {display(evaluate, eval_env)}")
        plan.append(dict(name=run["name"], train=train, evaluate=evaluate,
                         train_cuda=args.gpus, eval_cuda=args.gpu_ids[0]))
    save_json(args.run_root / "commands.json", plan)
    if args.execute:
        check_outputs(args, selected)
        if any(not run["metrics"].is_file() for run in selected):
            validate_weights(configs[selected[0]["name"]])
        for run in selected:
            if run["metrics"].is_file():
                print(f"Skipping completed evaluation: {run['name']}")
                continue
            train, evaluate = commands(args, run, configs[run["name"]])
            metadata = dict(run=run["name"], condition=run["condition"], configuration=str(configs[run["name"]]),
                            manifest=str(manifest_path), steps=args.steps, horizon=run["horizon"],
                            goal_conditioning=run["goal"], time_conditioning=run["time"],
                            time_definition=TIME_DEFINITION, time_units="seconds",
                            time_source="expert_frame_gap_seconds", global_batch=args.batch_size * len(args.gpu_ids),
                            seed=args.seed, split_mode=manifest["split_mode"],
                            train_anchor_count=len(manifest["train_anchors"]),
                            eval_anchor_count=len(manifest["eval_anchors"]),
                            supervised_action_positions=args.steps * args.batch_size * len(args.gpu_ids) * run["horizon"],
                            train_command=train, eval_command=evaluate)
            if not run["checkpoint"].is_file():
                metadata["training_wall_seconds"] = launch(train, train_env, args.run_root / "logs" / f"{run['name']}.train.log")
            if not run["checkpoint"].is_file():
                raise FileNotFoundError(f"Training returned without the expected checkpoint: {run['checkpoint']}")
            metadata["evaluation_wall_seconds"] = launch(evaluate, eval_env, args.run_root / "logs" / f"{run['name']}.eval.log")
            read_metrics(run)
            save_json(args.run_root / "metadata" / f"{run['name']}.json", metadata)
            summarize(args, runs, manifest)
    summarize(args, runs, manifest)
    print(f"\n{'Finished' if args.execute else 'Prepared only; add --execute to run'}. Summary: {args.run_root / 'summary.csv'}")
    return 0


def terminate(signum, _frame) -> None:
    # Raise through launch() so background stop also cleans its process group.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("Interrupted; owned subprocesses have been stopped.", file=sys.stderr)
        raise SystemExit(130)
