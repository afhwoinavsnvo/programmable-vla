#!/usr/bin/env bash
# Three input conditions at H=1,2,4,8,16, each with its own matched PI baseline.
# Default: prepare only. Add --execute to run, --download-model to fetch weights.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../../.." && pwd)"
python_bin="${STARVLA_PYTHON:-/var/tmp/jliu3375/miniconda3/envs/starVLA/bin/python}"
config_path="$script_dir/starvla_cotrain_robotwin_abs.yaml"
download_model=false
forwarded=()

while (( $# )); do
  case "$1" in
    --download-model) download_model=true; shift ;;
    --config)
      if (( $# < 2 )); then
        echo "--config requires a YAML path." >&2
        exit 2
      fi
      config_path="$2"
      forwarded+=("$1" "$2")
      shift 2
      ;;
    --config=*) config_path="${1#*=}"; forwarded+=("$1"); shift ;;
    *) forwarded+=("$1"); shift ;;
  esac
done

if [[ ! -x "$python_bin" ]]; then
  echo "Set STARVLA_PYTHON to the Python executable in your starVLA environment." >&2
  exit 1
fi

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# This H100 node fails in transport/nvls.cc with CUDA 401. Disabling NVLink
# SHARP fixes the tested four-GPU collectives while keeping ordinary P2P.
# An explicit caller value can override this node-specific workaround.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
echo "NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE"

if [[ "$download_model" == true ]]; then
  "$python_bin" - "$config_path" "$script_dir" <<'PY'
from pathlib import Path
import sys

from omegaconf import OmegaConf

sys.path.insert(0, sys.argv[2])
from run_ablation import validate_weights

config = Path(sys.argv[1]).resolve()
try:
    validate_weights(config)
except FileNotFoundError:
    from huggingface_hub import snapshot_download

    directory = Path(str(OmegaConf.load(config).framework.qwenvl.base_vlm)).resolve()
    if directory.name != "Qwen3-VL-4B-Instruct":
        raise SystemExit("Automatic download supports Qwen3-VL-4B-Instruct only; prepare your configured model manually.")
    print(f"Downloading Qwen/Qwen3-VL-4B-Instruct to {directory}", flush=True)
    snapshot_download(repo_id="Qwen/Qwen3-VL-4B-Instruct", local_dir=str(directory), max_workers=4)
    validate_weights(config)
else:
    print("Configured VLM weights are already available.", flush=True)
PY
fi

exec "$python_bin" "$script_dir/run_ablation.py" \
  --conditions baseline goal goal_time \
  --matched-baselines \
  --run-root /var/tmp/jliu3375/robotwin_three_methods_smoke \
  "${forwarded[@]}"
