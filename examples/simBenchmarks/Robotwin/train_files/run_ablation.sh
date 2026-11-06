#!/usr/bin/env bash
# PI baseline / goal-pose / goal-and-time experiments and offline action metrics.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../../.." && pwd)"
python_bin="${STARVLA_PYTHON:-/var/tmp/jliu3375/miniconda3/envs/starVLA/bin/python}"

if [[ ! -x "$python_bin" ]]; then
  echo "Set STARVLA_PYTHON to the Python executable in your starVLA environment." >&2
  exit 1
fi

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" "$script_dir/run_ablation.py" "$@"
