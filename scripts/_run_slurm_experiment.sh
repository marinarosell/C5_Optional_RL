#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-config/pong_dqn_experiments.json}"
BENCHMARK_EPISODES="${BENCHMARK_EPISODES:-100}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_MODE="${WANDB_MODE:-online}"

export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=300
export REQUIRE_CUDA=1
export WANDB_MODE

require_value() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    echo "ERROR: Fill $name in $0 before submitting this experiment." >&2
    exit 2
  fi
}

if [[ -z "${EXPERIMENT:-}" ]]; then
  echo "ERROR: EXPERIMENT is not set in $0." >&2
  exit 2
fi

for required_name in "${REQUIRED_BEST_PARAMS[@]:-}"; do
  require_value "$required_name"
done

echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Experiment: $EXPERIMENT"
echo "Python: $PYTHON_BIN"
echo "Resolved Python: $(command -v "$PYTHON_BIN" || true)"
"$PYTHON_BIN" --version
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "nvidia-smi:"
nvidia-smi || true
echo "PyTorch CUDA preflight:"
"$PYTHON_BIN" - <<'PYTORCH_CHECK'
import sys
import torch
print("sys.executable:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
print("torch.cuda.device_count():", torch.cuda.device_count())
if torch.cuda.is_available():
    print("torch.cuda.get_device_name(0):", torch.cuda.get_device_name(0))
else:
    raise SystemExit("ERROR: PyTorch cannot see CUDA. Check PYTHON_BIN / CUDA PyTorch installation / SLURM GPU allocation.")
PYTORCH_CHECK

if [[ "$USE_WANDB" == "1" ]]; then
  echo "W&B preflight:"
  echo "WANDB_MODE: ${WANDB_MODE:-unset}"
  echo "WANDB_API_KEY present: $(if [[ -n "${WANDB_API_KEY:-}" ]]; then echo yes; else echo no; fi)"
  "$PYTHON_BIN" - <<'WANDB_CHECK'
import os
import sys

try:
    import wandb
except Exception as exc:
    raise SystemExit(f"ERROR: wandb import failed: {exc!r}")

print("wandb.__version__:", wandb.__version__)
print("WANDB_MODE:", os.environ.get("WANDB_MODE", "<unset>"))
if os.environ.get("WANDB_MODE") == "disabled":
    raise SystemExit("ERROR: WANDB_MODE=disabled, runs will not be saved.")
WANDB_CHECK
fi

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_ARGS=(--use-wandb)
fi

"$PYTHON_BIN" -u src/train_pong_dqn.py \
  --config "$CONFIG" \
  --experiment "$EXPERIMENT" \
  "${TRAIN_OVERRIDES[@]:-}" \
  "${WANDB_ARGS[@]}"

"$PYTHON_BIN" -u src/benchmark_pong_dqn.py \
  --config "$CONFIG" \
  --experiment "$EXPERIMENT" \
  --episodes "$BENCHMARK_EPISODES" \
  --prefix "$EXPERIMENT" \
  "${WANDB_ARGS[@]}" \
  "${BENCHMARK_OVERRIDES[@]:-}"
