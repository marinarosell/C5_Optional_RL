#!/usr/bin/env bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 4096
#SBATCH --gres gpu:1
#SBATCH -o logs/%x_%u_%j.out
#SBATCH -e logs/%x_%u_%j.err

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs models results videos/benchmarks

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="config/pong_dqn_experiments.json"
EXPERIMENT="r01_skip_classic_dqn"
BENCHMARK_EPISODES="${BENCHMARK_EPISODES:-100}"

export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=300

echo "Host: $(hostname)"
echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Experiment: $EXPERIMENT"
echo "Python: $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

"$PYTHON_BIN" -u src/train_pong_dqn.py \
  --config "$CONFIG" \
  --experiment "$EXPERIMENT" \
  --use-wandb

"$PYTHON_BIN" -u src/benchmark_pong_dqn.py \
  --config "$CONFIG" \
  --experiment "$EXPERIMENT" \
  --episodes "$BENCHMARK_EPISODES" \
  --prefix "$EXPERIMENT"
