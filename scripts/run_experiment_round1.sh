#!/usr/bin/env bash
# Submit one SLURM job per experiment in a configured round.
# Run from the login node with:
#   bash scripts/run_experiment_round1.sh
#
# You may also submit this launcher with sbatch, but it does not need a GPU
# itself because it only submits the real experiment jobs.

set -euo pipefail

CONFIG=${1:-config/pong_dqn_experiments.json}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
ROUND_NAME="${ROUND_NAME:-round_1_preprocessing}"
BENCHMARK_EPISODES="${BENCHMARK_EPISODES:-100}"
SLURM_JOB_SCRIPT="${SLURM_JOB_SCRIPT:-scripts/slurm_experiment_job.sh}"

if [[ "$CONFIG" != /* && -f "$CONFIG" ]]; then
  CONFIG="$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")"
fi

cd "$ROOT_DIR"
mkdir -p logs models results videos/benchmarks

echo "Project root: $ROOT_DIR"
echo "Running experiment round: $ROUND_NAME"
echo "Using config: $CONFIG"
echo "Using Python: $PYTHON_BIN"
echo "Submitting one SLURM job per experiment with: $SLURM_JOB_SCRIPT"

"$PYTHON_BIN" src/run_experiment_round.py \
  --config "$CONFIG" \
  --round "$ROUND_NAME" \
  --benchmark-episodes "$BENCHMARK_EPISODES" \
  --mode sbatch \
  --script "$SLURM_JOB_SCRIPT" \
  --python-bin "$PYTHON_BIN" \
  --job-name-prefix "$ROUND_NAME" \
  --logs-dir logs \
  --use-wandb
