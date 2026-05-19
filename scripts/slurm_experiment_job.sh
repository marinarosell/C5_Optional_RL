#!/usr/bin/env bash
#SBATCH -n 4
#SBATCH -N 1
#SBATCH -t 0-24:00
#SBATCH -p mlow
#SBATCH -q masterlow
#SBATCH --mem 4096
#SBATCH --gres gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs models results videos/benchmarks

# Customize this block for your cluster environment.
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

echo "Running on host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "Command: $1"

bash -lc "$1"
