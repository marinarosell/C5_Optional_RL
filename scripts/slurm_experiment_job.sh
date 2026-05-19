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
export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=300

# Customize this block for your cluster environment.
if [[ -n "${CONDA_ENV_NAME:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
fi

echo "Running on host: $(hostname)"
echo "SLURM job id: ${SLURM_JOB_ID:-unknown}"
echo "SLURM submit dir: ${SLURM_SUBMIT_DIR}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "PATH: ${PATH}"
echo "Python executable:"
command -v python3 || true
python3 --version || true
echo "GPU status before command:"
nvidia-smi || true
echo "PyTorch CUDA check:"
python3 - <<'PY' || true
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("cuda_device_count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("cuda_device_name:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_cuda_check_failed:", repr(exc))
PY
echo "Command: $1"

echo "Starting command at $(date)"
bash -lc "$1"
echo "Finished command at $(date)"
echo "GPU status after command:"
nvidia-smi || true
