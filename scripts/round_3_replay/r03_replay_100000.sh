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

EXPERIMENT="r03_replay_100000"

# TODO: fill with the best parameters from rounds 1 and 2.
BEST_ENV_FRAMESKIP=""
BEST_WRAPPER_SKIP=""
BEST_LEARNING_RATE=""

REQUIRED_BEST_PARAMS=(BEST_ENV_FRAMESKIP BEST_WRAPPER_SKIP BEST_LEARNING_RATE)
TRAIN_OVERRIDES=(--env-frameskip "$BEST_ENV_FRAMESKIP" --wrapper-skip "$BEST_WRAPPER_SKIP" --learning-rate "$BEST_LEARNING_RATE")
BENCHMARK_OVERRIDES=(--env-frameskip "$BEST_ENV_FRAMESKIP" --wrapper-skip "$BEST_WRAPPER_SKIP")

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_run_slurm_experiment.sh"
