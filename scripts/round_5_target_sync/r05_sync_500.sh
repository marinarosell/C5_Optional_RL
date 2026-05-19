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

EXPERIMENT="r05_sync_500"

# TODO: fill with the best parameters from rounds 1, 2, 3, and 4.
BEST_ENV_FRAMESKIP=""
BEST_WRAPPER_SKIP=""
BEST_LEARNING_RATE=""
BEST_REPLAY_SIZE=""
BEST_REPLAY_START_SIZE=""
BEST_EPS_DECAY=""

REQUIRED_BEST_PARAMS=(BEST_ENV_FRAMESKIP BEST_WRAPPER_SKIP BEST_LEARNING_RATE BEST_REPLAY_SIZE BEST_REPLAY_START_SIZE BEST_EPS_DECAY)
TRAIN_OVERRIDES=(--env-frameskip "$BEST_ENV_FRAMESKIP" --wrapper-skip "$BEST_WRAPPER_SKIP" --learning-rate "$BEST_LEARNING_RATE" --replay-size "$BEST_REPLAY_SIZE" --replay-start-size "$BEST_REPLAY_START_SIZE" --eps-decay "$BEST_EPS_DECAY")
BENCHMARK_OVERRIDES=(--env-frameskip "$BEST_ENV_FRAMESKIP" --wrapper-skip "$BEST_WRAPPER_SKIP")

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_run_slurm_experiment.sh"
