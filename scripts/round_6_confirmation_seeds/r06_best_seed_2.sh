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

EXPERIMENT="r06_best_seed_2"

# TODO: fill with the final best parameters from rounds 1 through 5.
BEST_ENV_FRAMESKIP="1"
BEST_WRAPPER_SKIP="4"
BEST_LEARNING_RATE="0.0001"
BEST_REPLAY_SIZE="10000"
BEST_REPLAY_START_SIZE="10000"
BEST_EPS_DECAY="0.99995"
BEST_SYNC_TARGET_FRAMES="1000"

REQUIRED_BEST_PARAMS=(BEST_ENV_FRAMESKIP BEST_WRAPPER_SKIP BEST_LEARNING_RATE BEST_REPLAY_SIZE BEST_REPLAY_START_SIZE BEST_EPS_DECAY BEST_SYNC_TARGET_FRAMES)
TRAIN_OVERRIDES=(--env-frameskip "$BEST_ENV_FRAMESKIP" --wrapper-skip "$BEST_WRAPPER_SKIP" --learning-rate "$BEST_LEARNING_RATE" --replay-size "$BEST_REPLAY_SIZE" --replay-start-size "$BEST_REPLAY_START_SIZE" --eps-decay "$BEST_EPS_DECAY" --sync-target-frames "$BEST_SYNC_TARGET_FRAMES")
BENCHMARK_OVERRIDES=(--env-frameskip "$BEST_ENV_FRAMESKIP" --wrapper-skip "$BEST_WRAPPER_SKIP")

source "scripts/_run_slurm_experiment.sh"
