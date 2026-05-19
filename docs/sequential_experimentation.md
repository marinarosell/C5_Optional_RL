# Sequential Experimentation Plan

The goal is to change one parameter family at a time, benchmark all candidates
in that round, then use the best candidate as the baseline for the next round.

## Selection Criteria

Prefer the experiment with the best balance of:

- Higher 100-episode average reward
- Lower 100-episode reward standard deviation
- Faster convergence, measured by frames to reach reward thresholds
- Lower wall-clock training time
- Reasonable best/worst GIF behavior

If two candidates have similar benchmark rewards, prefer the one that converges
faster or has lower variance.

## Rounds

1. `round_1_preprocessing`
   - Varies `env_frameskip` and `wrapper_skip`.
   - Tests whether the custom wrapper should add extra action repeats on top of
     `ALE/Pong-v5`.

2. `round_2_learning_rate`
   - Varies `learning_rate`.
   - Keep the best preprocessing setup from round 1.

3. `round_3_replay`
   - Varies `replay_size` and `replay_start_size`.
   - Keep the best preprocessing and learning rate.

4. `round_4_epsilon_decay`
   - Varies `eps_decay`.
   - Measures the exploration schedule's effect on convergence speed and final
     policy quality.

5. `round_5_target_sync`
   - Varies `sync_target_frames`.
   - Measures target-network stability versus adaptation speed.

6. `round_6_confirmation_seeds`
   - Re-runs the final selected setup multiple times.
   - Use this to check whether the result is robust or just a lucky run.

## Running Round 1 On SLURM

From the project root:

```bash
sbatch scripts/round_1_preprocessing/r01_skip_current.sh
sbatch scripts/round_1_preprocessing/r01_skip_no_extra_wrapper.sh
sbatch scripts/round_1_preprocessing/r01_skip_classic_dqn.sh
sbatch scripts/round_1_preprocessing/r01_skip_no_skip_high_control.sh
```

Each script trains one experiment and then runs the benchmark. To reduce
benchmark cost while testing the queue setup:

```bash
BENCHMARK_EPISODES=5 sbatch scripts/round_1_preprocessing/r01_skip_current.sh
```

To use a specific Python environment:

```bash
PYTHON_BIN=/path/to/env/bin/python sbatch scripts/round_1_preprocessing/r01_skip_current.sh
```

## Running Later Rounds

For rounds 2 through 6, each script contains a TODO block with empty fields.
Fill those fields using the best parameters from the previous rounds before
submitting the job. If a required field is still empty, the script stops before
training.

Round 2, learning rate:

```bash
sbatch scripts/round_2_learning_rate/r02_lr_2_5e_5.sh
sbatch scripts/round_2_learning_rate/r02_lr_5e_5.sh
sbatch scripts/round_2_learning_rate/r02_lr_1e_4.sh
sbatch scripts/round_2_learning_rate/r02_lr_2_5e_4.sh
```

Round 3, replay buffer:

```bash
sbatch scripts/round_3_replay/r03_replay_10000.sh
sbatch scripts/round_3_replay/r03_replay_50000.sh
sbatch scripts/round_3_replay/r03_replay_100000.sh
```

Round 4, epsilon decay:

```bash
sbatch scripts/round_4_epsilon_decay/r04_eps_slow.sh
sbatch scripts/round_4_epsilon_decay/r04_eps_default.sh
sbatch scripts/round_4_epsilon_decay/r04_eps_fast.sh
```

Round 5, target-network sync:

```bash
sbatch scripts/round_5_target_sync/r05_sync_500.sh
sbatch scripts/round_5_target_sync/r05_sync_1000.sh
sbatch scripts/round_5_target_sync/r05_sync_5000.sh
```

Round 6, confirmation seeds:

```bash
sbatch scripts/round_6_confirmation_seeds/r06_best_seed_1.sh
sbatch scripts/round_6_confirmation_seeds/r06_best_seed_2.sh
sbatch scripts/round_6_confirmation_seeds/r06_best_seed_3.sh
```

Example after round 1: if `r01_skip_no_extra_wrapper` wins, edit every round 2
script like this:

```bash
BEST_ENV_FRAMESKIP="4"
BEST_WRAPPER_SKIP="1"
```

## Running One Experiment Without SLURM

```bash
python -u src/train_pong_dqn.py --config config/pong_dqn_experiments.json --experiment r01_skip_current --use-wandb
python -u src/benchmark_pong_dqn.py --config config/pong_dqn_experiments.json --experiment r01_skip_current --episodes 100 --prefix r01_skip_current
```

## Promoting A Winner

After each round:

1. Open `notebooks/visualize_experiment_results.ipynb`.
2. Compare benchmark reward, standard deviation, training time, and convergence
   thresholds.
3. Pick the winner.
4. Copy that winner's fixed settings into the next round's experiments in
   `config/pong_dqn_experiments.json`.
5. Run the next round.

For example, if `r01_skip_no_extra_wrapper` wins round 1, keep:

```json
"env_frameskip": 4,
"wrapper_skip": 1
```

fixed in all round 2 learning-rate experiments.
