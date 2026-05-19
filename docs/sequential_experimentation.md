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

## Running A Round

From the project root:

```bash
python src/run_experiment_round.py --config config/pong_dqn_experiments.json --round round_1_preprocessing
```

For a quick smoke test of a round:

```bash
python src/run_experiment_round.py --config config/pong_dqn_experiments.json --round round_1_preprocessing --max-frames 5000 --benchmark-episodes 5
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
