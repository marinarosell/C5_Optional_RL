# C5_Optional_RL

Authors: Marina Rosell Murillo and Gerard Asbert Marcos

This project trains, evaluates, and compares Deep Q-Network agents on the
Gymnasium Atari Pong environment. It includes a configurable training script,
experiment definitions, GIF generation for trained policies, a 100-episode
benchmark workflow, and notebooks for training plus visualizing experiment
results.

## Pong DQN

Recommended Python version: Python 3.11.

Install dependencies:

```bash
cd C5_Optional_RL
pip install -r requirements.txt
```

Train one configured experiment:

```bash
python src/train_pong_dqn.py --config config/pong_dqn_experiments.json --experiment baseline
```

Run a full sequential experimentation round:

```bash
python src/run_experiment_round.py --config config/pong_dqn_experiments.json --round round_1_preprocessing
```

The configured rounds change one parameter family at a time. After each round,
choose the best experiment using the benchmark average reward, standard
deviation, training time, and convergence thresholds. Then copy that winner's
settings into the next round's candidate experiments before launching the next
round. See `docs/sequential_experimentation.md` for the full workflow.

Record GIFs from a trained model:

```bash
python src/test_pong_dqn.py --config config/pong_dqn_experiments.json --experiment baseline --episodes 3
```

GIFs are saved in `videos/` by default.

Run the standard 100-episode benchmark and save only the best/worst GIFs:

```bash
python src/benchmark_pong_dqn.py --config config/pong_dqn_experiments.json --experiment baseline
```

Benchmark metrics are saved in `results/<experiment>_test_metrics.json`. Open
`notebooks/visualize_experiment_results.ipynb` to compare all benchmarked
experiments and view their best/worst videos.

Training metrics are saved in `results/<run_name>_train_metrics.json`, including
wall-clock training time, best/final moving-average reward, and the frame where
the moving-average reward first crosses the configured thresholds.
