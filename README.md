# C5 Optional RL - DQN on Pong

Authors: Marina Rosell Murillo and Gerard Asbert Marcos

This project implements, trains, evaluates, and compares Deep Q-Network (DQN)
agents on the Gymnasium Atari `ALE/Pong-v5` environment. It includes:

- a PyTorch DQN implementation;
- Atari preprocessing wrappers;
- JSON experiment configurations;
- SLURM scripts for all experiment rounds;
- 100-episode benchmark evaluation;
- best/worst episode GIF export;
- notebooks for visualizing metrics and videos.

The final selected models are stored in `models/`, and all benchmark metrics
and GIF paths are stored in `results/`.

## Repository Structure

```text
C5_Optional_RL/
├── config/pong_dqn_experiments.json
├── docs/
├── models/
├── notebooks/
│   ├── example_code_baseline_train_test.ipynb
│   └── visualize_experiment_results.ipynb
├── results/
├── scripts/
├── src/
│   ├── train_pong_dqn.py
│   ├── test_pong_dqn.py
│   └── benchmark_pong_dqn.py
├── videos/benchmarks/
└── requirements.txt
```

## Setup

Recommended Python version: Python 3.11.

From the repository root:

```bash
cd C5_Optional_RL
pip install -r requirements.txt
```

On the cluster, we used the `rl` Conda environment:

```bash
/ghome/group02/miniconda3/envs/rl/bin/python
```

The SLURM scripts require CUDA by default. Before launching a long run, check
that PyTorch can see the GPU:

```bash
sbatch scripts/debug_cuda.sh
```

If the debug job reports `torch.cuda.is_available(): False`, submit jobs with
the CUDA-enabled Python executable:

```bash
PYTHON_BIN=/ghome/group02/miniconda3/envs/rl/bin/python sbatch scripts/round_1_preprocessing/r01_skip_current.sh
```

## Part 1: Training a DQN Agent

### Train One Experiment Locally

To train one configured experiment without SLURM:

```bash
python -u src/train_pong_dqn.py \
  --config config/pong_dqn_experiments.json \
  --experiment r06_best_seed_1
```

The model checkpoint is saved to the path specified by the experiment in
`config/pong_dqn_experiments.json`, for example:

```text
models/r06_best_seed_1.dat
```

Training metrics are saved in:

```text
results/<experiment>_train_metrics.json
```

These metrics include wall-clock training time, number of frames, number of
episodes, best/final moving-average reward, and convergence threshold frames.

### Train One Experiment with SLURM

Each SLURM script trains one experiment and then runs the 100-episode benchmark.
For example:

```bash
sbatch scripts/round_1_preprocessing/r01_skip_current.sh
```

To use a specific Python executable:

```bash
PYTHON_BIN=/ghome/group02/miniconda3/envs/rl/bin/python \
sbatch scripts/round_1_preprocessing/r01_skip_current.sh
```

To shorten the benchmark while testing the queue setup:

```bash
BENCHMARK_EPISODES=20 \
sbatch scripts/round_1_preprocessing/r01_skip_current.sh
```

### Sequential Experiment Workflow

The experiments were organized into six rounds:

1. `round_1_preprocessing`: frame skipping and sticky actions.
2. `round_2_learning_rate`: learning rate.
3. `round_3_replay`: replay buffer size.
4. `round_4_epsilon_decay`: exploration schedule.
5. `round_5_target_sync`: target network synchronization.
6. `round_6_confirmation_seeds`: final configuration with different seeds.

Round 1 scripts:

```bash
sbatch scripts/round_1_preprocessing/r01_skip_current.sh
sbatch scripts/round_1_preprocessing/r01_skip_no_extra_wrapper.sh
sbatch scripts/round_1_preprocessing/r01_skip_classic_dqn.sh
sbatch scripts/round_1_preprocessing/r01_skip_no_skip_high_control.sh
sbatch scripts/round_1_preprocessing/r01_skip_current_repeat_prob0.sh
sbatch scripts/round_1_preprocessing/r01_skip_no_extra_wrapper_repeat_prob0.sh
sbatch scripts/round_1_preprocessing/r01_skip_classic_dqn_repeat_prob0.sh
sbatch scripts/round_1_preprocessing/r01_skip_no_skip_high_control_repeat_prob0.sh
```

Later rounds are available under:

```text
scripts/round_2_learning_rate/
scripts/round_3_replay/
scripts/round_4_epsilon_decay/
scripts/round_5_target_sync/
scripts/round_6_confirmation_seeds/
```

The full sequential workflow is described in:

```text
docs/sequential_experimentation.md
```

## Part 2: Testing the Trained Agent

### Run a 100-Episode Benchmark

To evaluate a trained model over 100 independent episodes and save only the
best and worst GIFs:

```bash
python -u src/benchmark_pong_dqn.py \
  --config config/pong_dqn_experiments.json \
  --experiment r06_best_seed_1 \
  --prefix r06_best_seed_1
```

This creates:

```text
results/r06_best_seed_1_test_metrics.json
videos/benchmarks/r06_best_seed_1_best_episode_001_reward_21.gif
videos/benchmarks/r06_best_seed_1_worst_episode_001_reward_21.gif
```

The metrics file reports the average reward, standard deviation, minimum
reward, maximum reward, and paths to the best/worst GIFs.

### Record GIFs for All Evaluated Episodes

To save one GIF per episode instead of only the best and worst:

```bash
python -u src/test_pong_dqn.py \
  --config config/pong_dqn_experiments.json \
  --experiment r06_best_seed_1 \
  --episodes 3 \
  --record-mode all \
  --output-dir videos \
  --prefix r06_best_seed_1
```

## Visualizing Results and GIFs

Open the notebook:

```text
notebooks/visualize_experiment_results.ipynb
```

This notebook loads every `results/*_metrics.json` file, compares rewards,
plots training/evaluation summaries, and displays the best and worst episode
GIFs for all experiments. This is the easiest place to inspect all generated
animations together.

The generated benchmark GIFs are also stored directly in:

```text
videos/benchmarks/
```

## Example GIFs

The first model below corresponds to an early Round 1 configuration. The final
model corresponds to the selected configuration evaluated with seed 42.

| Model | Best episode | Worst episode |
|---|---|---|
| Early Round 1 model (`r01_skip_current`) | ![](videos/benchmarks/r01_skip_current_best_episode_048_reward_16.gif) | ![](videos/benchmarks/r01_skip_current_worst_episode_005_reward_-15.gif) |
| Final model (`r06_best_seed_1`) | ![](videos/benchmarks/r06_best_seed_1_best_episode_001_reward_21.gif) | ![](videos/benchmarks/r06_best_seed_1_worst_episode_001_reward_21.gif) |

For the final deterministic model, the best and worst episodes both obtain
reward `21`, so the two GIFs are successful games. This is expected because the
final evaluation uses `repeat_action_probability=0.0`.

## Final Model

The final selected configuration is represented by the confirmation-seed models:

```text
models/r06_best_seed_1.dat
models/r06_best_seed_2.dat
models/r06_best_seed_3.dat
```

The main final model used in the report is:

```text
models/r06_best_seed_1.dat
```

It can be benchmarked with:

```bash
python -u src/benchmark_pong_dqn.py \
  --config config/pong_dqn_experiments.json \
  --experiment r06_best_seed_1 \
  --prefix r06_best_seed_1
```

## Outputs to Submit

The assignment requires source code, trained models, requirements, README
instructions, and generated animations. In this repository:

- source code: `src/`
- training/evaluation scripts: `scripts/`
- experiment configuration: `config/pong_dqn_experiments.json`
- trained models: `models/`
- metrics: `results/`
- best/worst GIFs: `videos/benchmarks/`
- visualization notebook: `notebooks/visualize_experiment_results.ipynb`
- requirements: `requirements.txt`
- report helper text and figures: `docs/`
