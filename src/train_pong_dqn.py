#!/usr/bin/env python3
"""Train a DQN agent on the ALE/Pong-v5 environment.

This script extracts the notebook training pipeline into a reusable CLI script
with configurable hyperparameters and optional Weights & Biases logging.
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import datetime
import collections
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import gymnasium as gym
import ale_py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def require_wandb_available():
    """Fail loudly when W&B logging was requested but cannot run."""
    if not _HAS_WANDB:
        raise RuntimeError(
            'Weights & Biases logging was requested with --use-wandb, but the '
            'wandb package is not installed in this Python environment. Install '
            'it with `pip install wandb` or run with USE_WANDB=0.'
        )
    if os.environ.get('WANDB_MODE') == 'disabled':
        raise RuntimeError('WANDB_MODE=disabled, so this run will not be saved to W&B.')


@dataclass
class TrainConfig:
    """Training and evaluation settings for one DQN experiment."""

    env_name: str = 'ALE/Pong-v5'
    env_frameskip: int = None
    wrapper_skip: int = 4
    seed: int = None
    gamma: float = 0.99
    batch_size: int = 32
    learning_rate: float = 1e-4
    replay_size: int = 10000
    replay_start_size: int = 10000
    sync_target_frames: int = 1000
    eps_start: float = 1.0
    eps_decay: float = 0.999985
    eps_min: float = 0.02
    max_frames: int = 500000
    mean_reward_bound: float = 19.0
    reward_average_size: int = 10
    convergence_reward_thresholds: list = field(default_factory=lambda: [-15.0, -10.0, -5.0, 0.0])
    save_path: str = 'ALE_Pong_v5_dqn_solution.dat'
    training_metrics_path: str = ''
    use_wandb: bool = False
    wandb_project: str = 'C5-RL-Pong-DQN'
    run_name: str = 'pong_dqn_experiment'
    eval_only: bool = False
    model_path: str = 'ALE_Pong_v5_dqn_solution.dat'


def make_env(env_name: str, render_mode=None, env_frameskip=None, wrapper_skip=4):
    """Create the wrapped Atari environment for Pong training."""
    kwargs = {'render_mode': render_mode}
    if env_frameskip is not None:
        kwargs['frameskip'] = env_frameskip
    env = gym.make(env_name, **kwargs)
    if wrapper_skip and wrapper_skip > 1:
        env = MaxAndSkipEnv(env, skip=wrapper_skip)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    env = ScaledFloatFrame(env)
    return env


def print_env_info(name: str, env):
    """Print basic observation statistics for a wrapped environment."""
    obs, info = env.reset()
    print(f'*** {name} Environment ***')
    print(f'Observation shape: {obs.shape}, dtype: {obs.dtype}, range: [{obs.min()}, {obs.max()}]')


class FireResetEnv(gym.Wrapper):
    """Wrapper that performs the Atari FIRE action after reset."""

    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        # Forward the action to the underlying environment.
        return self.env.step(action)

    def reset(self, *, seed=None, options=None):
        # Preserve Gymnasium reset signature and return (obs, info).
        obs, info = self.env.reset(seed=seed, options=options)
        # Perform the game-start actions required by Pong.
        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset()
        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat actions and return the max of the last few frames."""

    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        self._obs_buffer.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    """Convert raw Atari RGB frames to 84x84 grayscale observations."""

    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            raise ValueError(f'Unknown resolution: {frame.size}')
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    """Stack the last N processed frames for temporal context."""

    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0), old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self, *, seed=None, options=None):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    """Reorder image axes from HWC to CHW for PyTorch compatibility."""

    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.uint8)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    """Scale pixel values to floating point range [0.0, 1.0]."""

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_DQN(input_shape, output_shape):
    """Build the convolutional DQN network used by the Atari agent."""
    return nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape),
    )


Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class ExperienceReplay:
    """Simple circular replay buffer for experience sampling."""

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent:
    """Agent wrapper that handles environment interaction and replay storage."""

    def __init__(self, env, exp_replay_buffer, seed=None):
        self.env = env
        self.exp_replay_buffer = exp_replay_buffer
        self.seed = seed
        self.reset_count = 0
        self._reset()

    def _reset(self):
        reset_seed = self.seed if self.reset_count == 0 else None
        self.current_state, _ = self.env.reset(seed=reset_seed)
        self.reset_count += 1
        self.total_reward = 0.0

    def step(self, net, epsilon=0.0, device='cpu'):
        """Take one action and store the resulting transition."""
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array([self.current_state]), dtype=torch.float32, device=device)
            q_values = net(state_tensor)
            _, action_tensor = torch.max(q_values, dim=1)
            action = int(action_tensor.item())

        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward

        experience = Experience(self.current_state, action, reward, done, new_state)
        self.exp_replay_buffer.append(experience)
        self.current_state = new_state

        if done:
            done_reward = self.total_reward
            self._reset()

        return done_reward


def train(
    env_name,
    env_frameskip,
    wrapper_skip,
    seed,
    gamma,
    batch_size,
    learning_rate,
    replay_size,
    replay_start_size,
    sync_target_frames,
    eps_start,
    eps_decay,
    eps_min,
    max_frames,
    mean_reward_bound,
    reward_average_size,
    convergence_reward_thresholds,
    save_path,
    training_metrics_path,
    use_wandb,
    project_name,
    run_name,
    device,
):
    """Train the DQN agent and save the resulting model parameters."""
    gym.register_envs(ale_py)

    wandb_run = None
    if use_wandb:
        require_wandb_available()
        wandb_run = wandb.init(project=project_name, name=run_name, config={
            'env_name': env_name,
            'env_frameskip': env_frameskip,
            'wrapper_skip': wrapper_skip,
            'seed': seed,
            'gamma': gamma,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'replay_size': replay_size,
            'replay_start_size': replay_start_size,
            'sync_target_frames': sync_target_frames,
            'eps_start': eps_start,
            'eps_decay': eps_decay,
            'eps_min': eps_min,
            'max_frames': max_frames,
            'mean_reward_bound': mean_reward_bound,
        })
        print('W&B run:', wandb_run.url)

    start_time = time.time()
    start_datetime = datetime.datetime.now()
    print('>>> Training starts at', start_datetime)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    env = make_env(env_name, env_frameskip=env_frameskip, wrapper_skip=wrapper_skip)
    if seed is not None:
        env.action_space.seed(seed)
    net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    target_net.load_state_dict(net.state_dict())

    buffer = ExperienceReplay(replay_size)
    agent = Agent(env, buffer, seed=seed)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    epsilon = eps_start
    frame_idx = 0
    total_rewards = []
    best_mean_reward = None
    threshold_frames = {str(threshold): None for threshold in convergence_reward_thresholds}

    while frame_idx < max_frames:
        frame_idx += 1
        epsilon = max(eps_min, epsilon * eps_decay)

        reward = agent.step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-reward_average_size:])
            print(f'Frame: {frame_idx} | Games: {len(total_rewards)} | Mean reward: {mean_reward:.3f} | epsilon: {epsilon:.3f}')
            if use_wandb and _HAS_WANDB:
                wandb.log({'epsilon': epsilon, 'reward_100': mean_reward, 'reward': reward}, step=frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                best_mean_reward = mean_reward

            for threshold in convergence_reward_thresholds:
                threshold_key = str(threshold)
                if threshold_frames[threshold_key] is None and mean_reward >= threshold:
                    threshold_frames[threshold_key] = frame_idx

            if mean_reward >= mean_reward_bound:
                print(f'Solved after {frame_idx} frames and {len(total_rewards)} games!')
                break

        if len(buffer) < replay_start_size:
            continue

        states, actions, rewards, dones, next_states = buffer.sample(batch_size)
        states_v = torch.tensor(states, dtype=torch.float32, device=device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32, device=device)
        actions_v = torch.tensor(actions, device=device)
        rewards_v = torch.tensor(rewards, device=device)
        done_mask = torch.BoolTensor(dones).to(device)

        state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

        next_state_values = target_net(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0
        next_state_values = next_state_values.detach()

        expected_state_action_values = rewards_v + gamma * next_state_values

        loss = nn.MSELoss()(state_action_values, expected_state_action_values)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if frame_idx % sync_target_frames == 0:
            target_net.load_state_dict(net.state_dict())

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), save_file)
    print('Training completed, model saved to', save_path)

    end_datetime = datetime.datetime.now()
    elapsed_seconds = time.time() - start_time
    metrics_path = Path(training_metrics_path) if training_metrics_path else Path('results') / f'{run_name}_train_metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    train_metrics = {
        'run_name': run_name,
        'env_name': env_name,
        'env_frameskip': env_frameskip,
        'wrapper_skip': wrapper_skip,
        'seed': seed,
        'save_path': save_path,
        'started_at': start_datetime.isoformat(),
        'ended_at': end_datetime.isoformat(),
        'elapsed_seconds': elapsed_seconds,
        'frames': frame_idx,
        'episodes': len(total_rewards),
        'best_mean_reward': None if best_mean_reward is None else float(best_mean_reward),
        'final_mean_reward': None if not total_rewards else float(np.mean(total_rewards[-reward_average_size:])),
        'reward_average_size': reward_average_size,
        'threshold_frames': threshold_frames,
        'total_rewards': [float(reward) for reward in total_rewards],
        'config': {
            'gamma': gamma,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'replay_size': replay_size,
            'replay_start_size': replay_start_size,
            'sync_target_frames': sync_target_frames,
            'eps_start': eps_start,
            'eps_decay': eps_decay,
            'eps_min': eps_min,
            'max_frames': max_frames,
            'mean_reward_bound': mean_reward_bound,
        },
    }
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(train_metrics, f, indent=2)
    print('Training metrics saved to', metrics_path)

    if use_wandb and _HAS_WANDB:
        wandb.log({
            'train/elapsed_seconds': elapsed_seconds,
            'train/frames': frame_idx,
            'train/episodes': len(total_rewards),
            'train/best_mean_reward': best_mean_reward,
            'train/final_mean_reward': train_metrics['final_mean_reward'],
        })
        wandb.save(str(metrics_path))
        wandb.save(str(save_file))
        wandb.finish()

    return net, env, total_rewards


def evaluate(env_name, model_path, device='cpu', env_frameskip=None, wrapper_skip=4):
    """Load a trained model and run a single evaluation episode."""
    env = make_env(env_name, env_frameskip=env_frameskip, wrapper_skip=wrapper_skip)
    net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    state, _ = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        state_v = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        q_vals = net(state_v)
        action = int(torch.argmax(q_vals, dim=1).item())
        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        state = next_state

    print('Evaluation total reward:', total_reward)
    return total_reward


def parse_args():
    """Parse CLI arguments, optionally merged with a JSON experiment config."""
    parser = argparse.ArgumentParser(description='Train DQN on ALE/Pong-v5')
    parser.add_argument('--config', help='Path to a JSON config file.')
    parser.add_argument('--experiment', help='Experiment name inside config["experiments"].')
    parser.add_argument('--env-name', dest='env_name')
    parser.add_argument('--env-frameskip', dest='env_frameskip', type=int)
    parser.add_argument('--wrapper-skip', dest='wrapper_skip', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.add_argument('--learning-rate', dest='learning_rate', type=float)
    parser.add_argument('--replay-size', dest='replay_size', type=int)
    parser.add_argument('--replay-start-size', dest='replay_start_size', type=int)
    parser.add_argument('--sync-target-frames', dest='sync_target_frames', type=int)
    parser.add_argument('--eps-start', dest='eps_start', type=float)
    parser.add_argument('--eps-decay', dest='eps_decay', type=float)
    parser.add_argument('--eps-min', dest='eps_min', type=float)
    parser.add_argument('--max-frames', dest='max_frames', type=int)
    parser.add_argument('--mean-reward-bound', dest='mean_reward_bound', type=float)
    parser.add_argument('--reward-average-size', dest='reward_average_size', type=int)
    parser.add_argument('--training-metrics-path', dest='training_metrics_path')
    parser.add_argument('--save-path', dest='save_path')
    parser.add_argument('--use-wandb', dest='use_wandb', action='store_true', default=None)
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false')
    parser.add_argument('--wandb-project', dest='wandb_project')
    parser.add_argument('--run-name', dest='run_name')
    parser.add_argument('--eval-only', dest='eval_only', action='store_true', default=None)
    parser.add_argument('--train-only', dest='eval_only', action='store_false')
    parser.add_argument('--model-path', dest='model_path')
    args = parser.parse_args()

    config_values = load_config(args.config, args.experiment) if args.config else {}
    cli_values = {
        key: value for key, value in vars(args).items()
        if key not in ('config', 'experiment') and value is not None
    }
    merged_values = {**config_values, **cli_values}
    return build_config(merged_values), args.config, args.experiment


def load_config(config_path, experiment_name=None):
    """Load default and named experiment settings from a JSON config file."""
    path = Path(config_path)
    with path.open('r', encoding='utf-8') as f:
        raw_config = json.load(f)

    if not isinstance(raw_config, dict):
        raise ValueError('Config file must contain a JSON object.')

    base_config = raw_config.get('defaults', {})
    if not isinstance(base_config, dict):
        raise ValueError('Config "defaults" must be an object.')

    experiments = raw_config.get('experiments', {})
    if experiments and not isinstance(experiments, dict):
        raise ValueError('Config "experiments" must be an object.')

    selected = {}
    if experiment_name:
        if experiment_name not in experiments:
            available = ', '.join(sorted(experiments)) or '<none>'
            raise ValueError(f'Experiment "{experiment_name}" not found. Available: {available}')
        selected = experiments[experiment_name]
        if not isinstance(selected, dict):
            raise ValueError(f'Experiment "{experiment_name}" must be an object.')

    if not experiment_name and experiments:
        available = ', '.join(sorted(experiments))
        print(f'No --experiment selected; using defaults only. Available experiments: {available}')

    return {**base_config, **selected}


def build_config(values):
    """Validate config keys and convert a dict into TrainConfig."""
    valid_fields = {field.name for field in fields(TrainConfig)}
    unknown_keys = sorted(set(values) - valid_fields)
    if unknown_keys:
        raise ValueError(f'Unknown config option(s): {", ".join(unknown_keys)}')
    return TrainConfig(**values)


def print_torch_diagnostics():
    """Print enough CUDA/PyTorch info to debug SLURM environment issues."""
    print('Python executable:', sys.executable)
    print('Torch version:', torch.__version__)
    print('Torch CUDA version:', torch.version.cuda)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'))
    print('torch.cuda.is_available():', torch.cuda.is_available())
    print('torch.cuda.device_count():', torch.cuda.device_count())
    if torch.cuda.is_available():
        print('CUDA device 0:', torch.cuda.get_device_name(0))


def select_device(require_cuda=False):
    print_torch_diagnostics()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA was required but torch.cuda.is_available() is False. '
            'This usually means the job is using a CPU-only PyTorch install, '
            'the wrong Python environment, or SLURM did not allocate a GPU.'
        )
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    config, config_path, experiment_name = parse_args()
    require_cuda = os.environ.get('REQUIRE_CUDA', '0') == '1'
    device = select_device(require_cuda=require_cuda)
    print('Using device:', device)
    if config_path:
        print(f'Loaded config: {config_path}')
    if experiment_name:
        print(f'Experiment: {experiment_name}')
    print('Run config:', json.dumps(asdict(config), indent=2))

    if config.eval_only:
        evaluate(
            config.env_name,
            config.model_path,
            device=device,
            env_frameskip=config.env_frameskip,
            wrapper_skip=config.wrapper_skip,
        )
    else:
        train(
            env_name=config.env_name,
            env_frameskip=config.env_frameskip,
            wrapper_skip=config.wrapper_skip,
            seed=config.seed,
            gamma=config.gamma,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            replay_size=config.replay_size,
            replay_start_size=config.replay_start_size,
            sync_target_frames=config.sync_target_frames,
            eps_start=config.eps_start,
            eps_decay=config.eps_decay,
            eps_min=config.eps_min,
            max_frames=config.max_frames,
            mean_reward_bound=config.mean_reward_bound,
            reward_average_size=config.reward_average_size,
            convergence_reward_thresholds=config.convergence_reward_thresholds,
            save_path=config.save_path,
            training_metrics_path=config.training_metrics_path,
            use_wandb=config.use_wandb,
            project_name=config.wandb_project,
            run_name=config.run_name,
            device=device,
        )
