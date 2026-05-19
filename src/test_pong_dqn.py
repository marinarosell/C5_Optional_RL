#!/usr/bin/env python3
"""Evaluate a trained Pong DQN and save GIFs/metrics for played episodes."""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import ale_py
import numpy as np
import torch
from PIL import Image

from train_pong_dqn import TrainConfig, build_config, load_config, make_DQN, make_env, select_device

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained DQN on ALE/Pong-v5 and save GIF videos.')
    parser.add_argument('--config', default='config/pong_dqn_experiments.json', help='Path to a JSON experiment config.')
    parser.add_argument('--experiment', default='baseline', help='Experiment name inside config["experiments"].')
    parser.add_argument('--model-path', help='Model checkpoint to load. Defaults to the experiment save_path.')
    parser.add_argument('--env-name', help='Gymnasium environment name. Defaults to the config env_name.')
    parser.add_argument('--env-frameskip', type=int, help='Override Gymnasium ALE frameskip.')
    parser.add_argument('--wrapper-skip', type=int, help='Override custom MaxAndSkip wrapper skip.')
    parser.add_argument('--episodes', type=int, default=3, help='Number of independent episodes to evaluate.')
    parser.add_argument('--output-dir', default='videos', help='Directory where GIF files are saved.')
    parser.add_argument('--prefix', default='pong_dqn', help='Prefix for generated GIF filenames.')
    parser.add_argument('--fps', type=int, default=20, help='Playback frames per second for the GIF.')
    parser.add_argument('--max-steps', type=int, default=10000, help='Maximum agent decisions per episode.')
    parser.add_argument('--seed', type=int, help='Optional base seed for evaluation episodes.')
    parser.add_argument(
        '--record-mode',
        choices=('all', 'best-worst', 'none'),
        default='all',
        help='Which episodes to save as GIFs. Use best-worst for 100-episode benchmark runs.',
    )
    parser.add_argument(
        '--metrics-path',
        help='Path to a JSON metrics file. Defaults to results/<experiment>_test_metrics.json.',
    )
    parser.add_argument('--use-wandb', action='store_true', help='Log benchmark metrics and GIFs to W&B.')
    return parser.parse_args()


def load_test_config(args):
    config_values = load_config(args.config, args.experiment) if args.config else {}
    config = build_config(config_values) if config_values else TrainConfig()

    env_name = args.env_name or config.env_name
    if args.env_frameskip is not None:
        config.env_frameskip = args.env_frameskip
    if args.wrapper_skip is not None:
        config.wrapper_skip = args.wrapper_skip
    model_path = args.model_path or config.model_path
    if args.model_path is None and config.save_path != TrainConfig().save_path:
        model_path = config.save_path

    return config, env_name, Path(model_path)


def select_action(net, state, device):
    with torch.no_grad():
        state_v = torch.tensor(np.array([state]), dtype=torch.float32, device=device)
        q_vals = net(state_v)
        return int(torch.argmax(q_vals, dim=1).item())


def save_episode_gif(frames, output_path, fps):
    if not frames:
        raise RuntimeError('No frames were recorded. Make sure the environment supports rgb_array rendering.')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / fps)
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def run_episode(env, net, device, max_steps, seed=None, record_frames=True):
    state, _ = env.reset(seed=seed)
    frames = []
    total_reward = 0.0
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        if record_frames:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        action = select_action(net, state, device)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

    return total_reward, steps, frames


def summarize_results(results):
    rewards = np.array([result['reward'] for result in results], dtype=np.float32)
    return {
        'episodes': len(results),
        'average_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
    }


def default_metrics_path(args):
    experiment_name = args.experiment or 'manual'
    return Path('results') / f'{experiment_name}_test_metrics.json'


def write_metrics(metrics_path, payload):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def log_benchmark_to_wandb(args, config, metrics_path, summary):
    if not args.use_wandb:
        return
    if not _HAS_WANDB:
        raise RuntimeError('Benchmark W&B logging requested, but wandb is not installed.')
    if os.environ.get('WANDB_MODE') == 'disabled':
        raise RuntimeError('WANDB_MODE=disabled, so benchmark results will not be saved to W&B.')

    run = wandb.init(
        project=config.wandb_project,
        name=f'{args.experiment}_benchmark',
        job_type='benchmark',
        config={
            'experiment': args.experiment,
            'episodes': args.episodes,
            'env_name': config.env_name,
            'env_frameskip': config.env_frameskip,
            'wrapper_skip': config.wrapper_skip,
            'model_path': str(config.save_path),
        },
    )
    wandb.log({
        'benchmark/average_reward': summary['average_reward'],
        'benchmark/std_reward': summary['std_reward'],
        'benchmark/min_reward': summary['min_reward'],
        'benchmark/max_reward': summary['max_reward'],
        'benchmark/episodes': summary['episodes'],
    })
    for key in ('best_gif_path', 'worst_gif_path'):
        if key in summary and Path(summary[key]).exists():
            wandb.log({f'benchmark/{key}': wandb.Video(summary[key], format='gif')})
    wandb.save(str(metrics_path))
    print('W&B benchmark run:', run.url)
    wandb.finish()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError('--episodes must be at least 1.')
    if args.fps < 1:
        raise ValueError('--fps must be at least 1.')

    gym.register_envs(ale_py)
    config, env_name, model_path = load_test_config(args)
    require_cuda = os.environ.get('REQUIRE_CUDA', '0') == '1'
    device = select_device(require_cuda=require_cuda)

    print('Using device:', device)
    print('Environment:', env_name)
    print('Model:', model_path)

    env = make_env(
        env_name,
        render_mode='rgb_array',
        env_frameskip=config.env_frameskip,
        wrapper_skip=config.wrapper_skip,
    )
    net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    output_dir = Path(args.output_dir)
    metrics_path = Path(args.metrics_path) if args.metrics_path else default_metrics_path(args)
    results = []
    best_episode = None
    worst_episode = None

    try:
        for episode_idx in range(args.episodes):
            seed = None if args.seed is None else args.seed + episode_idx
            should_record = args.record_mode in ('all', 'best-worst')
            reward, steps, frames = run_episode(
                env,
                net,
                device,
                args.max_steps,
                seed=seed,
                record_frames=should_record,
            )

            result = {
                'episode': episode_idx + 1,
                'reward': reward,
                'steps': steps,
                'frames': len(frames),
                'seed': seed,
            }

            if args.record_mode == 'all':
                gif_path = output_dir / f'{args.prefix}_episode_{episode_idx + 1:03d}_reward_{reward:.0f}.gif'
                save_episode_gif(frames, gif_path, args.fps)
                result['gif_path'] = str(gif_path)

            results.append(result)

            if args.record_mode == 'best-worst':
                candidate = {
                    'episode': episode_idx + 1,
                    'reward': reward,
                    'steps': steps,
                    'frames': frames,
                    'seed': seed,
                }
                if best_episode is None or reward > best_episode['reward']:
                    best_episode = candidate
                if worst_episode is None or reward < worst_episode['reward']:
                    worst_episode = candidate

            saved_text = f", saved={result['gif_path']}" if 'gif_path' in result else ''
            print(f'Episode {episode_idx + 1}: reward={reward:.2f}, steps={steps}{saved_text}')
    finally:
        env.close()

    video_paths = {}
    if args.record_mode == 'best-worst':
        best_path = output_dir / f'{args.prefix}_best_episode_{best_episode["episode"]:03d}_reward_{best_episode["reward"]:.0f}.gif'
        worst_path = output_dir / f'{args.prefix}_worst_episode_{worst_episode["episode"]:03d}_reward_{worst_episode["reward"]:.0f}.gif'
        save_episode_gif(best_episode['frames'], best_path, args.fps)
        save_episode_gif(worst_episode['frames'], worst_path, args.fps)
        video_paths = {
            'best_gif_path': str(best_path),
            'worst_gif_path': str(worst_path),
        }
        results[best_episode['episode'] - 1]['gif_path'] = str(best_path)
        results[worst_episode['episode'] - 1]['gif_path'] = str(worst_path)
        print(f'Best episode saved: {best_path}')
        print(f'Worst episode saved: {worst_path}')

    summary = summarize_results(results)
    payload = {
        'experiment': args.experiment,
        'env_name': env_name,
        'env_frameskip': config.env_frameskip,
        'wrapper_skip': config.wrapper_skip,
        'model_path': str(model_path),
        'record_mode': args.record_mode,
        'fps': args.fps,
        'max_steps': args.max_steps,
        'summary': {**summary, **video_paths},
        'episodes': results,
    }
    write_metrics(metrics_path, payload)
    log_benchmark_to_wandb(args, config, metrics_path, payload['summary'])

    print('Summary:', json.dumps(payload['summary'], indent=2))
    print('Metrics saved to:', metrics_path)


if __name__ == '__main__':
    main()
