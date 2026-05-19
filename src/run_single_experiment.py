#!/usr/bin/env python3
"""Run one configured experiment by importing the training and benchmark code."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import ale_py
import torch

from test_pong_dqn import (
    run_episode,
    save_episode_gif,
    summarize_results,
    write_metrics,
)
from train_pong_dqn import build_config, load_config, make_DQN, make_env, train


def parse_args():
    parser = argparse.ArgumentParser(description='Train and/or benchmark one configured Pong DQN experiment.')
    parser.add_argument('--config', default='config/pong_dqn_experiments.json')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--skip-training', action='store_true')
    parser.add_argument('--skip-benchmark', action='store_true')
    parser.add_argument('--benchmark-episodes', type=int, default=100)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--output-dir', default='videos/benchmarks')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--max-steps', type=int, default=10000)
    parser.add_argument('--seed', type=int)
    return parser.parse_args()


def load_experiment_config(config_path, experiment):
    values = load_config(config_path, experiment)
    return build_config(values)


def train_experiment(config, device):
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


def benchmark_experiment(config, experiment, device, episodes, output_dir, fps, max_steps, seed):
    env = make_env(
        config.env_name,
        render_mode='rgb_array',
        env_frameskip=config.env_frameskip,
        wrapper_skip=config.wrapper_skip,
    )
    net = make_DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load(config.save_path, map_location=device))
    net.eval()

    results = []
    best_episode = None
    worst_episode = None
    try:
        for episode_idx in range(episodes):
            episode_seed = None if seed is None else seed + episode_idx
            reward, steps, frames = run_episode(
                env,
                net,
                device,
                max_steps,
                seed=episode_seed,
                record_frames=True,
            )
            result = {
                'episode': episode_idx + 1,
                'reward': reward,
                'steps': steps,
                'frames': len(frames),
                'seed': episode_seed,
            }
            results.append(result)

            candidate = {
                'episode': episode_idx + 1,
                'reward': reward,
                'steps': steps,
                'frames': frames,
                'seed': episode_seed,
            }
            if best_episode is None or reward > best_episode['reward']:
                best_episode = candidate
            if worst_episode is None or reward < worst_episode['reward']:
                worst_episode = candidate

            print(f'Benchmark episode {episode_idx + 1}: reward={reward:.2f}, steps={steps}', flush=True)
    finally:
        env.close()

    output_dir = Path(output_dir)
    best_path = output_dir / f'{experiment}_best_episode_{best_episode["episode"]:03d}_reward_{best_episode["reward"]:.0f}.gif'
    worst_path = output_dir / f'{experiment}_worst_episode_{worst_episode["episode"]:03d}_reward_{worst_episode["reward"]:.0f}.gif'
    save_episode_gif(best_episode['frames'], best_path, fps)
    save_episode_gif(worst_episode['frames'], worst_path, fps)

    results[best_episode['episode'] - 1]['gif_path'] = str(best_path)
    results[worst_episode['episode'] - 1]['gif_path'] = str(worst_path)

    summary = summarize_results(results)
    summary['best_gif_path'] = str(best_path)
    summary['worst_gif_path'] = str(worst_path)

    payload = {
        'experiment': experiment,
        'env_name': config.env_name,
        'env_frameskip': config.env_frameskip,
        'wrapper_skip': config.wrapper_skip,
        'model_path': config.save_path,
        'record_mode': 'best-worst',
        'fps': fps,
        'max_steps': max_steps,
        'summary': summary,
        'episodes': results,
    }
    metrics_path = Path('results') / f'{experiment}_test_metrics.json'
    write_metrics(metrics_path, payload)
    print('Benchmark summary:', json.dumps(summary, indent=2), flush=True)
    print('Benchmark metrics saved to:', metrics_path, flush=True)


def main():
    args = parse_args()
    run_configured_experiment(
        config_path=args.config,
        experiment=args.experiment,
        skip_training=args.skip_training,
        skip_benchmark=args.skip_benchmark,
        benchmark_episodes=args.benchmark_episodes,
        max_frames=args.max_frames,
        use_wandb=args.use_wandb,
        output_dir=args.output_dir,
        fps=args.fps,
        max_steps=args.max_steps,
        seed=args.seed,
    )


def run_configured_experiment(
    config_path,
    experiment,
    skip_training=False,
    skip_benchmark=False,
    benchmark_episodes=100,
    max_frames=None,
    use_wandb=False,
    output_dir='videos/benchmarks',
    fps=20,
    max_steps=10000,
    seed=None,
):
    """Train and/or benchmark one experiment from a config file."""
    config = load_experiment_config(config_path, experiment)
    if max_frames is not None:
        config.max_frames = max_frames
    if use_wandb:
        config.use_wandb = True

    gym.register_envs(ale_py)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)
    print('Experiment:', experiment, flush=True)
    print('Config:', json.dumps(config.__dict__, indent=2), flush=True)

    if not skip_training:
        train_experiment(config, device)
    if not skip_benchmark:
        benchmark_experiment(
            config,
            experiment,
            device,
            benchmark_episodes,
            output_dir,
            fps,
            max_steps,
            seed if seed is not None else config.seed,
        )


if __name__ == '__main__':
    main()
