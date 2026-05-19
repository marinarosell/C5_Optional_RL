#!/usr/bin/env python3
"""Run all experiments listed in one sequential experimentation round."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Run a configured round of Pong DQN experiments.')
    parser.add_argument('--config', default='config/pong_dqn_experiments.json')
    parser.add_argument('--round', required=True, dest='round_name', help='Round name from config["rounds"].')
    parser.add_argument('--skip-training', action='store_true', help='Only run benchmarks for this round.')
    parser.add_argument('--skip-benchmark', action='store_true', help='Only train models for this round.')
    parser.add_argument('--benchmark-episodes', type=int, default=100)
    parser.add_argument('--max-frames', type=int, help='Optional training max_frames override for quick trial runs.')
    parser.add_argument('--use-wandb', action='store_true', help='Enable Weights & Biases logging for training runs.')
    return parser.parse_args()


def load_round(config_path, round_name):
    with Path(config_path).open('r', encoding='utf-8') as f:
        config = json.load(f)
    rounds = config.get('rounds', {})
    if round_name not in rounds:
        available = ', '.join(sorted(rounds)) or '<none>'
        raise ValueError(f'Round "{round_name}" not found. Available rounds: {available}')
    return rounds[round_name]


def run_command(command):
    print('\n>>>', ' '.join(command))
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    experiments = load_round(args.config, args.round_name)

    for experiment in experiments:
        if not args.skip_training:
            train_command = [
                sys.executable,
                'src/train_pong_dqn.py',
                '--config',
                args.config,
                '--experiment',
                experiment,
            ]
            if args.max_frames is not None:
                train_command.extend(['--max-frames', str(args.max_frames)])
            if args.use_wandb:
                train_command.append('--use-wandb')
            run_command(train_command)

        if not args.skip_benchmark:
            benchmark_command = [
                sys.executable,
                'src/benchmark_pong_dqn.py',
                '--config',
                args.config,
                '--experiment',
                experiment,
                '--episodes',
                str(args.benchmark_episodes),
                '--prefix',
                experiment,
            ]
            run_command(benchmark_command)


if __name__ == '__main__':
    main()
