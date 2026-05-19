#!/usr/bin/env python3
"""Run or print commands for one sequential experimentation round."""

import argparse
import json
import shlex
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
    parser.add_argument(
        '--mode',
        choices=('local', 'print', 'sbatch'),
        default='local',
        help='local runs commands now, print emits shell commands, sbatch submits one job per experiment.',
    )
    parser.add_argument(
        '--script',
        help='SLURM wrapper script used with --mode sbatch. The experiment command is passed as script arguments.',
    )
    parser.add_argument('--job-name-prefix', default='pong', help='Prefix for SLURM job names in --mode sbatch.')
    parser.add_argument('--logs-dir', default='logs', help='Directory for SLURM stdout/stderr files.')
    parser.add_argument('--python-bin', default=sys.executable, help='Python executable used inside generated experiment commands.')
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


def quote_command(command):
    return ' '.join(shlex.quote(part) for part in command)


def build_experiment_command(args, experiment):
    commands = []
    if not args.skip_training:
        command = [
            args.python_bin,
            'src/train_pong_dqn.py',
            '--config',
            args.config,
            '--experiment',
            experiment,
        ]
        if args.max_frames is not None:
            command.extend(['--max-frames', str(args.max_frames)])
        if args.use_wandb:
            command.append('--use-wandb')
        commands.append(command)

    if not args.skip_benchmark:
        commands.append([
            args.python_bin,
            'src/benchmark_pong_dqn.py',
            '--config',
            args.config,
            '--experiment',
            experiment,
            '--episodes',
            str(args.benchmark_episodes),
            '--prefix',
            experiment,
        ])

    return commands


def run_experiment_locally(commands):
    for command in commands:
        run_command(command)


def submit_sbatch(args, experiment, commands):
    if not args.script:
        raise ValueError('--script is required with --mode sbatch.')

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    sbatch_command = [
        'sbatch',
        '--job-name',
        f'{args.job_name_prefix}_{experiment}',
        '--output',
        str(logs_dir / f'{experiment}_%j.out'),
        '--error',
        str(logs_dir / f'{experiment}_%j.err'),
        args.script,
        quote_command_sequence(commands),
    ]
    run_command(sbatch_command)


def quote_command_sequence(commands):
    return ' && '.join(quote_command(command) for command in commands)


def main():
    args = parse_args()
    experiments = load_round(args.config, args.round_name)

    for experiment in experiments:
        commands = build_experiment_command(args, experiment)
        if args.mode == 'local':
            run_experiment_locally(commands)
        elif args.mode == 'print':
            print(quote_command_sequence(commands))
        elif args.mode == 'sbatch':
            submit_sbatch(args, experiment, commands)


if __name__ == '__main__':
    main()
