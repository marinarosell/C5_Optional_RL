#!/usr/bin/env python3
"""Run the standard 100-episode Pong DQN benchmark."""

import sys

from test_pong_dqn import main


DEFAULT_ARGS = [
    '--episodes',
    '100',
    '--record-mode',
    'best-worst',
    '--output-dir',
    'videos/benchmarks',
    '--prefix',
    'pong_dqn_benchmark',
]


if __name__ == '__main__':
    sys.argv = [sys.argv[0], *DEFAULT_ARGS, *sys.argv[1:]]
    main()
