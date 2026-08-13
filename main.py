"""
Main CLI Launcher for Neuro-Symbolic Crypto Trading Framework.
Provides commands for training forecasting models, 5-Action Deep Q-Network (DQN) agents, backtesting, and paper material generation.
Matches paper title: "A NEURO-SYMBOLIC AI TRADING ARCHITECTURE COMBINING HYBRID LR-LSTM PREDICTION, DEEP Q-NETWORK, AND SYMBOLIC SAFETY NETS"
"""

import sys
import argparse
from train_hybrid import train_and_eval_hybrid
from train_dqn import train_dqn
from run_test import run_full_benchmark_suite
from generate_paper_materials import generate_all_paper_materials

SYMBOLS = ['btc', 'eth', 'xrp']
TIMESTEPS = 50000


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Deep Q-Network (DQN) Crypto Trading CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: train-hybrid
    hybrid_parser = subparsers.add_parser("train-hybrid", help="Train Hybrid LR-LSTM Forecasting Model")
    hybrid_parser.add_argument("--symbol", type=str, default="btc", choices=["btc", "eth", "xrp", "all"])

    # Command: train-rl
    rl_parser = subparsers.add_parser("train-rl", help="Train 5-Action Deep Q-Network (DQN)")
    rl_parser.add_argument("--symbol", type=str, default="btc", choices=["btc", "eth", "xrp", "all"])
    rl_parser.add_argument("--scenario", type=str, default="adaptive", choices=["adaptive", "baseline", "default"])
    rl_parser.add_argument("--timesteps", type=int, default=TIMESTEPS)

    # Command: backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run Backtest & Statistical Significance Suite")
    backtest_parser.add_argument("--symbol", type=str, default="all", choices=["btc", "eth", "xrp", "all"])

    # Command: generate-paper
    subparsers.add_parser("generate-paper", help="Generate LaTeX & Markdown Paper Tables & Material")

    # Command: run-all
    subparsers.add_parser("run-all", help="Run full pipeline: forecasting, 5-action DQN training, backtest, paper tables")

    args = parser.parse_args()

    if args.command == "train-hybrid":
        symbols = SYMBOLS if args.symbol == "all" else [args.symbol]
        for s in symbols:
            train_and_eval_hybrid(s)

    elif args.command == "train-rl":
        symbols = SYMBOLS if args.symbol == "all" else [args.symbol]
        for s in symbols:
            train_dqn(s, args.scenario, args.timesteps)

    elif args.command == "backtest":
        symbols = tuple(SYMBOLS) if args.symbol == "all" else (args.symbol,)
        run_full_benchmark_suite(symbols)

    elif args.command == "generate-paper":
        generate_all_paper_materials()

    elif args.command == "run-all" or args.command is None:
        print("Executing Full End-to-End Pipeline (5-Action Deep Q-Network)...")

        # Phase 1: Hybrid LR-LSTM forecasting
        for s in SYMBOLS:
            train_and_eval_hybrid(s)

        # Phase 2: Train 5-Action Deep Q-Network (DQN)
        for s in SYMBOLS:
            train_dqn(s, "adaptive", TIMESTEPS)
            train_dqn(s, "baseline", TIMESTEPS)

        # Phase 3: Benchmark & statistical tests
        run_full_benchmark_suite(tuple(SYMBOLS))

        # Phase 4: Paper materials
        generate_all_paper_materials()

        print("\nFull Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
