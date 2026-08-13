"""
Main CLI Launcher for Neuro-Symbolic Crypto Trading Framework.
Provides commands for training forecasting models, DRL agents, backtesting, and paper material generation.
"""

import sys
import argparse
from train_hybrid import train_and_eval_hybrid
from train_dqn import train_dqn
from run_test import run_full_benchmark_suite
from generate_paper_materials import generate_all_paper_materials


def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic AI Crypto Trading Architecture CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: train-hybrid
    hybrid_parser = subparsers.add_parser("train-hybrid", help="Train Hybrid LR-LSTM Forecasting Model")
    hybrid_parser.add_argument("--symbol", type=str, default="btc", choices=["btc", "eth", "xrp", "all"])

    # Command: train-rl
    rl_parser = subparsers.add_parser("train-rl", help="Train DRL Agent (DQN / QR-DQN)")
    rl_parser.add_argument("--symbol", type=str, default="btc", choices=["btc", "eth", "xrp", "all"])
    rl_parser.add_argument("--scenario", type=str, default="adaptive", choices=["adaptive", "baseline", "default"])
    rl_parser.add_argument("--timesteps", type=int, default=50000)

    # Command: backtest
    backtest_parser = subparsers.add_parser("backtest", help="Run Backtest & Statistical Significance Suite")
    backtest_parser.add_argument("--symbol", type=str, default="all", choices=["btc", "eth", "xrp", "all"])

    # Command: generate-paper
    subparsers.add_parser("generate-paper", help="Generate LaTeX & Markdown Paper Tables & Material")

    # Command: run-all
    subparsers.add_parser("run-all", help="Run full pipeline: forecasting, DRL training, backtest, paper tables")

    args = parser.parse_args()

    if args.command == "train-hybrid":
        symbols = ["btc", "eth", "xrp"] if args.symbol == "all" else [args.symbol]
        for s in symbols:
            train_and_eval_hybrid(s)

    elif args.command == "train-rl":
        symbols = ["btc", "eth", "xrp"] if args.symbol == "all" else [args.symbol]
        for s in symbols:
            train_dqn(s, args.scenario, args.timesteps)

    elif args.command == "backtest":
        symbols = ("btc", "eth", "xrp") if args.symbol == "all" else (args.symbol,)
        run_full_benchmark_suite(symbols)

    elif args.command == "generate-paper":
        generate_all_paper_materials()

    elif args.command == "run-all" or args.command is None:
        print("Executing Full End-to-End Pipeline...")
        for s in ["btc", "eth", "xrp"]:
            train_and_eval_hybrid(s)
            train_dqn(s, "adaptive", 30000)
            train_dqn(s, "baseline", 30000)
        run_full_benchmark_suite(("btc", "eth", "xrp"))
        generate_all_paper_materials()
        print("\nFull Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
