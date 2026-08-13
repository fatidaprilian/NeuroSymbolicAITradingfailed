"""
Proximal Policy Optimization (PPO) and Advantage Actor-Critic (A2C) Baseline Training Module.
Executes discrete 5-action baseline model training for comparative DRL benchmarks.
"""

import os
import torch
import pandas as pd
from stable_baselines3 import PPO, A2C
from src.features import load_and_preprocess_data, train_val_test_split
from trading_env import CryptoTradingEnv5Action

MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_ppo_agent(symbol: str = 'btc', scenario: str = 'adaptive', total_timesteps: int = 50000):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()

    print(f"\n============================================================")
    print(f"TRAINING PPO AGENT: [{symbol_lower.upper()}] Scenario: [{scenario_lower.upper()}]")
    print(f"============================================================")

    df = load_and_preprocess_data(symbol_lower)
    df_train, _, _ = train_val_test_split(df)

    enable_net = (scenario_lower != 'baseline')
    env = CryptoTradingEnv5Action(df_train, symbol=symbol_lower, enable_safety_net=enable_net)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=0,
        device="cpu",
        seed=42
    )

    print(f"Training PPO for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = f"{MODEL_DIR}/ppo_agent_{symbol_lower}_{scenario_lower}.zip"
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def train_a2c_agent(symbol: str = 'btc', scenario: str = 'adaptive', total_timesteps: int = 50000):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()

    print(f"\n============================================================")
    print(f"TRAINING A2C AGENT: [{symbol_lower.upper()}] Scenario: [{scenario_lower.upper()}]")
    print(f"============================================================")

    df = load_and_preprocess_data(symbol_lower)
    df_train, _, _ = train_val_test_split(df)

    enable_net = (scenario_lower != 'baseline')
    env = CryptoTradingEnv5Action(df_train, symbol=symbol_lower, enable_safety_net=enable_net)

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=0.0007,
        n_steps=5,
        gamma=0.99,
        gae_lambda=1.0,
        verbose=0,
        device="cpu",
        seed=42
    )

    print(f"Training A2C for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = f"{MODEL_DIR}/a2c_agent_{symbol_lower}_{scenario_lower}.zip"
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def train_all_baseline_agents(timesteps: int = 50000):
    symbols = ['btc', 'eth', 'xrp']
    scenarios = ['adaptive', 'baseline']

    for sym in symbols:
        for sc in scenarios:
            train_ppo_agent(sym, sc, timesteps)
            train_a2c_agent(sym, sc, timesteps)


if __name__ == "__main__":
    train_all_baseline_agents(timesteps=50000)
