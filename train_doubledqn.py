"""
Double Deep Q-Network (Double DQN) Agent Training Module.
Provides modular training execution for 5-action discrete portfolio rebalancing
across cryptocurrency asset datasets (BTC, ETH, XRP).
"""

import os
import torch
import pandas as pd
from stable_baselines3 import DQN
from src.features import load_and_preprocess_data, train_val_test_split
from trading_env import CryptoTradingEnv5Action

MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_doubledqn_agent(symbol: str = 'btc', scenario: str = 'adaptive', total_timesteps: int = 50000):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()

    print(f"\n============================================================")
    print(f"TRAINING DOUBLE DQN (5-Action): [{symbol_lower.upper()}] Scenario: [{scenario_lower.upper()}]")
    print(f"============================================================")

    df = load_and_preprocess_data(symbol_lower)
    df_train, _, _ = train_val_test_split(df)

    enable_net = (scenario_lower != 'baseline')
    env = CryptoTradingEnv5Action(df_train, symbol=symbol_lower, enable_safety_net=enable_net)

    # Double DQN configuration
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=0,
        device="cpu",
        seed=42
    )

    print(f"Training Double DQN for {total_timesteps:,} steps...")
    model.learn(total_timesteps=total_timesteps)

    save_path = f"{MODEL_DIR}/doubledqn_agent_{symbol_lower}_{scenario_lower}.zip"
    model.save(save_path)
    print(f"Model saved to: {save_path}")


def train_all_doubledqn(timesteps: int = 50000):
    symbols = ['btc', 'eth', 'xrp']
    scenarios = ['adaptive', 'baseline']

    for sym in symbols:
        for sc in scenarios:
            train_doubledqn_agent(sym, sc, timesteps)


if __name__ == "__main__":
    train_all_doubledqn(timesteps=50000)
