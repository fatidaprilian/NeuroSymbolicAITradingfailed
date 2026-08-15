"""
Train Deep Q-Network (DQN / Double-DQN) Agents for Neuro-Symbolic Crypto Trading.
Uses CryptoTradingEnv5Action (Discrete 5-Action Space: Huang & Su, 2024; Vergara & Kristjanpoller, 2024) for partial position sizing.
Matches paper title: "A NEURO-SYMBOLIC AI TRADING ARCHITECTURE COMBINING HYBRID LR-LSTM PREDICTION, DEEP Q-NETWORK, AND SYMBOLIC SAFETY NETS"
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.features import load_and_preprocess_data, train_val_test_split, FEATURE_COLUMNS
from trading_env import CryptoTradingEnv5Action

MODEL_DIR = "ml_models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def train_dqn(symbol: str = 'btc', scenario: str = 'adaptive', timesteps: int = 50000):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()

    print(f"\n{'='*60}")
    print(f"TRAINING DEEP Q-NETWORK (DQN): [{symbol_lower.upper()}] Scenario: [{scenario_lower.upper()}]")
    print(f"{'='*60}")

    df = load_and_preprocess_data(symbol_lower)

    # Attach Hybrid Predictions
    try:
        model_lr = joblib.load(f"{MODEL_DIR}/model_lr_baseline_{symbol_lower}.pkl")
        scaler_lr = joblib.load(f"{MODEL_DIR}/scaler_lr_{symbol_lower}.pkl")
        X_lr = scaler_lr.transform(df[FEATURE_COLUMNS])
        df['prediction'] = model_lr.predict(X_lr)
    except Exception:
        df['prediction'] = df['close']

    df_train, df_val, df_test = train_val_test_split(df)
    test_data_path = f"{MODEL_DIR}/test_data_{symbol_lower}_{scenario_lower}.csv"
    df_test.to_csv(test_data_path)

    enable_net = (scenario_lower != 'baseline')
    env_symbol = symbol_lower if scenario_lower != 'default' else 'default'

    env = DummyVecEnv([lambda: CryptoTradingEnv5Action(df_train, symbol=env_symbol, enable_safety_net=enable_net)])

    model_path = f"{MODEL_DIR}/dqn_agent_{symbol_lower}_{scenario_lower}.zip"

    model_dqn = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.6,
        exploration_final_eps=0.10,
        policy_kwargs=dict(net_arch=[128, 128]),
        device='cpu',  # Consistent device
        verbose=0,
    )

    print(f"Training DQN (5-Action Space) for {timesteps:,} steps...")
    model_dqn.learn(total_timesteps=timesteps)

    model_dqn.save(model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def train_rl(symbol: str = 'btc', scenario: str = 'adaptive', algo: str = 'dqn', timesteps: int = 50000):
    return train_dqn(symbol, scenario, timesteps=timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 5-Action Deep Q-Network (DQN) for Crypto Trading")
    parser.add_argument('--symbol', type=str, default='btc')
    parser.add_argument('--scenario', type=str, default='adaptive', choices=['adaptive', 'baseline', 'default'])
    parser.add_argument('--timesteps', type=int, default=50000)
    args = parser.parse_args()

    train_dqn(args.symbol, args.scenario, args.timesteps)
