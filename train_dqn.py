"""
Train DQN / QR-DQN Agent for Neuro-Symbolic Crypto Trading.
Uses standardized feature engineering pipeline and logs reward components.
"""

import os
import argparse
import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from src.features import load_and_preprocess_data, train_val_test_split, FEATURE_COLUMNS
from trading_env import CryptoTradingEnv

MODEL_DIR = "ml_models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def train_dqn(symbol: str = 'btc', scenario: str = 'adaptive', timesteps: int = 100000):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()

    print(f"\n{'='*60}")
    print(f"TRAINING DQN AGENT: [{symbol_lower.upper()}] Scenario: [{scenario_lower.upper()}]")
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

    env = DummyVecEnv([lambda: CryptoTradingEnv(df_train, symbol=env_symbol, enable_safety_net=enable_net)])

    model_path = f"{MODEL_DIR}/dqn_agent_{symbol_lower}_{scenario_lower}.zip"

    model_dqn = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-5,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[256, 256, 128]),
        verbose=0,
    )

    print(f"Training DQN for {timesteps:,} steps...")
    model_dqn.learn(total_timesteps=timesteps)

    model_dqn.save(model_path)
    print(f"Model saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN Agent for Crypto Trading")
    parser.add_argument('--symbol', type=str, default='btc')
    parser.add_argument('--scenario', type=str, default='adaptive', choices=['adaptive', 'baseline', 'default'])
    parser.add_argument('--timesteps', type=int, default=50000)
    args = parser.parse_args()

    train_dqn(args.symbol, args.scenario, args.timesteps)
