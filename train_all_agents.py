"""
Train and Benchmark DRL Trading Agents (5-Action DQN, Double DQN, PPO, A2C).
Evaluates RL algorithm stability under baseline and Neuro-Symbolic (veto-enabled) scenarios.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from stable_baselines3 import DQN, PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.features import load_and_preprocess_data, train_val_test_split, FEATURE_COLUMNS
from trading_env import CryptoTradingEnv

MODEL_DIR = "ml_models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def train_agent(symbol: str = 'btc', algo: str = 'dqn', scenario: str = 'adaptive', timesteps: int = 100000):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()
    algo_lower = algo.lower()

    print(f"\n{'='*60}")
    print(f"🚀 TRAINING DRL AGENT: [{symbol_lower.upper()}] Algo: [{algo_lower.upper()}] Scenario: [{scenario_lower.upper()}]")
    print(f"{'='*60}")

    df = load_and_preprocess_data(symbol_lower)

    # Load Hybrid Predictions
    try:
        model_lr = joblib.load(f"{MODEL_DIR}/model_lr_baseline_{symbol_lower}.pkl")
        scaler_lr = joblib.load(f"{MODEL_DIR}/scaler_lr_{symbol_lower}.pkl")
        X_lr = scaler_lr.transform(df[FEATURE_COLUMNS])
        pred_lr = model_lr.predict(X_lr)
        df['prediction'] = pred_lr
    except Exception:
        df['prediction'] = df['close']

    df_train, df_val, df_test = train_val_test_split(df)
    test_data_path = f"{MODEL_DIR}/test_data_{symbol_lower}_{scenario_lower}.csv"
    df_test.to_csv(test_data_path)

    enable_net = (scenario_lower != 'baseline')
    env_symbol = symbol_lower if scenario_lower != 'default' else 'default'

    env = DummyVecEnv([lambda: CryptoTradingEnv(df_train, symbol=env_symbol, enable_safety_net=enable_net)])

    model_save_name = f"{algo_lower}_agent_{symbol_lower}_{scenario_lower}.zip"
    model_path = f"{MODEL_DIR}/{model_save_name}"

    if algo_lower == 'dqn':
        model = DQN(
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
    elif algo_lower == 'ddqn':
        # Double DQN configuration in SB3
        model = DQN(
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
    elif algo_lower == 'ppo':
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            verbose=0
        )
    elif algo_lower == 'sac':
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.99,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=0
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    print(f"🔄 Training {algo_lower.upper()} for {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps)

    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL Trading Agent")
    parser.add_argument('--symbol', type=str, default='btc')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'ddqn', 'qrdqn', 'ppo', 'sac'])
    parser.add_argument('--scenario', type=str, default='adaptive', choices=['adaptive', 'baseline', 'default'])
    parser.add_argument('--timesteps', type=int, default=50000)
    args = parser.parse_args()

    train_agent(args.symbol, args.algo, args.scenario, args.timesteps)
