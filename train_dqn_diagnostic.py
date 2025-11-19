"""
TRAIN DQN DIAGNOSTIC V15.3

Changes from V15.2:
- Uses trading_env_diagnostic V15.3 (critical rebalance)
- Softer idle penalty (0.002 base, 0.05 max, slower growth)
- Stronger block penalties (safety/rapid/hard limit)
- NEW patience reward for smart HOLD in risky conditions
- Tuned MAX_TRADES (BTC:150, ETH:120, XRP:80) for diagnostic
"""

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from trading_env_diagnostic import CryptoTradingEnvDiagnostic
import os
import argparse
from tqdm import tqdm


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="Training Diagnostic V15.3", unit="steps")

    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % 1000 == 0:
            self.pbar.set_postfix({
                'episode': self.n_calls,
                'steps': self.num_timesteps
            })
        return True

    def _on_training_end(self):
        self.pbar.close()


# === ARGPARSE ===
parser = argparse.ArgumentParser(description='Train V15.3 Diagnostic Agent')
parser.add_argument('--symbol', type=str, default='btc',
                    help='Asset symbol (btc/eth/xrp)')
parser.add_argument('--scenario', type=str, default='adaptive',
                    choices=['adaptive', 'default', 'baseline'],
                    help='Training scenario')
parser.add_argument('--steps', type=int, default=50000,
                    help='Diagnostic steps (default: 50k)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()
TOTAL_TIMESTEPS = args.steps

data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"

print(f"\n{'='*70}")
print(f"🔬 DIAGNOSTIC V15.3: [{symbol_lower.upper()}] [{scenario.upper()}]")
print(f"{'='*70}")
print(f"   KEY CHANGES FROM V15.2:")
print(f"   ✅ Much softer idle penalty (0.002 base, 0.05 max, slower growth)")
print(f"   ✅ Stronger safety / rapid / hard-limit penalties")
print(f"   ✅ NEW patience reward for smart HOLD in risky conditions")
print(f"   ✅ Tuned MAX_TRADES (BTC:150, ETH:120, XRP:80)")
print(f"{'='*70}\n")

# === LOAD DATA ===
print(f"📂 Loading data {symbol_lower} from {data_filename}...")
df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
df = df.ffill()

# === FEATURE ENGINEERING ===
print("🛠️ Feature Engineering V2...")
df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].fillna(50)

df['BB_middle'] = df['close'].rolling(window=20).mean()
df['BB_std'] = df['close'].rolling(window=20).std()
df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])

df['tr1'] = df['high'] - df['low']
df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR'] = df['tr'].rolling(window=14).mean()

df.dropna(inplace=True)
print(f"📊 Data after feature engineering: {len(df)} rows")

# === HYBRID PREDICTION ===
print(f"🧠 Loading LR & LSTM models for {symbol_lower}...")
model_lr = joblib.load(os.path.join(
    model_dir, f'model_lr_baseline_{symbol_lower}.pkl'))
scaler_lr = joblib.load(os.path.join(
    model_dir, f'scaler_lr_{symbol_lower}.pkl'))
model_lstm = load_model(os.path.join(
    model_dir, f'best_lstm_model_{symbol_lower}.keras'))
scaler_lstm_features = joblib.load(os.path.join(
    model_dir, f'scaler_lstm_features_{symbol_lower}.pkl'))
scaler_lstm_target = joblib.load(os.path.join(
    model_dir, f'scaler_lstm_target_{symbol_lower}.pkl'))

print("🔮 Generating Hybrid predictions...")
features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR']

X_all = df[features]
X_scaled_lr = scaler_lr.transform(X_all)
pred_lr_all = model_lr.predict(X_scaled_lr)

SEQ_LENGTH = 60
pred_lstm_all = np.full(len(df), np.nan)
X_scaled_lstm = scaler_lstm_features.transform(X_all.values)
X_seq = np.array([X_scaled_lstm[i:i+SEQ_LENGTH]
                 for i in range(len(X_scaled_lstm) - SEQ_LENGTH)])
pred_lstm_seq = model_lstm.predict(X_seq, verbose=0, batch_size=2048)
pred_lstm_all[SEQ_LENGTH:] = scaler_lstm_target.inverse_transform(
    pred_lstm_seq).flatten()

df['prediction'] = (0.8 * pred_lr_all) + (0.2 * pred_lstm_all)
df.dropna(inplace=True)
print(f"✅ Data ready for diagnostic ({symbol_lower}): {len(df)} hours")

# === TRAIN/TEST SPLIT ===
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
print(f"Diagnostic with {len(df_train)} hours of training data.")

# === ENVIRONMENT SETUP V15.3 ===
env_symbol_arg = 'default' if scenario == 'default' else symbol_lower
enable_net_arg = False if scenario == 'baseline' else True

print(f"\n🏗️ Environment V15.3 Setup:")
print(f"   Symbol: '{env_symbol_arg}'")
print(f"   Safety Net: {enable_net_arg}")

env = DummyVecEnv([lambda: CryptoTradingEnvDiagnostic(
    df_train,
    symbol=env_symbol_arg,
    enable_safety_net=enable_net_arg,
    diagnostic_mode=True,
    diagnostic_log_path='diagnostics_v15_3'
)])

obs_shape = env.observation_space.shape[0]
print(f"✅ Diagnostic Environment V15.3 ready. Obs shape: {obs_shape}")
if obs_shape != 9:
    raise ValueError(
        f"❌ CRITICAL ERROR! Expected 9 features, got {obs_shape}!")

# === DQN AGENT SETUP ===
print(f"\n🤖 Initializing DQN diagnostic agent...")
model_dqn = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=0.00005,
    buffer_size=500000,
    learning_starts=10000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)

# === TRAINING ===
print(f"\n{'='*70}")
print(
    f"🚀 STARTING V15.3 DIAGNOSTIC: {symbol_lower.upper()} [{scenario.upper()}]")
print(f"{'='*70}")
print(f"📊 Diagnostic steps: {TOTAL_TIMESTEPS:,}")
print(f"🎯 Expected improvements:")
print(f"   - Trade count 50–150 (tidak full hit limit)")
print(f"   - Jauh lebih sedikit hard-block spam")
print(f"   - Reward lebih stabil (idle vs blocked vs profit)")
print(f"{'='*70}\n")

tqdm_cb = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[tqdm_cb])

print(f"\n{'='*70}")
print(f"✅ V15.3 Diagnostic training completed!")
print(f"📁 Diagnostic logs saved to: diagnostics_v15_3/")
print(f"\n🔍 Next steps:")
print(f"   1. Run: python analyze_diagnostics.py --version v15_3 (opsional arg)")
print(f"   2. Evaluate trade count + return + blocks")
print(f"   3. Kalau OK, baru di-port ke environment production")
print(f"{'='*70}\n")
