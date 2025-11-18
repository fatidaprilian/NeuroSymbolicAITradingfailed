import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env_diagnostic import CryptoTradingEnvDiagnostic
import os
import argparse

# === ARGUMENT PARSER ===
parser = argparse.ArgumentParser(description='DQN Diagnostic Training')
parser.add_argument('--symbol', type=str, default='btc',
                    help='Symbol to train (btc/eth/xrp)')
parser.add_argument('--scenario', type=str, default='adaptive',
                    choices=['adaptive', 'default', 'baseline'])
parser.add_argument('--steps', type=int, default=50000,
                    help='Training timesteps (default: 50k for diagnostic)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()
TOTAL_TIMESTEPS = args.steps

print(f"\n{'='*60}")
print(f"🔬 DIAGNOSTIC TRAINING: [{symbol_lower.upper()}] [{scenario.upper()}]")
print(f"   Quick run: {TOTAL_TIMESTEPS:,} steps for analysis purposes")
print(f"{'='*60}")

# === PATHS ===
data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"
diagnostic_dir = "diagnostics"

# === 1. LOAD DATA ===
print(f"📂 Loading data {symbol_lower} dari {data_filename}...")
try:
    df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print(f"❌ Error: File {data_filename} tidak ditemukan.")
    exit()

df = df.ffill()

# === 2. FEATURE ENGINEERING ===
print("🛠️ Feature Engineering...")
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

# === 3. LOAD PREDICTION MODELS ===
print(f"🧠 Loading LR & LSTM models for {symbol_lower}...")
try:
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
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. {e}")
    exit()

# === 4. GENERATE PREDICTIONS ===
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

# === 5. TRAIN-TEST SPLIT ===
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
print(f"Training with {len(df_train)} hours of data.")

# === 6. CREATE DIAGNOSTIC ENVIRONMENT ===
env_symbol_arg = 'default' if scenario == 'default' else symbol_lower
enable_net_arg = False if scenario == 'baseline' else True

print(f"🔬 Creating DIAGNOSTIC environment...")
print(f"   symbol='{env_symbol_arg}', enable_safety_net={enable_net_arg}")
print(f"   diagnostic_mode=True, logging to '{diagnostic_dir}/' folder")

env = DummyVecEnv([lambda: CryptoTradingEnvDiagnostic(
    df_train,
    symbol=env_symbol_arg,
    enable_safety_net=enable_net_arg,
    diagnostic_mode=True,
    diagnostic_log_path=diagnostic_dir
)])

# === 7. CREATE DQN AGENT ===
print(f"🤖 Initializing DQN agent...")
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

# === 8. TRAIN ===
print(f"🚀 Starting diagnostic training ({TOTAL_TIMESTEPS:,} steps)...")
print("   This will take ~10-15 minutes for 50k steps")
print("   (vs 2-3 hours for full 1M steps)\n")

model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS)

print(f"\n✅ Diagnostic training completed!")
print(f"📊 Diagnostic logs saved in: {diagnostic_dir}/")
print(f"\nNext steps:")
print(f"  1. Run: python analyze_diagnostics.py")
print(f"  2. Read the analysis report")
print(f"  3. Implement recommended fixes")
print(f"  4. Re-run this diagnostic to verify\n")
