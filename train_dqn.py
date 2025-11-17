import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv
import os
import argparse

# --- Setup Argumen ---
parser = argparse.ArgumentParser(
    description='Train DQN Agent V4 (Fixed Inactivity).')
parser.add_argument('--symbol', type=str, default='btc',
                    help='Symbol to train on (btc, eth, xrp)')
parser.add_argument('--scenario', type=str, default='adaptive',
                    choices=['adaptive', 'default', 'baseline'],
                    help='Training scenario for ablation study')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()
data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"

# Buat nama model dan data tes yang unik untuk skenario ini
model_save_name = f"dqn_agent_{symbol_lower}_{scenario}.zip"
test_data_save_name = f"test_data_{symbol_lower}_{scenario}.csv"

print(f"\n{'='*60}")
print(
    f"🚀 MEMULAI TRAINING: [{symbol_lower.upper()}] Skenario: [{scenario.upper()}]")
print(f"{'='*60}")

# --- 1. SIAPKAN DATA & PREDIKSI HYBRID (V2 DENGAN ATR) ---
print(f"📂 Memuat data {symbol_lower} dari {data_filename}...")
try:
    df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print(f"❌ Error: File {data_filename} tidak ditemukan.")
    exit()

print("🛠️ Feature Engineering V2 (with ATR)...")
df = df.ffill()
df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
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

# === DROP NaN SETELAH FEATURE ENGINEERING ===
df.dropna(inplace=True)
print(f"📊 Data setelah feature engineering: {len(df)} baris")

print(f"🧠 Memuat model LR & LSTM spesifik untuk {symbol_lower}...")
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
    print(
        f"❌ Error: Model {symbol_lower} tidak ditemukan di 'ml_models/'. {e}")
    exit()

print("🔮 Menghasilkan prediksi Hybrid untuk seluruh data...")
features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR']
X_all = df[features]
X_scaled_lr = scaler_lr.transform(X_all)
pred_lr_all = model_lr.predict(X_scaled_lr)
SEQ_LENGTH = 60
pred_lstm_all = np.full(len(df), np.nan)
X_scaled_lstm = scaler_lstm_features.transform(X_all.values)
xs = []
for i in range(len(X_scaled_lstm) - SEQ_LENGTH):
    xs.append(X_scaled_lstm[i:(i + SEQ_LENGTH)])
X_lstm_seq = np.array(xs)
pred_lstm_seq_scaled = model_lstm.predict(
    X_lstm_seq, verbose=0, batch_size=2048)
pred_lstm_seq = scaler_lstm_target.inverse_transform(
    pred_lstm_seq_scaled).flatten()
pred_lstm_all[SEQ_LENGTH:] = pred_lstm_seq
df['prediction'] = (0.8 * pred_lr_all) + (0.2 * pred_lstm_all)

# === DROP NaN SETELAH PREDIKSI ===
df.dropna(inplace=True)
print(f"✅ Data siap untuk trading ({symbol_lower}): {len(df)} jam")

# --- 2. SPLIT DATA UNTUK RL ---
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]
print(f"Training RL {symbol_lower} dengan {len(df_train)} jam data.")

# --- 3. SETUP ENVIRONMENT V4 (REVISED) ---
env_symbol_arg = symbol_lower
enable_net_arg = True

if scenario == 'baseline':
    enable_net_arg = False
elif scenario == 'default':
    env_symbol_arg = 'default'  # Paksa gunakan threshold default di env

print(
    f"Inisialisasi Env: symbol='{env_symbol_arg}', enable_safety_net={enable_net_arg}")

env = DummyVecEnv([lambda: CryptoTradingEnv(
    df_train,
    symbol=env_symbol_arg,
    enable_safety_net=enable_net_arg
)])

print(
    f"✅ Environment [{scenario.upper()}] siap. Obs shape: {env.observation_space.shape[0]}")

# --- 4. SETUP AGENT DQN (FIX 2: LONGER EXPLORATION) ---
print(f"🤖 Menginisialisasi agen DQN untuk [{scenario.upper()}]...")
model_dqn = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001,
    buffer_size=500000,
    learning_starts=10000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    # === REVISED: LONGER EXPLORATION ===
    exploration_fraction=0.3,      # Naik dari 0.1 → 0.3 (explore 300k steps)
    exploration_initial_eps=1.0,   # Mulai dari 100% random
    exploration_final_eps=0.05     # Turun ke 5% (lebih tinggi dari 2%)
)

# --- 5. MULAI TRAINING ---
TOTAL_TIMESTEPS = 1_000_000
print(
    f"\n🚀 === MEMULAI TRAINING {symbol_lower.upper()} [{scenario.upper()}] (1 JUTA LANGKAH) ===")
print("⚠️ VERSI V4: Reward tuning + Longer exploration + 50% initial position")
print("Ini akan memakan waktu SANGAT LAMA. Harap bersabar...")
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=100)
print(f"✅ Training DQN [{scenario.upper()}] selesai!")

# --- 6. SIMPAN MODEL & DATA TEST ---
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, model_save_name)
model_dqn.save(model_path)
print(f"💾 Model agen [{scenario.upper()}] disimpan di: {model_path}")

test_data_path = os.path.join(model_dir, test_data_save_name)
df_test.to_csv(test_data_path)
print(f"💾 Data test [{scenario.upper()}] disimpan di: {test_data_path}")
print(f"--- ✅ SELESAI: {symbol_lower.upper()} [{scenario.upper()}] ---")
