import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import CryptoTradingEnv  # Pastikan ini V2 (dengan 9 input)
import os
import argparse  # Library untuk membaca argumen terminal

# --- Setup Argumen ---
parser = argparse.ArgumentParser(
    description='Train DQN Agent V2 (Multi-Coin).')
parser.add_argument('--symbol',
                    type=str,
                    default='btc',
                    help='Symbol to train on (e.g., btc, eth, xrp)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
data_filename = f"{symbol_lower}_1h_data.csv"
# <- Variabel didefinisikan sebagai 'model_dir' (lowercase)
model_dir = "ml_models"

print(
    f"--- 🚀 MEMULAI TRAINING DQN V2 (1 JUTA LANGKAH) UNTUK: {symbol_lower.upper()} ---")

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
df.dropna(inplace=True)

# --- LOAD MODEL PREDIKSI SPESIFIK KOIN ---
print(f"🧠 Memuat model LR & LSTM spesifik untuk {symbol_lower}...")
try:
    # Menggunakan 'model_dir' (lowercase) yang benar
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
    print("💡 Pastikan kamu sudah menjalankan train_baseline.py dan train_lstm.py untuk simbol ini.")
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

print(f"   ↳ Menjalankan inferensi LSTM massal untuk {symbol_lower}...")
pred_lstm_seq_scaled = model_lstm.predict(
    X_lstm_seq, verbose=0, batch_size=2048)
pred_lstm_seq = scaler_lstm_target.inverse_transform(
    pred_lstm_seq_scaled).flatten()
pred_lstm_all[SEQ_LENGTH:] = pred_lstm_seq

df['prediction'] = (0.8 * pred_lr_all) + (0.2 * pred_lstm_all)
df.dropna(inplace=True)
print(f"Data siap untuk trading ({symbol_lower}): {len(df)} jam")

# --- 2. SPLIT DATA UNTUK RL ---
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]
print(f"Training RL {symbol_lower} dengan {len(df_train)} jam data.")

# --- 3. SETUP ENVIRONMENT V2 (Input 9 fitur) ---
env = DummyVecEnv([lambda: CryptoTradingEnv(df_train)])
print(
    f"✅ Environment {symbol_lower} siap. Observation shape: {env.observation_space.shape[0]}")

# --- 4. SETUP AGENT DQN ---
print(f"🤖 Menginisialisasi agen DQN untuk {symbol_lower}...")
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
    exploration_fraction=0.1,
    exploration_final_eps=0.02
)

# --- 5. MULAI TRAINING (SINTA 2 GRIND) ---
TOTAL_TIMESTEPS = 1_000_000
print(f"\n🚀 === MEMULAI TRAINING {symbol_lower.upper()} (1 JUTA LANGKAH) ===")
print("Ini akan memakan waktu SANGAT LAMA. Harap bersabar...")
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=100)
print(f"✅ Training DQN {symbol_lower} selesai!")

# --- 6. SIMPAN MODEL & DATA TEST ---
# Menggunakan 'model_dir' (lowercase) yang benar
model_path = os.path.join(model_dir, f"dqn_trading_agent_{symbol_lower}")
model_dqn.save(model_path)
print(f"💾 Model agen {symbol_lower} disimpan di: {model_path}.zip")

test_data_path = os.path.join(
    model_dir, f'test_data_for_backtest_{symbol_lower}.csv')
df_test.to_csv(test_data_path)
print(f"💾 Data test {symbol_lower} disimpan di: {test_data_path}")
print(f"--- ✅ SELESAI: {symbol_lower.upper()} ---")
