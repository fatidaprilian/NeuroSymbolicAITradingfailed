import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import argparse  # Library untuk membaca argumen terminal

# --- Setup Argumen ---
parser = argparse.ArgumentParser(
    description='Train LSTM V2 model (Multi-Coin).')
parser.add_argument('--symbol',
                    type=str,
                    default='btc',
                    help='Symbol to train on (e.g., btc, eth, xrp)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"

print(f"--- 🚀 MEMULAI TRAINING LSTM V2 UNTUK: {symbol_lower.upper()} ---")

# --- 1. LOAD DATA ---
print(f"📂 Memuat data {symbol_lower} dari {data_filename}...")
try:
    df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print(
        f"❌ Error: File {data_filename} tidak ditemukan. Jalankan fetch_data.py --symbol {symbol_lower}USDT dulu.")
    exit()

# --- 2. FEATURE ENGINEERING (V2 with ATR) ---
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

# --- 3. PERSIAPAN DATA LSTM ---
features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR']
data = df[features].values
target = df['close'].values

# Normalisasi sangat penting untuk LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = target_scaler.fit_transform(target.reshape(-1, 1))

# Buat sequences (sliding window)


def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(target[i + seq_length])
    return np.array(xs), np.array(ys)


SEQ_LENGTH = 60
X, y = create_sequences(data_scaled, target_scaled, SEQ_LENGTH)

# Split data: 85% train, 15% validation
train_size = int(len(X) * 0.85)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]
print(
    f"Data Training LSTM ({symbol_lower}): {X_train.shape} | Data Validation: {X_val.shape}")

# --- 4. BANGUN & TRAIN MODEL ---
print("🏗️ Membangun arsitektur LSTM V2...")
model = Sequential([
    LSTM(units=64, return_sequences=True, input_shape=(
        X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Nama file dinamis untuk menyimpan model
model_path = os.path.join(model_dir, f'best_lstm_model_{symbol_lower}.keras')
scaler_feat_path = os.path.join(
    model_dir, f'scaler_lstm_features_{symbol_lower}.pkl')
scaler_target_path = os.path.join(
    model_dir, f'scaler_lstm_target_{symbol_lower}.pkl')

checkpoint = ModelCheckpoint(
    model_path, monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

print(f"🚀 Mulai training LSTM V2 for {symbol_lower} (30 Epochs)...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint, early_stop],
    verbose=1
)

# --- 5. SIMPAN MODEL & SCALERS ---
joblib.dump(scaler, scaler_feat_path)
joblib.dump(target_scaler, scaler_target_path)
print(f"✅ Training LSTM {symbol_lower} selesai & tersimpan di {model_dir}/")
print(f"--- ✅ SELESAI: {symbol_lower.upper()} ---")
