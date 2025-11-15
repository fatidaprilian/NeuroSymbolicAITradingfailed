import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import argparse  # Library untuk membaca argumen terminal

# --- Setup Argumen ---
parser = argparse.ArgumentParser(
    description='Train Linear Regression baseline model V2 (Multi-Coin).')
# Kita set default 'btc' agar jika dijalankan tanpa argumen, dia tetap mentraining BTC
parser.add_argument('--symbol',
                    type=str,
                    default='btc',
                    help='Symbol to train on (e.g., btc, eth, xrp)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"

print(f"--- 🚀 MEMULAI TRAINING BASELINE UNTUK: {symbol_lower.upper()} ---")

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

# === NEW: ATR (Average True Range) ===
df['tr1'] = df['high'] - df['low']
df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR'] = df['tr'].rolling(window=14).mean()

df['target'] = df['close'].shift(-1)
df.dropna(inplace=True)

# --- 3. SPLIT DATA ---
features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR']
X = df[features]
y = df['target']

train_size = int(len(df) * 0.7)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

print(
    f"Data Training ({symbol_lower}): {len(X_train)} | Data Testing: {len(X_test)}")

# --- 4. NORMALISASI & TRAINING ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n🚀 Mulai training Linear Regression V2 for {symbol_lower}...")
model_lr = LinearRegression()
model_lr.fit(X_train_scaled, y_train)

# --- 5. EVALUASI & SIMPAN ---
y_pred = model_lr.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Training {symbol_lower} Selesai. RMSE: {rmse:.2f}, R2: {r2:.4f}")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Simpan model dengan nama dinamis
model_path = os.path.join(model_dir, f'model_lr_baseline_{symbol_lower}.pkl')
scaler_path = os.path.join(model_dir, f'scaler_lr_{symbol_lower}.pkl')

joblib.dump(model_lr, model_path)
joblib.dump(scaler, scaler_path)
print(f"💾 Model LR {symbol_lower} tersimpan di {model_dir}/")
print(f"--- ✅ SELESAI: {symbol_lower.upper()} ---")
