import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
print("📂 Memuat data...")
try:
    df = pd.read_csv('btc_1h_data.csv',
                     index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print("❌ Error: File 'btc_1h_data.csv' tidak ditemukan.")
    exit()

# --- 2. FEATURE ENGINEERING (MANUAL VERSION - SAMA PERSIS DENGAN BASELINE) ---
print("🛠️ Membuat indikator teknikal (Re-generating features)...")
# Pastikan proses ini SAMA PERSIS dengan yang dilakukan di train_baseline.py
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
df['target'] = df['close'].shift(-1)
df.dropna(inplace=True)

features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower']
X = df[features]
y = df['target']

# --- 3. PREPARE TEST DATA ---
# Kita butuh data Test yang belum pernah dilihat model saat training
train_size = int(len(df) * 0.70)
val_size = int(len(df) * 0.15)
test_start_index = train_size + val_size

# Data untuk Linear Regression (langsung potong)
X_test_lr_raw = X.iloc[test_start_index:]
y_test_raw = y.iloc[test_start_index:]

print(f"Data Test Raw: {len(X_test_lr_raw)} baris")

# --- 4. LOAD MODELS & SCALERS ---
print("📂 Memuat model dan scaler yang sudah tersimpan...")
try:
    model_lr = joblib.load('model_lr_baseline.pkl')
    scaler_lr = joblib.load('scaler_lr.pkl')
    model_lstm = load_model('best_lstm_model.keras')
    scaler_lstm_features = joblib.load('scaler_lstm_features.pkl')
    scaler_lstm_target = joblib.load('scaler_lstm_target.pkl')
except FileNotFoundError as e:
    print(
        f"❌ Error: {e}. Pastikan kamu sudah jalankan train_baseline.py DAN train_lstm.py sampai sukses!")
    exit()

# --- 5. GENERATE PREDICTIONS ---
print("🔮 Menghasilkan prediksi dari masing-masing model...")

# A. Prediksi Linear Regression
X_test_lr_scaled = scaler_lr.transform(X_test_lr_raw)
pred_lr = model_lr.predict(X_test_lr_scaled)

# B. Prediksi LSTM
# LSTM butuh sequence 60 jam ke belakang.
SEQ_LENGTH = 60
# Ambil data mundur 60 jam dari titik mulai test agar prediksi pertama tidak hilang
X_test_lstm_raw = X.iloc[test_start_index - SEQ_LENGTH:].values
# Normalisasi pakai scaler milik LSTM
X_test_lstm_scaled = scaler_lstm_features.transform(X_test_lstm_raw)

# Fungsi pembentuk sequence


def create_sequences_test(data, seq_length):
    xs = []
    # Loop hanya sebanyak data test yang asli
    num_samples = len(data) - seq_length
    for i in range(num_samples):
        xs.append(data[i:(i + seq_length)])
    return np.array(xs)


X_test_lstm_seq = create_sequences_test(X_test_lstm_scaled, SEQ_LENGTH)
print(f"Shape input LSTM: {X_test_lstm_seq.shape}")

# Prediksi LSTM
pred_lstm_scaled = model_lstm.predict(X_test_lstm_seq, verbose=0)
pred_lstm = scaler_lstm_target.inverse_transform(pred_lstm_scaled).flatten()

# --- 6. ALIGN DATA (PENTING!) ---
# Pastikan semua array punya panjang yang sama persis sebelum digabung
min_len = min(len(pred_lr), len(pred_lstm), len(y_test_raw))
pred_lr = pred_lr[:min_len]
pred_lstm = pred_lstm[:min_len]
y_test_final = y_test_raw.iloc[:min_len].values

print(f"Jumlah data final untuk evaluasi: {min_len}")

# --- 7. HYBRID COMBINATION ---
print("⚗️ Meracik Hybrid Model...")
# Bobot: LR 80%, LSTM 20% (karena performa LR jauh lebih bagus tadi)
w_lr = 0.8
w_lstm = 0.2
pred_hybrid = (w_lr * pred_lr) + (w_lstm * pred_lstm)

# --- 8. EVALUASI AKHIR ---
rmse_lr = np.sqrt(mean_squared_error(y_test_final, pred_lr))
rmse_lstm = np.sqrt(mean_squared_error(y_test_final, pred_lstm))
rmse_hybrid = np.sqrt(mean_squared_error(y_test_final, pred_hybrid))
r2_hybrid = r2_score(y_test_final, pred_hybrid)

print("\n" + "="*40)
print("📊 HASIL PERBANDINGAN AKHIR (TEST SET)")
print("="*40)
print(f"RMSE Linear Regression : {rmse_lr:.2f}")
print(f"RMSE LSTM Model        : {rmse_lstm:.2f}")
print("-" * 40)
print(f"✅ RMSE HYBRID MODEL   : {rmse_hybrid:.2f}")
print(f"✅ R2 Score Hybrid     : {r2_hybrid:.4f}")
print("="*40)

# Kesimpulan otomatis
if rmse_hybrid < rmse_lr and rmse_hybrid < rmse_lstm:
    print("🎉 KESIMPULAN: Hybrid Model BERHASIL mengalahkan kedua model individu!")
elif rmse_hybrid < rmse_lstm:
    print("⚠️ KESIMPULAN: Hybrid lebih baik dari LSTM, tapi masih kalah dari Linear Regression murni.")
else:
    print("❌ KESIMPULAN: Hybrid Model gagal meningkatkan performa.")

# --- 9. VISUALISASI ---
plt.figure(figsize=(12, 6))
# Plot 150 jam terakhir agar terlihat detail perbedaannya
last_n = 150
plt.plot(y_test_final[-last_n:], label='Harga Asli (Actual)',
         color='black', linewidth=2.5, alpha=0.7)
# plt.plot(pred_lr[-last_n:], label='Prediksi LR', color='blue', linestyle='--', alpha=0.5) # Opsional: nyalakan jika ingin lihat garis LR
# plt.plot(pred_lstm[-last_n:], label='Prediksi LSTM', color='green', linestyle=':', alpha=0.5) # Opsional: nyalakan jika ingin lihat garis LSTM
plt.plot(pred_hybrid[-last_n:], label='Prediksi Hybrid (Final)',
         color='red', linewidth=2, alpha=0.9)
plt.title(f'FINAL HYBRID MODEL: Actual vs Predicted ({last_n} Jam Terakhir)')
plt.xlabel('Jam ke-')
plt.ylabel('Harga BTC (USDT)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
