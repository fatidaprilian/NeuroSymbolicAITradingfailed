import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from binance.client import Client
from stable_baselines3 import DQN
import os

client = Client()
MODELS = {}


def load_all_models():
    """Memuat semua model ML dan RL ke memori."""
    global MODELS
    base_path = "ml_models/"
    print("📂 Loading models...")
    # Kita butuh flag untuk tahu model apa saja yang berhasil dimuat
    loaded_symbols = []

    # Coba load model untuk setiap koin
    for symbol in ['btc', 'eth', 'xrp']:
        try:
            MODELS[f'lr_{symbol}'] = joblib.load(os.path.join(
                base_path, f'model_lr_baseline_{symbol}.pkl'))
            MODELS[f'scaler_lr_{symbol}'] = joblib.load(
                os.path.join(base_path, f'scaler_lr_{symbol}.pkl'))
            MODELS[f'lstm_{symbol}'] = load_model(os.path.join(
                base_path, f'best_lstm_model_{symbol}.keras'))
            MODELS[f'scaler_lstm_feat_{symbol}'] = joblib.load(
                os.path.join(base_path, f'scaler_lstm_features_{symbol}.pkl'))
            MODELS[f'scaler_lstm_target_{symbol}'] = joblib.load(
                os.path.join(base_path, f'scaler_lstm_target_{symbol}.pkl'))
            print(f"🤖 Loading DQN Agent for {symbol}...")
            MODELS[f'dqn_{symbol}'] = DQN.load(os.path.join(
                base_path, f'dqn_trading_agent_{symbol}.zip'))
            loaded_symbols.append(symbol.upper())
        except Exception as e:
            print(f"⚠️ Warning: Gagal memuat model untuk {symbol}. Error: {e}")

    if not loaded_symbols:
        print(
            "❌ CRITICAL: Tidak ada model yang berhasil dimuat. Backend tidak bisa trading.")
        return False

    print(f"✅ Model berhasil dimuat untuk: {', '.join(loaded_symbols)}")
    return True


def fetch_live_data(symbol="BTCUSDT", limit=500):
    """Tarik data candlestick terbaru untuk simbol tertentu."""
    klines = client.get_historical_klines(
        symbol, Client.KLINE_INTERVAL_1HOUR, limit=limit)
    df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume',
                      'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.drop(columns=['open_time'], inplace=True)
    for col in df.columns:
        df[col] = df[col].astype(float)
    return df


def generate_features(df):
    """Feature Engineering V2: Ditambah Volatility (ATR)."""
    df = df.copy()
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
    return df


def get_hybrid_prediction(symbol_lower='btc'):
    """Pipeline prediksi (LR + LSTM) untuk koin spesifik."""
    symbol_upper = f"{symbol_lower.upper()}USDT"

    # Cek apakah model untuk simbol ini ada
    if f'lr_{symbol_lower}' not in MODELS:
        return None, f"Model untuk {symbol_lower} tidak dimuat."

    df = fetch_live_data(symbol=symbol_upper, limit=200)
    df_features = generate_features(df)
    last_row = df_features.iloc[-1:]

    features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
                'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR']

    X_lr = last_row[features]
    X_lr_scaled = MODELS[f'scaler_lr_{symbol_lower}'].transform(X_lr)
    pred_lr = MODELS[f'lr_{symbol_lower}'].predict(X_lr_scaled)[0]

    df_clean = df_features.dropna()
    if len(df_clean) < 60:
        return None, "Not enough data for LSTM"

    last_60 = df_clean[features].values[-60:]
    last_60_scaled = MODELS[f'scaler_lstm_feat_{symbol_lower}'].transform(
        last_60)
    X_lstm = np.array([last_60_scaled])

    pred_lstm_scaled = MODELS[f'lstm_{symbol_lower}'].predict(
        X_lstm, verbose=0)
    pred_lstm = MODELS[f'scaler_lstm_target_{symbol_lower}'].inverse_transform(
        pred_lstm_scaled)[0][0]

    pred_hybrid = (0.8 * pred_lr) + (0.2 * pred_lstm)
    current_price = df['close'].iloc[-1]

    return {
        "timestamp": str(df.index[-1]),
        "current_price": current_price,
        "prediction_next_hour": float(pred_hybrid),
        "predicted_change_percent": ((pred_hybrid - current_price) / current_price) * 100,
        "components": {"lr_prediction": float(pred_lr), "lstm_prediction": float(pred_lstm)}
    }, None


def get_historical_data(symbol_lower='btc', hours=24):
    """Data untuk grafik candlestick."""
    df = fetch_live_data(symbol=f"{symbol_lower.upper()}USDT", limit=hours)
    chart_data = []
    for index, row in df.iterrows():
        chart_data.append({
            "x": int(index.timestamp() * 1000),
            "y": [row['open'], row['high'], row['low'], row['close']]
        })
    return chart_data


def run_automated_trading_job(symbol_lower='btc'):
    """Fungsi Scheduler: Otak otomatisasi dengan ATR."""
    from .binance_utils import get_testnet_balance
    from .trading_executor import execute_trade

    symbol_upper = f"{symbol_lower.upper()}USDT"
    print(f"\n⏰ === JOB {symbol_upper} DIMULAI ===")

    # Cek apakah model untuk koin ini siap
    if f'dqn_{symbol_lower}' not in MODELS:
        print(f"❌ Job {symbol_upper} Skipped: Model DQN tidak dimuat.")
        return

    # 1. Analisis Pasar
    print(f"🔍 Market Analysis {symbol_upper}...")
    pred_result, error = get_hybrid_prediction(symbol_lower)
    if error:
        print(f"❌ Job {symbol_upper} Skipped: Prediction error ({error})")
        return

    # 2. Cek Dompet
    print("💰 Checking Wallet...")
    balances, error = get_testnet_balance(
        symbol_lower)  # Minta saldo koin spesifik
    if error:
        print(f"❌ Job {symbol_upper} Skipped: Wallet error ({error})")
        return

    current_price = pred_result['current_price']
    # Hitung net worth
    net_worth = balances['usdt'] + (balances[symbol_lower] * current_price)

    # 3. Siapkan Data untuk Agen DQN & Safety Net
    df_latest = fetch_live_data(symbol=symbol_upper, limit=100)
    df_features = generate_features(df_latest)
    last_row = df_features.iloc[-1]

    # Hitung NATR (Normalized ATR)
    current_atr = last_row['ATR']
    natr = (current_atr / current_price) * 100 if current_price > 0 else 0

    # State untuk DQN (9 input)
    obs = np.array([
        current_price, pred_result['prediction_next_hour'],
        last_row['RSI'], last_row['MACD'], last_row['SMA_7'],
        current_atr,  # Kirim ATR (bukan NATR, karena model dilatih pakai ATR)
        balances['usdt'], balances[symbol_lower], net_worth
    ], dtype=np.float32)

    # Data untuk Safety Net (Lapis Logika)
    market_state = {
        'symbol_upper': symbol_upper,
        'symbol_lower': symbol_lower,
        'price': current_price,
        'sma_30': last_row['SMA_30'],
        'rsi': last_row['RSI'],
        'natr': natr  # Kirim NATR (dalam %) untuk filter volatilitas
    }

    # 4. Ambil Keputusan AI
    print(f"🤖 AI {symbol_upper} Thinking...")
    action, _ = MODELS[f'dqn_{symbol_lower}'].predict(obs, deterministic=True)
    if isinstance(action, np.ndarray):
        action = action.item()
    print(
        f"💡 AI Decision {symbol_upper}: {['SELL', 'HOLD', 'BUY'][action]} (Code: {action})")

    # 5. Eksekusi
    execute_trade(action, market_state)
    print(f"✅ === JOB {symbol_upper} FINISHED ===\n")
