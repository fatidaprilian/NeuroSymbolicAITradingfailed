import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
# Pastikan trading_env yang di-import adalah versi yang sudah direvisi
from trading_env import CryptoTradingEnv
import argparse

# --- Setup Argumen ---
parser = argparse.ArgumentParser(
    description='Run Sinta 2 Backtest (Multi-Coin + ATR Filter).')
parser.add_argument('--symbol',
                    type=str,
                    default='btc',
                    help='Symbol to backtest on (e.g., btc, eth, xrp)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
MODEL_DIR = "ml_models"

# (Konfigurasi VOLATILITY_THRESHOLD_PERCENT tidak diperlukan lagi di sini,
# karena sudah ditangani di dalam trading_env.py)

# --- FUNGSI METRIK FINANSIAL ---


def calculate_metrics(net_worth_series):
    df = pd.DataFrame(net_worth_series, columns=['net_worth'])
    df['returns'] = df['net_worth'].pct_change()

    initial = df['net_worth'].iloc[0]
    final = df['net_worth'].iloc[-1]
    total_return = (final - initial) / initial * 100

    mean_return = df['returns'].mean()
    std_return = df['returns'].std()
    sharpe_ratio = (mean_return / std_return) * \
        np.sqrt(8760) if std_return > 0 else 0

    df['cummax'] = df['net_worth'].cummax()
    df['drawdown'] = (df['net_worth'] - df['cummax']) / df['cummax']
    max_drawdown = df['drawdown'].min() * 100

    return total_return, sharpe_ratio, max_drawdown


# --- 1. LOAD DATA TEST ---
print(
    f"--- 🚀 MEMULAI FINAL BACKTEST ({symbol_lower.upper()}) DENGAN FILTER ATR ---")
print(f"📂 Memuat data test V2 {symbol_lower}...")
try:
    test_data_path = os.path.join(
        MODEL_DIR, f'test_data_for_backtest_{symbol_lower}.csv')
    df_test = pd.read_csv(
        test_data_path, index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print(f"❌ Error: File {test_data_path} tidak ditemukan.")
    exit()

# --- 2. LOAD AGEN ---
print(f"🤖 Memuat agen DQN {symbol_lower}...")
try:
    model_path = os.path.join(MODEL_DIR, f"dqn_trading_agent_{symbol_lower}")
    model_dqn = DQN.load(model_path)
except Exception as e:
    print(f"❌ Error: Model DQN {symbol_lower} tidak ditemukan.")
    exit()

# --- 3. SIMULASI BACKTEST (REVISI) ---
print("🚀 Memulai Backtesting (Mode: AI + Safety Net dari Environment)...")
# Gunakan Env (yang sudah direvisi) dengan data test
env = CryptoTradingEnv(df_test)
obs, _ = env.reset()

net_worth_history = [env.initial_balance]
timestamps = [df_test.index[0]]
trades = []  # Kita tetap catat trades untuk visualisasi

# (Counter trigger tidak lagi relevan di sini, karena env yang menangani)

for i in range(len(df_test) - 1):
    action_ai, _ = model_dqn.predict(obs, deterministic=True)
    if isinstance(action_ai, np.ndarray):
        action_ai = action_ai.item()

    # ======================================================================
    # === BAGIAN REVISI: LOGIKA SAFETY NET MANUAL DIHAPUS ===
    # ======================================================================
    #
    # Blok '3 LAPIS SAFETY NET (Symbolic Layer)' yang sebelumnya
    # ada di sini (if is_too_volatile, if is_downtrend, dll.)
    # sekarang DIHAPUS.
    #
    # Kita langsung serahkan aksi mentah 'action_ai' ke environment,
    # dan biarkan 'trading_env.py' yang baru menangani aturannya.
    #
    # ======================================================================

    # Ambil state sebelum step untuk mencatat trade (jika terjadi)
    prev_balance_usdt = env.balance_usdt
    prev_balance_btc = env.balance_btc

    # Langsung panggil step dengan aksi mentah dari AI
    obs, reward, done, _, info = env.step(action_ai)

    # Cek apakah trade benar-benar terjadi (dengan membandingkan saldo)
    # Ini diperlukan agar visualisasi chart sesuai dengan apa yang
    # *benar-benar dieksekusi* oleh environment
    current_price = df_test.iloc[i]['close']
    if env.balance_btc > prev_balance_btc and prev_balance_usdt > 10:
        trades.append(
            {'time': df_test.index[i], 'type': 'BUY', 'price': current_price})
    elif env.balance_usdt > prev_balance_usdt and prev_balance_btc > 0.0001:
        trades.append(
            {'time': df_test.index[i], 'type': 'SELL', 'price': current_price})

    net_worth_history.append(info['net_worth'])
    timestamps.append(df_test.index[i+1])

    if done:
        break

# --- 4. EVALUASI & LAPORAN ---
bot_ret, bot_sharpe, bot_mdd = calculate_metrics(net_worth_history)
hodl_net_worth = (df_test['close'] /
                  df_test['close'].iloc[0]) * env.initial_balance
hodl_ret, hodl_sharpe, hodl_mdd = calculate_metrics(hodl_net_worth.values)

win, loss, last_buy = 0, 0, 0
for t in trades:
    if t['type'] == 'BUY':
        last_buy = t['price']
    elif t['type'] == 'SELL' and last_buy > 0:
        if t['price'] > last_buy:
            win += 1
        else:
            loss += 1
total_closed = win + loss
win_rate = (win / total_closed * 100) if total_closed > 0 else 0

print("\n" + "="*50)
print(f"📊 LAPORAN FINAL SINTA 2 ({symbol_lower.upper()} + FILTER ATR)")
print("="*50)
print(f"Training Steps       : 1,000,000")
# (Safety Net Triggers tidak bisa dihitung dari sini lagi,
#  tapi itu tidak masalah, hasil akhir lebih penting)
print("-"*50)
print(f"{'Metric':<20} | {'Final Bot':<12} | {'Benchmark (HODL)':<12}")
print("-"*50)
print(f"{'Total Return':<20} | {bot_ret:>10.2f}%  | {hodl_ret:>10.2f}%")
print(f"{'Sharpe Ratio':<20} | {bot_sharpe:>10.4f}  | {hodl_sharpe:>10.4f}")
print(f"{'Max Drawdown':<20} | {bot_mdd:>10.2f}%  | {hodl_mdd:>10.2f}%")
print("-"*50)
print(f"Trades: {len(trades)} | Win Rate: {win_rate:.1f}% ({win}W/{loss}L)")
print("="*50)

# (Kode visualisasi matplotlib tetap sama)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(timestamps, net_worth_history,
         label=f'Final Bot ({symbol_lower.upper()})', color='blue', linewidth=2)
plt.plot(df_test.index, hodl_net_worth, label='Buy & Hold',
         color='gray', linestyle='--', alpha=0.6)
plt.title(f'Equity Curve: {symbol_lower.upper()} vs HODL (w/ ATR Filter)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.subplot(2, 1, 2)
plt.plot(df_test.index, df_test['close'],
         label=f'{symbol_lower.upper()} Price', color='black', alpha=0.5)
buys = [t for t in trades if t['type'] == 'BUY']
sells = [t for t in trades if t['type'] == 'SELL']
plt.scatter([b['time'] for b in buys], [b['price']
            for b in buys], color='green', marker='^', s=50, label='BUY')
plt.scatter([s['time'] for s in sells], [s['price']
            for s in sells], color='red', marker='v', s=50, label='SELL')
plt.title('Trading Executions on Price Chart')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
