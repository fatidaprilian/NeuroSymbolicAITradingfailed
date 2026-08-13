import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
from trading_env import CryptoTradingEnv  # Impor V3 Anda
import argparse
import json
from datetime import datetime

# --- Setup Argumen ---
parser = argparse.ArgumentParser(description='Run Single Backtest (V3 Env)')
parser.add_argument('--symbol', type=str, default='btc',
                    help='Symbol to test (btc, eth, xrp)')
parser.add_argument('--scenario', type=str, default='adaptive',
                    choices=['adaptive', 'default', 'baseline'],
                    help='Scenario model to load and test')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()
MODEL_DIR = "ml_models"
OUTPUT_DIR = "final_results"  # Folder output baru

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nama model dan data tes yang akan di-load
model_load_name = f"dqn_agent_{symbol_lower}_{scenario}.zip"
test_data_load_name = f"test_data_{symbol_lower}_{scenario}.csv"
output_name = f"report_{symbol_lower}_{scenario}"

# --- FUNGSI METRIK ---


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


# ======================================================================
# === MAIN BACKTEST EXECUTION ===
# ======================================================================
print(f"\n{'='*60}")
print(
    f"🔬 MEMULAI BACKTEST: [{symbol_lower.upper()}] Skenario: [{scenario.upper()}]")
print(f"{'='*60}")

# --- Load Data Test ---
print(f"📂 Loading data: {test_data_load_name}...")
try:
    test_data_path = os.path.join(MODEL_DIR, test_data_load_name)
    df_test = pd.read_csv(
        test_data_path, index_col='timestamp', parse_dates=True)
except FileNotFoundError:
    print(
        f"❌ File {test_data_path} tidak ditemukan. Jalankan train_dqn.py dulu.")
    exit()

# --- Load Model ---
print(f"🤖 Loading model: {model_load_name}...")
try:
    model_path = os.path.join(MODEL_DIR, model_load_name)
    model_dqn = DQN.load(model_path)
except Exception as e:
    print(f"❌ Model tidak ditemukan: {e}. Jalankan train_dqn.py dulu.")
    exit()

# --- Siapkan Environment (HARUS SESUAI DENGAN TRAINING) ---
env_symbol_arg = symbol_lower
enable_net_arg = True
if scenario == 'baseline':
    enable_net_arg = False
elif scenario == 'default':
    env_symbol_arg = 'default'

print(
    f"Inisialisasi Env Test: symbol='{env_symbol_arg}', enable_safety_net={enable_net_arg}")
env = CryptoTradingEnv(
    df_test,
    symbol=env_symbol_arg,
    enable_safety_net=enable_net_arg,
    log_trades=True  # Selalu aktifkan logging saat tes
)

# --- Jalankan Simulasi ---
print("🚀 Memulai simulasi backtest...")
obs, _ = env.reset()
net_worth_history = [env.initial_balance]
timestamps = [df_test.index[0]]

for i in range(len(df_test) - 1):
    action_ai, _ = model_dqn.predict(obs, deterministic=True)

    # Serahkan aksi mentah ke Env V3 yang sudah pintar
    obs, reward, done, _, info = env.step(action_ai.item())

    net_worth_history.append(info['net_worth'])
    timestamps.append(df_test.index[i+1])

    if done:
        break

print("✅ Simulasi selesai.")

# --- Ambil Hasil dari Environment ---
trades = env.trade_history
safety_triggers = env.safety_net_triggers.copy()

# --- Hitung Metrik ---
bot_ret, bot_sharpe, bot_mdd = calculate_metrics(net_worth_history)

# HODL Benchmark
hodl_net_worth = (df_test['close'] /
                  df_test['close'].iloc[0]) * env.initial_balance
hodl_ret, hodl_sharpe, hodl_mdd = calculate_metrics(hodl_net_worth.values)

# Win Rate
win, loss, last_buy = 0, 0, 0
for t in trades:
    if t['action'] == 'BUY':
        last_buy = t['price']
    elif t['action'] == 'SELL' and last_buy > 0:
        if t['price'] > last_buy:
            win += 1
        else:
            loss += 1
total_closed = win + loss
win_rate = (win / total_closed * 100) if total_closed > 0 else 0

# --- Tampilkan Laporan Konsol ---
print(f"\n{'='*60}")
print(
    f"📊 LAPORAN BACKTEST: [{symbol_lower.upper()}] Skenario: [{scenario.upper()}]")
print(f"{'='*60}")
print(f"{'Metric':<20} | {'Final Bot':<12} | {'Benchmark (HODL)':<12}")
print(f"{'-'*60}")
print(f"{'Total Return':<20} | {bot_ret:>10.2f}%  | {hodl_ret:>10.2f}%")
print(f"{'Sharpe Ratio':<20} | {bot_sharpe:>10.4f}  | {hodl_sharpe:>10.4f}")
print(f"{'Max Drawdown':<20} | {bot_mdd:>10.2f}%  | {hodl_mdd:>10.2f}%")
print(f"{'-'*60}")
print(f"Trades Executed: {len(trades)} ({win}W / {loss}L)")
print(f"Win Rate: {win_rate:.1f}%")
print(f"Safety Net Blocks: {safety_triggers['total_blocks']}")
print(f"  ├─ Buy (Downtrend+OB): {safety_triggers['buy_blocked_downtrend']}")
print(f"  ├─ Buy (Volatile): {safety_triggers['buy_blocked_volatile']}")
print(f"  └─ Sell (Volatile): {safety_triggers['sell_blocked_volatile']}")
print(f"{'='*60}")

# --- Simpan Laporan Visual ---
plt.figure(figsize=(12, 8))
plt.suptitle(f"Backtest Report: {symbol_lower.upper()} (Scenario: {scenario.upper()})",
             fontsize=16, fontweight='bold')

# Plot 1: Equity Curve
plt.subplot(2, 1, 1)
plt.plot(timestamps, net_worth_history,
         label=f'Final Bot ({scenario})', color='blue', linewidth=2)
plt.plot(df_test.index, hodl_net_worth, label='Buy & Hold',
         color='gray', linestyle='--', alpha=0.6)
plt.title(f"Equity Curve (Return: {bot_ret:.2f}%)")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Trade Executions
plt.subplot(2, 1, 2)
plt.plot(df_test.index, df_test['close'],
         label=f'{symbol_lower.upper()} Price', color='black', alpha=0.5)

buys = [t for t in trades if t['action'] == 'BUY']
sells = [t for t in trades if t['action'] == 'SELL']

plt.scatter([t['time'] for t in buys], [t['price'] for t in buys],
            color='green', marker='^', s=50, label='BUY')
plt.scatter([t['time'] for t in sells], [t['price'] for t in sells],
            color='red', marker='v', s=50, label='SELL')

plt.title(f'Trading Executions ({len(trades)} trades)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle

output_png = os.path.join(OUTPUT_DIR, f'{output_name}.png')
plt.savefig(output_png, dpi=150)
print(f"💾 Chart saved to: {output_png}")
# plt.show() # Matikan show() agar script automation berjalan lancar
