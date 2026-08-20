"""
SMA Crossover (Golden Cross / Death Cross) Rule-Based Baseline Strategy.
Evaluates deterministic moving average crossover (SMA_7 vs SMA_30) on BTC, ETH, and XRP
using identical test split data and 0.1% transaction fee structure.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from src.stats_eval import calculate_portfolio_metrics

MODEL_DIR = "ml_models"
CHARTS_DIR = "final_results/charts"
TABLES_DIR = "final_results/tables"

os.makedirs(TABLES_DIR, exist_ok=True)


def run_sma_crossover(symbol: str = 'btc', initial_balance: float = 10000.0, fee: float = 0.001):
    symbol_lower = symbol.lower()
    data_path = f"{MODEL_DIR}/test_data_{symbol_lower}_baseline.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test data not found at {data_path}")

    df_test = pd.read_csv(data_path)
    if 'timestamp' in df_test.columns:
        df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
        df_test.set_index('timestamp', inplace=True)
    else:
        df_test.index = pd.to_datetime(df_test.index)

    # Ensure SMA_7 and SMA_30 exist
    if 'SMA_7' not in df_test.columns:
        df_test['SMA_7'] = df_test['close'].rolling(window=7).mean()
    if 'SMA_30' not in df_test.columns:
        df_test['SMA_30'] = df_test['close'].rolling(window=30).mean()

    df_test = df_test.ffill().bfill()

    balance_usdt = initial_balance
    balance_crypto = 0.0
    net_worth_history = [initial_balance]
    timestamps = [df_test.index[0]]
    trade_history = []

    sma7_vals = df_test['SMA_7'].values
    sma30_vals = df_test['SMA_30'].values
    close_vals = df_test['close'].values
    idx_vals = df_test.index

    for i in range(1, len(df_test)):
        curr_price = close_vals[i]
        curr_time = idx_vals[i]

        prev_sma7 = sma7_vals[i - 1]
        prev_sma30 = sma30_vals[i - 1]
        curr_sma7 = sma7_vals[i]
        curr_sma30 = sma30_vals[i]

        # Golden Cross: SMA_7 crosses above SMA_30 -> BUY ALL
        is_golden_cross = (prev_sma7 <= prev_sma30) and (curr_sma7 > curr_sma30)
        # Death Cross: SMA_7 crosses below SMA_30 -> SELL ALL
        is_death_cross = (prev_sma7 >= prev_sma30) and (curr_sma7 < curr_sma30)

        if is_golden_cross and balance_usdt > 0.0:
            fee_cost = balance_usdt * fee
            usable_usdt = balance_usdt - fee_cost
            crypto_bought = usable_usdt / curr_price
            balance_crypto += crypto_bought
            balance_usdt = 0.0
            trade_history.append({
                'time': curr_time,
                'action': 'BUY',
                'price': curr_price,
                'amount': crypto_bought,
                'fee': fee_cost
            })

        elif is_death_cross and balance_crypto > 1e-6:
            gross_usdt = balance_crypto * curr_price
            fee_cost = gross_usdt * fee
            net_usdt = gross_usdt - fee_cost
            balance_usdt += net_usdt
            sold_crypto = balance_crypto
            balance_crypto = 0.0
            trade_history.append({
                'time': curr_time,
                'action': 'SELL',
                'price': curr_price,
                'amount': sold_crypto,
                'fee': fee_cost
            })

        current_net_worth = balance_usdt + (balance_crypto * curr_price)
        net_worth_history.append(current_net_worth)
        timestamps.append(curr_time)

    metrics = calculate_portfolio_metrics(net_worth_history)

    # --- Generate Chart (DPI=200, matching exact visual style) ---
    try:
        asset_chart_dir = os.path.join(CHARTS_DIR, symbol_lower)
        os.makedirs(asset_chart_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        timestamps_dt = pd.to_datetime(timestamps)
        hodl_nw = (df_test['close'] / df_test['close'].iloc[0]) * initial_balance
        test_index_dt = df_test.index[:len(hodl_nw)]

        ax1.plot(timestamps_dt, net_worth_history, label='SMA Crossover (Rule-Based)', color='#1f77b4', linewidth=1.8)
        ax1.plot(test_index_dt, hodl_nw, label='Buy & Hold (HODL)', color='#7f7f7f', linestyle='--', alpha=0.7)
        ax1.set_title(
            f"Equity Curve: {symbol_lower.upper()} [SMA Crossover] (Return: {metrics['total_return']:+.2f}%, MDD: {metrics['max_drawdown']:.2f}%)",
            fontsize=11, fontweight='bold', pad=14
        )
        ax1.set_ylabel("Portfolio Value (USD)", fontsize=10)
        ax1.legend(loc="upper right", framealpha=0.9, facecolor='white')
        ax1.grid(True, alpha=0.3)

        ax2.plot(df_test.index, df_test['close'], label=f'{symbol_lower.upper()} Price', color='#2ca02c', alpha=0.6)

        buys = [t for t in trade_history if t['action'] == 'BUY']
        sells = [t for t in trade_history if t['action'] == 'SELL']

        if buys:
            buy_times = [t['time'] for t in buys]
            buy_prices = [t['price'] for t in buys]
            ax2.scatter(buy_times, buy_prices, color='green', marker='^', s=45, label='BUY (Golden Cross)', zorder=5)

        if sells:
            sell_times = [t['time'] for t in sells]
            sell_prices = [t['price'] for t in sells]
            ax2.scatter(sell_times, sell_prices, color='red', marker='v', s=45, label='SELL (Death Cross)', zorder=5)

        ax2.set_title(f"Executions: {len(trade_history)}, Safety Blocks: 0", fontsize=11, fontweight='bold', pad=14)
        ax2.set_ylabel("Price (USD)", fontsize=10)
        ax2.set_xlabel("Date", fontsize=10)
        ax2.legend(loc="upper right", framealpha=0.9, facecolor='white')
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=30, ha='right')

        plt.tight_layout()
        chart_file = os.path.join(asset_chart_dir, f"chart_{symbol_lower}_smacrossover.png")
        plt.savefig(chart_file, dpi=200)
        plt.close()
        print(f"Saved chart: {chart_file}")
    except Exception as e:
        print(f"Warning: Chart generation failed: {e}")

    return {
        'symbol': symbol_lower.upper(),
        'metrics': metrics,
        'trades': len(trade_history),
        'net_worth_history': net_worth_history
    }


def run_all_sma_baselines(symbols=('btc', 'eth', 'xrp')):
    print(f"\n{'='*70}")
    print("RUNNING SMA CROSSOVER BASELINE EVALUATION (BTC, ETH, XRP)")
    print(f"{'='*70}\n")

    sma_results = {}
    for sym in symbols:
        res = run_sma_crossover(sym)
        sma_results[sym.upper()] = res

    # --- Load Existing Table 5 Benchmarks for Side-by-Side Comparison ---
    table5_benchmarks = {
        'BTC': [
            {'Algorithm': 'DQN (5-Action)', 'Strategy': 'Neuro-Symbolic', 'Return (%)': '-27.60%', 'Sharpe': '-2.9238', 'Sortino': '-2.8988', 'Max Drawdown': '-32.86%', 'Trades': 783, 'Safety Blocks': 125},
            {'Algorithm': 'DQN (5-Action)', 'Strategy': 'Pure Baseline', 'Return (%)': '-26.24%', 'Sharpe': '-2.6459', 'Sortino': '-2.6259', 'Max Drawdown': '-30.52%', 'Trades': 561, 'Safety Blocks': 0},
            {'Algorithm': 'Passive (HODL)', 'Strategy': 'Buy & Hold', 'Return (%)': '-17.85%', 'Sharpe': '-1.5215', 'Sortino': '-1.4977', 'Max Drawdown': '-29.36%', 'Trades': 1, 'Safety Blocks': 0},
        ],
        'ETH': [
            {'Algorithm': 'DQN (5-Action)', 'Strategy': 'Neuro-Symbolic', 'Return (%)': '-18.60%', 'Sharpe': '-1.3934', 'Sortino': '-1.3094', 'Max Drawdown': '-31.89%', 'Trades': 307, 'Safety Blocks': 113},
            {'Algorithm': 'DQN (5-Action)', 'Strategy': 'Pure Baseline', 'Return (%)': '-22.16%', 'Sharpe': '-1.5825', 'Sortino': '-1.5348', 'Max Drawdown': '-40.90%', 'Trades': 156, 'Safety Blocks': 0},
            {'Algorithm': 'Passive (HODL)', 'Strategy': 'Buy & Hold', 'Return (%)': '-18.40%', 'Sharpe': '-1.0953', 'Sortino': '-1.0753', 'Max Drawdown': '-36.90%', 'Trades': 1, 'Safety Blocks': 0},
        ],
        'XRP': [
            {'Algorithm': 'DQN (5-Action)', 'Strategy': 'Neuro-Symbolic', 'Return (%)': '-33.57%', 'Sharpe': '-3.0803', 'Sortino': '-3.0039', 'Max Drawdown': '-35.09%', 'Trades': 574, 'Safety Blocks': 213},
            {'Algorithm': 'DQN (5-Action)', 'Strategy': 'Pure Baseline', 'Return (%)': '-44.19%', 'Sharpe': '-4.0686', 'Sortino': '-3.9791', 'Max Drawdown': '-45.58%', 'Trades': 327, 'Safety Blocks': 0},
            {'Algorithm': 'Passive (HODL)', 'Strategy': 'Buy & Hold', 'Return (%)': '-28.93%', 'Sharpe': '-2.0982', 'Sortino': '-2.0688', 'Max Drawdown': '-35.26%', 'Trades': 1, 'Safety Blocks': 0},
        ]
    }

    # --- Build Table 7 Data Rows ---
    table7_rows = []
    for sym in ('BTC', 'ETH', 'XRP'):
        m_sma = sma_results[sym]['metrics']
        trades_sma = sma_results[sym]['trades']

        # 1. Proposed Neuro-Symbolic DQN
        row_nesy = {'Asset': sym}
        row_nesy.update(table5_benchmarks[sym][0])
        table7_rows.append(row_nesy)

        # 2. Pure Baseline DQN
        row_base = {'Asset': sym}
        row_base.update(table5_benchmarks[sym][1])
        table7_rows.append(row_base)

        # 3. SMA Crossover Baseline
        table7_rows.append({
            'Asset': sym,
            'Algorithm': 'SMA Crossover (7/30)',
            'Strategy': 'Rule-Based',
            'Return (%)': f"{m_sma['total_return']:+.2f}%",
            'Sharpe': f"{m_sma['sharpe_ratio']:.4f}",
            'Sortino': f"{m_sma['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_sma['max_drawdown']:.2f}%",
            'Trades': trades_sma,
            'Safety Blocks': 0
        })

        # 4. Passive HODL
        row_hodl = {'Asset': sym}
        row_hodl.update(table5_benchmarks[sym][2])
        table7_rows.append(row_hodl)

    df_t7 = pd.DataFrame(table7_rows)
    # Ensure exact column order
    cols_order = ['Asset', 'Algorithm', 'Strategy', 'Return (%)', 'Sharpe', 'Sortino', 'Max Drawdown', 'Trades', 'Safety Blocks']
    df_t7 = df_t7[cols_order]

    # --- Save Table 7 Markdown ---
    t7_path = os.path.join(TABLES_DIR, "table7_sma_crossover.md")
    with open(t7_path, 'w') as f:
        f.write("# Table 7: Performance Comparison with Rule-Based Strategy (SMA Crossover)\n\n")
        f.write(df_t7.to_markdown(index=False))
        f.write("\n")
    print(f"Saved Table 7: {t7_path}")

    # --- Print Terminal Summary ---
    print("\n" + "="*80)
    print("SUMMARY RESULTS: SMA CROSSOVER (7/30) RULE-BASED BASELINE")
    print("="*80)
    print(f"{'Asset':<6} | {'Strategy':<20} | {'Return (%)':<12} | {'Sharpe':<8} | {'Sortino':<8} | {'Max Drawdown':<14} | {'Trades':<6}")
    print("-" * 80)
    for sym in ('BTC', 'ETH', 'XRP'):
        m = sma_results[sym]['metrics']
        t = sma_results[sym]['trades']
        print(f"{sym:<6} | {'SMA Crossover (7/30)':<20} | {m['total_return']:>+10.2f}% | {m['sharpe_ratio']:>8.4f} | {m['sortino_ratio']:>8.4f} | {m['max_drawdown']:>12.2f}% | {t:>6}")
    print("="*80 + "\n")

    print("\n" + "="*80)
    print("TABLE 7: COMPLETE COMPARISON WITH DQN & PASSIVE HODL")
    print("="*80)
    print(df_t7.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    run_all_sma_baselines()
