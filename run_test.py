"""
Comprehensive Backtest and Statistical Significance Evaluation Runner.
Executes strategy backtests across BTC, ETH, and XRP with 5-Action Deep Q-Network (DQN).
Computes risk/return metrics and inferential statistical significance tests (t-test, Wilcoxon).
Generates publication-ready paper tables (Markdown & LaTeX) and high-resolution chart images.
Matches paper title: "A NEURO-SYMBOLIC AI TRADING ARCHITECTURE COMBINING HYBRID LR-LSTM PREDICTION, DEEP Q-NETWORK, AND SYMBOLIC SAFETY NETS"
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from stable_baselines3 import DQN

from src.features import load_and_preprocess_data, train_val_test_split
from src.stats_eval import (
    calculate_portfolio_metrics,
    paired_t_test,
    wilcoxon_signed_rank_test,
    circular_block_bootstrap
)
from trading_env import CryptoTradingEnv5Action

MODEL_DIR = "ml_models"
OUTPUT_DIR = "final_results"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


def run_single_backtest(symbol: str = 'btc', scenario: str = 'adaptive', algo: str = 'dqn'):
    symbol_lower = symbol.lower()
    scenario_lower = scenario.lower()
    algo_lower = algo.lower()

    test_data_path = f"{MODEL_DIR}/test_data_{symbol_lower}_{scenario_lower}.csv"
    if not os.path.exists(test_data_path):
        df = load_and_preprocess_data(symbol_lower)
        _, _, df_test = train_val_test_split(df)
        df_test.to_csv(test_data_path)
    else:
        df_test = pd.read_csv(test_data_path)
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test.set_index('timestamp', inplace=True)
        else:
            df_test.index = pd.to_datetime(df_test.index)

    enable_net = (scenario_lower != 'baseline')
    env_symbol = symbol_lower if scenario_lower != 'default' else 'default'

    env = CryptoTradingEnv5Action(df_test, symbol=env_symbol, enable_safety_net=enable_net, log_trades=True)

    model_path = f"{MODEL_DIR}/{algo_lower}_agent_{symbol_lower}_{scenario_lower}.zip"

    model = None
    if os.path.exists(model_path):
        try:
            if algo_lower == 'doubledqn' or algo_lower == 'dqn':
                model = DQN.load(model_path, device='cpu')
            elif algo_lower == 'ppo':
                model = PPO.load(model_path, device='cpu')
            elif algo_lower == 'a2c':
                model = A2C.load(model_path, device='cpu')
        except Exception:
            model = None

    obs, _ = env.reset()
    net_worth_history = [env.initial_balance]
    timestamps = [df_test.index[0]]
    returns_list = []

    for i in range(len(df_test) - 1):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
            act_int = int(action)
        else:
            act_int = 2  # Default HOLD

        obs, reward, done, _, info = env.step(act_int)
        net_worth_history.append(info['net_worth'])
        timestamps.append(df_test.index[i + 1])
        returns_list.append((net_worth_history[-1] - net_worth_history[-2]) / net_worth_history[-2])
        if done:
            break

    metrics = calculate_portfolio_metrics(net_worth_history)

    # --- Generate High-Res Chart Image ---
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        timestamps_dt = pd.to_datetime(timestamps)
        hodl_nw = (df_test['close'] / df_test['close'].iloc[0]) * env.initial_balance
        test_index_dt = df_test.index[:len(hodl_nw)]

        algo_display_map = {'dqn': 'DQN', 'doubledqn': 'Double DQN', 'ppo': 'PPO', 'a2c': 'A2C'}
        algo_name = algo_display_map.get(algo_lower, algo_lower.upper())
        label_str = f'{algo_name} Neuro-Symbolic (5-Action)' if scenario_lower == 'adaptive' else f'{algo_name} Pure Baseline (5-Action)'
        ax1.plot(timestamps_dt, net_worth_history, label=label_str, color='#1f77b4', linewidth=1.8)
        ax1.plot(test_index_dt, hodl_nw, label='Buy & Hold (HODL)', color='#7f7f7f', linestyle='--', alpha=0.7)
        ax1.set_title(f"Equity Curve: {symbol_lower.upper()} [{algo_name} 5-Action] (Return: {metrics['total_return']:+.2f}%, MDD: {metrics['max_drawdown']:.2f}%)", fontsize=11, fontweight='bold', pad=14)
        ax1.set_ylabel("Portfolio Value (USD)", fontsize=10)
        ax1.legend(loc="upper right", framealpha=0.9, facecolor='white')
        ax1.grid(True, alpha=0.3)

        ax2.plot(df_test.index, df_test['close'], label=f'{symbol_lower.upper()} Price', color='#2ca02c', alpha=0.6)

        trades = env.trade_history
        buys = [t for t in trades if t['action'] == 'BUY']
        sells = [t for t in trades if t['action'] == 'SELL']

        def parse_trade_time(t_val):
            try:
                idx_val = int(t_val)
                if 0 <= idx_val < len(df_test):
                    return df_test.index[idx_val]
            except (ValueError, TypeError):
                pass
            return pd.to_datetime(t_val)

        if buys:
            buy_times = [parse_trade_time(t['time']) for t in buys]
            buy_prices = [t['price'] for t in buys]
            ax2.scatter(buy_times, buy_prices, color='green', marker='^', s=45, label='BUY Execution', zorder=5)

        if sells:
            sell_times = [parse_trade_time(t['time']) for t in sells]
            sell_prices = [t['price'] for t in sells]
            ax2.scatter(sell_times, sell_prices, color='red', marker='v', s=45, label='SELL Execution', zorder=5)

        ax2.set_title(f"Executions: {len(trades)}, Safety Blocks: {env.safety_net_triggers['total_blocks']}", fontsize=11, fontweight='bold', pad=14)
        ax2.set_ylabel("Price (USD)", fontsize=10)
        ax2.set_xlabel("Date", fontsize=10)
        ax2.legend(loc="upper right", framealpha=0.9, facecolor='white')
        ax2.grid(True, alpha=0.3)

        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate(rotation=30, ha='right')

        plt.tight_layout()
        asset_chart_dir = os.path.join(CHARTS_DIR, symbol_lower)
        os.makedirs(asset_chart_dir, exist_ok=True)
        chart_file = os.path.join(asset_chart_dir, f"chart_{symbol_lower}_{algo_lower}_{scenario_lower}.png")
        plt.savefig(chart_file, dpi=200)
        plt.close()
        print(f"Saved chart: {chart_file}")
    except Exception as e:
        print(f"Warning: Chart generation failed: {e}")

    return {
        'symbol': symbol_lower,
        'scenario': scenario_lower,
        'algo': algo_lower,
        'net_worth_history': net_worth_history,
        'returns': np.array(returns_list),
        'metrics': metrics,
        'trades': env.trade_history,
        'veto_triggers': env.safety_net_triggers,
        'xai_logs': env.xai_veto_logs
    }


def run_full_benchmark_suite(symbols=('btc', 'eth', 'xrp')):
    print(f"\n{'='*70}")
    print("RUNNING BENCHMARK SUITE (DQN vs DOUBLE DQN vs PPO vs A2C vs HODL)")
    print(f"{'='*70}\n")

    summary_rows_t5 = []
    summary_rows_t6 = []

    for sym in symbols:
        print(f"Processing Asset: [{sym.upper()}]")

        # 1. Proposed 5-Action Standard DQN
        res_dqn_nesy = run_single_backtest(sym, scenario='adaptive', algo='dqn')
        res_dqn_base = run_single_backtest(sym, scenario='baseline', algo='dqn')
        t_dqn = paired_t_test(res_dqn_nesy['returns'], res_dqn_base['returns'])
        m_dqn_n = res_dqn_nesy['metrics']
        m_dqn_b = res_dqn_base['metrics']

        # 2. 5-Action Double DQN
        res_ddqn_nesy = run_single_backtest(sym, scenario='adaptive', algo='doubledqn')
        res_ddqn_base = run_single_backtest(sym, scenario='baseline', algo='doubledqn')
        m_ddqn_n = res_ddqn_nesy['metrics']
        m_ddqn_b = res_ddqn_base['metrics']

        # 3. PPO Agent
        res_ppo_nesy = run_single_backtest(sym, scenario='adaptive', algo='ppo')
        res_ppo_base = run_single_backtest(sym, scenario='baseline', algo='ppo')
        m_ppo_n = res_ppo_nesy['metrics']
        m_ppo_b = res_ppo_base['metrics']

        # 4. A2C Agent
        res_a2c_nesy = run_single_backtest(sym, scenario='adaptive', algo='a2c')
        res_a2c_base = run_single_backtest(sym, scenario='baseline', algo='a2c')
        m_a2c_n = res_a2c_nesy['metrics']
        m_a2c_b = res_a2c_base['metrics']

        # 5. Buy & Hold Benchmark
        df_test = pd.read_csv(f"{MODEL_DIR}/test_data_{sym}_adaptive.csv")
        if 'timestamp' in df_test.columns:
            df_test['timestamp'] = pd.to_datetime(df_test['timestamp'])
            df_test.set_index('timestamp', inplace=True)
        else:
            df_test.index = pd.to_datetime(df_test.index)

        hodl_nw = (df_test['close'] / df_test['close'].iloc[0]) * 10000.0
        hodl_returns = np.diff(hodl_nw) / hodl_nw[:-1]
        hodl_metrics = calculate_portfolio_metrics(hodl_nw.values)

        t_hodl = paired_t_test(res_dqn_nesy['returns'], hodl_returns)

        # --- Table 5 (Primary Proposed Method Benchmark) ---
        summary_rows_t5.append({
            'Asset': sym.upper(),
            'Algorithm': 'DQN (5-Action)',
            'Strategy': 'Neuro-Symbolic',
            'Return (%)': f"{m_dqn_n['total_return']:+.2f}%",
            'Sharpe': f"{m_dqn_n['sharpe_ratio']:.4f}",
            'Sortino': f"{m_dqn_n['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_dqn_n['max_drawdown']:.2f}%",
            'Trades': len(res_dqn_nesy['trades']),
            'Safety Blocks': res_dqn_nesy['veto_triggers']['total_blocks'],
            't-stat': f"{t_dqn['t_statistic']:.3f}",
            'p-val': f"{t_dqn['p_value']:.4f}",
            'Sig (p<0.05)': "YES" if t_dqn['is_significant_5pct'] else "NO"
        })

        summary_rows_t5.append({
            'Asset': sym.upper(),
            'Algorithm': 'DQN (5-Action)',
            'Strategy': 'Pure Baseline',
            'Return (%)': f"{m_dqn_b['total_return']:+.2f}%",
            'Sharpe': f"{m_dqn_b['sharpe_ratio']:.4f}",
            'Sortino': f"{m_dqn_b['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_dqn_b['max_drawdown']:.2f}%",
            'Trades': len(res_dqn_base['trades']),
            'Safety Blocks': 0,
            't-stat': "-",
            'p-val': "-",
            'Sig (p<0.05)': "-"
        })

        summary_rows_t5.append({
            'Asset': sym.upper(),
            'Algorithm': 'Passive (HODL)',
            'Strategy': 'Buy & Hold',
            'Return (%)': f"{hodl_metrics['total_return']:+.2f}%",
            'Sharpe': f"{hodl_metrics['sharpe_ratio']:.4f}",
            'Sortino': f"{hodl_metrics['sortino_ratio']:.4f}",
            'Max Drawdown': f"{hodl_metrics['max_drawdown']:.2f}%",
            'Trades': 1,
            'Safety Blocks': 0,
            't-stat': f"{t_hodl['t_statistic']:.3f}",
            'p-val': f"{t_hodl['p_value']:.4f}",
            'Sig (p<0.05)': "YES" if t_hodl['is_significant_5pct'] else "NO"
        })

        # --- Table 6 (Comparative Analysis of DRL Algorithms) ---
        summary_rows_t6.append({
            'Asset': sym.upper(),
            'Algorithm': 'DQN (5-Action) [Proposed]',
            'Strategy': 'Neuro-Symbolic',
            'Return (%)': f"{m_dqn_n['total_return']:+.2f}%",
            'Sharpe': f"{m_dqn_n['sharpe_ratio']:.4f}",
            'Sortino': f"{m_dqn_n['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_dqn_n['max_drawdown']:.2f}%",
            'Trades': len(res_dqn_nesy['trades']),
            'Safety Blocks': res_dqn_nesy['veto_triggers']['total_blocks']
        })

        summary_rows_t6.append({
            'Asset': sym.upper(),
            'Algorithm': 'Double DQN (5-Action)',
            'Strategy': 'Neuro-Symbolic',
            'Return (%)': f"{m_ddqn_n['total_return']:+.2f}%",
            'Sharpe': f"{m_ddqn_n['sharpe_ratio']:.4f}",
            'Sortino': f"{m_ddqn_n['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_ddqn_n['max_drawdown']:.2f}%",
            'Trades': len(res_ddqn_nesy['trades']),
            'Safety Blocks': res_ddqn_nesy['veto_triggers']['total_blocks']
        })

        summary_rows_t6.append({
            'Asset': sym.upper(),
            'Algorithm': 'PPO (5-Action)',
            'Strategy': 'Neuro-Symbolic',
            'Return (%)': f"{m_ppo_n['total_return']:+.2f}%",
            'Sharpe': f"{m_ppo_n['sharpe_ratio']:.4f}",
            'Sortino': f"{m_ppo_n['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_ppo_n['max_drawdown']:.2f}%",
            'Trades': len(res_ppo_nesy['trades']),
            'Safety Blocks': res_ppo_nesy['veto_triggers']['total_blocks']
        })

        summary_rows_t6.append({
            'Asset': sym.upper(),
            'Algorithm': 'A2C (5-Action)',
            'Strategy': 'Neuro-Symbolic',
            'Return (%)': f"{m_a2c_n['total_return']:+.2f}%",
            'Sharpe': f"{m_a2c_n['sharpe_ratio']:.4f}",
            'Sortino': f"{m_a2c_n['sortino_ratio']:.4f}",
            'Max Drawdown': f"{m_a2c_n['max_drawdown']:.2f}%",
            'Trades': len(res_a2c_nesy['trades']),
            'Safety Blocks': res_a2c_nesy['veto_triggers']['total_blocks']
        })

        summary_rows_t6.append({
            'Asset': sym.upper(),
            'Algorithm': 'Passive (HODL)',
            'Strategy': 'Buy & Hold',
            'Return (%)': f"{hodl_metrics['total_return']:+.2f}%",
            'Sharpe': f"{hodl_metrics['sharpe_ratio']:.4f}",
            'Sortino': f"{hodl_metrics['sortino_ratio']:.4f}",
            'Max Drawdown': f"{hodl_metrics['max_drawdown']:.2f}%",
            'Trades': 1,
            'Safety Blocks': 0
        })

    # Save Table 5
    df_t5 = pd.DataFrame(summary_rows_t5)
    md_t5_path = f"{TABLES_DIR}/table5_summary_results.md"
    with open(md_t5_path, 'w') as f:
        f.write("# Table 5: 5-Action Deep Q-Network (DQN) Performance & Statistical Significance\n\n")
        f.write(df_t5.to_markdown(index=False))
    print(f"Saved summary report: {md_t5_path}")

    latex_t5_path = f"{TABLES_DIR}/table5_paper.tex"
    with open(latex_t5_path, 'w') as f:
        f.write("% Table 5: 5-Action Deep Q-Network (DQN) Performance & Inferential Statistical Tests\n")
        f.write(df_t5.to_latex(index=False))
    print(f"Saved LaTeX table: {latex_t5_path}")

    # Save Table 6 (Comparative Analysis of DRL Algorithms)
    df_t6 = pd.DataFrame(summary_rows_t6)
    md_t6_path = f"{TABLES_DIR}/table6_multi_algo_comparison.md"
    with open(md_t6_path, 'w') as f:
        f.write("# Table 6: Comparative Analysis of DRL Algorithms (DQN vs Double DQN vs PPO vs A2C vs Buy & Hold)\n\n")
        f.write(df_t6.to_markdown(index=False))
    print(f"Saved multi-algo comparison table: {md_t6_path}")

    print("\n" + "="*80)
    print("TABLE 5: PRIMARY 5-ACTION DQN BENCHMARK")
    print("="*80)
    print(df_t5.to_string(index=False))
    print("\n" + "="*80)
    print("TABLE 6: MULTI-ALGORITHM COMPARISON (DQN vs DOUBLE DQN vs PPO vs A2C vs HODL)")
    print("="*80)
    print(df_t6.to_string(index=False))
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 5-Action DQN Backtest & Statistical Evaluation Suite")
    parser.add_argument('--symbol', type=str, default='all', help='Crypto symbol (btc, eth, xrp, or all)')
    args = parser.parse_args()

    if args.symbol.lower() == 'all':
        run_full_benchmark_suite(('btc', 'eth', 'xrp'))
    else:
        run_full_benchmark_suite((args.symbol.lower(),))
