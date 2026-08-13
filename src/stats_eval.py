"""
Statistical Evaluation Suite for Quantitative Trading Strategies.
Implements Paired t-tests, Wilcoxon Signed-Rank Tests, Circular Block Bootstrap,
and Risk-Adjusted Financial Performance Metrics (Sharpe, Sortino, MDD).
"""

import numpy as np
import pandas as pd
from scipy import stats


def calculate_portfolio_metrics(net_worth_series: np.ndarray, periods_per_year: int = 8760) -> dict:
    """
    Computes comprehensive financial trading metrics:
    - Cumulative Return (%)
    - Annualized Sharpe Ratio
    - Annualized Sortino Ratio
    - Maximum Drawdown (MDD %)
    - Win Rate (%)
    """
    nw = np.array(net_worth_series, dtype=np.float64)
    if len(nw) < 2:
        return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0}

    initial = nw[0]
    final = nw[-1]
    total_return = ((final - initial) / initial) * 100.0

    returns = np.diff(nw) / nw[:-1]
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)

    # Annualized Sharpe Ratio
    if std_ret > 1e-9:
        sharpe_ratio = (mean_ret / std_ret) * np.sqrt(periods_per_year)
    else:
        sharpe_ratio = 0.0

    # Downside Deviation & Sortino Ratio
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        std_downside = np.sqrt(np.mean(downside_returns**2))
        sortino_ratio = (mean_ret / std_downside) * np.sqrt(periods_per_year) if std_downside > 1e-9 else 0.0
    else:
        sortino_ratio = 0.0

    # Maximum Drawdown (MDD)
    cummax = np.maximum.accumulate(nw)
    drawdowns = (nw - cummax) / cummax
    max_drawdown = float(np.min(drawdowns)) * 100.0

    return {
        'total_return': float(total_return),
        'sharpe_ratio': float(sharpe_ratio),
        'sortino_ratio': float(sortino_ratio),
        'max_drawdown': float(max_drawdown)
    }


def paired_t_test(returns_strategy: np.ndarray, returns_benchmark: np.ndarray) -> dict:
    """
    Parametric Paired t-test on return differentials: d_t = r_strat,t - r_bench,t.
    H0: mean(d_t) == 0.
    """
    r_strat = np.array(returns_strategy, dtype=np.float64)
    r_bench = np.array(returns_benchmark, dtype=np.float64)

    min_len = min(len(r_strat), len(r_bench))
    diff = r_strat[:min_len] - r_bench[:min_len]

    t_stat, p_val = stats.ttest_rel(r_strat[:min_len], r_bench[:min_len])

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_val),
        'is_significant_5pct': bool(p_val < 0.05)
    }


def wilcoxon_signed_rank_test(returns_strategy: np.ndarray, returns_benchmark: np.ndarray) -> dict:
    """
    Non-parametric Wilcoxon Signed-Rank Test for non-normal return distributions.
    H0: median return differential is zero.
    """
    r_strat = np.array(returns_strategy, dtype=np.float64)
    r_bench = np.array(returns_benchmark, dtype=np.float64)

    min_len = min(len(r_strat), len(r_bench))
    diff = r_strat[:min_len] - r_bench[:min_len]

    # Filter out zero differences for Wilcoxon
    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) < 5:
        return {'w_statistic': 0.0, 'p_value': 1.0, 'is_significant_5pct': False}

    res = stats.wilcoxon(diff_nonzero)

    return {
        'w_statistic': float(res.statistic),
        'p_value': float(res.pvalue),
        'is_significant_5pct': bool(res.pvalue < 0.05)
    }


def circular_block_bootstrap(
    returns_strategy: np.ndarray,
    returns_benchmark: np.ndarray,
    n_bootstraps: int = 1000,
    block_size: int = 24,
    seed: int = 42
) -> dict:
    """
    Circular Block Bootstrap for dependent financial time series (Politis & Romano 1994).
    Resamples blocks of length block_size with wrap-around to preserve autocorrelation.
    Computes 95% Confidence Interval for mean return differential and empirical p-value.
    """
    np.random.seed(seed)
    r_strat = np.array(returns_strategy, dtype=np.float64)
    r_bench = np.array(returns_benchmark, dtype=np.float64)

    min_len = min(len(r_strat), len(r_bench))
    diff = r_strat[:min_len] - r_bench[:min_len]
    N = len(diff)

    if N < block_size:
        return {'ci_lower': 0.0, 'ci_upper': 0.0, 'p_value': 1.0, 'is_significant_5pct': False}

    # Extended array for circular wrapping
    extended_diff = np.concatenate([diff, diff[:block_size]])
    num_blocks = int(np.ceil(N / block_size))

    bootstrap_means = []
    for _ in range(n_bootstraps):
        start_indices = np.random.randint(0, N, size=num_blocks)
        sampled_diffs = []
        for idx in start_indices:
            sampled_diffs.extend(extended_diff[idx:idx + block_size])
        sampled_diffs = np.array(sampled_diffs[:N])
        bootstrap_means.append(np.mean(sampled_diffs))

    bootstrap_means = np.array(bootstrap_means)

    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))

    # Two-tailed empirical p-value (H0: mean diff <= 0)
    p_val = float(np.mean(bootstrap_means <= 0) * 2.0)
    p_val = min(p_val, 1.0)

    return {
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'p_value': p_val,
        'is_significant_5pct': bool(ci_lower > 0 or p_val < 0.05)
    }
