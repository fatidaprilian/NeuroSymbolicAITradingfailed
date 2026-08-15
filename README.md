# A Neuro-Symbolic AI Trading Architecture Combining Hybrid LR–LSTM Prediction, Deep Q-Network, and Symbolic Safety Nets

A robust, risk-aware Neuro-Symbolic AI cryptocurrency trading framework for Bitcoin (BTC), Ethereum (ETH), and Ripple (XRP). It integrates an adaptive Linear Regression-LSTM price forecasting ensemble, a **5-Action Discrete Space Deep Q-Network (DQN)** agent $\{0: \text{SELL 100\%}, 1: \text{SELL 50\%}, 2: \text{HOLD}, 3: \text{BUY 50\%}, 4: \text{BUY 100\%}\}$ (Huang & Su, 2024; Vergara & Kristjanpoller, 2024), and a deterministic **Rule-Based Symbolic Safety Net (ATR, RSI, SMA)** acting as an un-overrideable veto circuit breaker against high-risk buy expansions during volatile market stress (Kochliaridis et al., 2023; Jiang et al., 2026).

---

## Clean Research Repository Structure

```
NeuroSymbolicAITrading/
├── data/                    # Processed CSV historical price datasets (BTC, ETH, XRP)
├── ml_models/               # Trained model checkpoints (.zip)
├── final_results/           # Structured Output Artifacts
│   ├── charts/              # High-Res Charts (DPI 200) Organized per Asset
│   │   ├── btc/             # BTC Equity Curve & Execution Charts
│   │   ├── eth/             # ETH Equity Curve & Execution Charts
│   │   └── xrp/             # XRP Equity Curve & Execution Charts
│   ├── tables/              # Markdown & LaTeX Paper Tables
│   │   ├── table4_forecasting_metrics.md  # MAE, RMSE, MAPE, R² Forecasting Metrics
│   │   ├── table5_summary_results.md      # Primary 5-Action DQN Benchmark Table & t-tests
│   │   ├── table6_multi_algo_comparison.md# Comparative Multi-Algorithm Benchmark (DQN vs Double DQN vs HODL)
│   │   └── table5_paper.tex               # LaTeX Table for Journal Submission
│   └── paper_texts/         # Paper Draft Texts & Literature Mapping
│       ├── quantitative_abstract.md       # Quantitative Abstract (Indonesian & English)
│       ├── novelty_statement.md           # 3-Pillar Technical Novelty Statement
│       ├── eth_drawdown_discussion.md     # Asset Dynamics & Asset-Dependent Safety Net Discussion
│       ├── dqn_qrdqn_clarification.md     # Methodological Clarification on DQN & Scope
│       ├── hyperparameter_justification.md# Hyperparameter & Asset-Dependent Threshold Justification
│       └── related_work_table.md          # Comparative Mapping Table (2021-2026 Literature)
├── src/                     # Core Business Logic & Machine Learning Engine
│   ├── __init__.py
│   ├── features.py          # Technical indicator calculation engine
│   ├── models.py            # Hybrid LR-LSTM & adaptive weighting engine
│   ├── stats_eval.py        # Inferential statistical tests (Paired t-test, Wilcoxon, Bootstrap)
│   └── trading_env.py       # 5-Action CryptoTradingEnv5Action & XAI logging
├── trading_env.py           # Top-level Gym environment wrapper
├── train_hybrid.py          # Hybrid forecasting training script
├── train_dqn.py             # 5-Action DQN agent training script
├── train_doubledqn.py       # Double DQN agent training script
├── train_ppo_sac.py         # Baseline PPO & A2C agent training script
├── run_test.py              # Main backtest runner & statistical suite
├── generate_paper_materials.py # Journal paper table & narrative generator
├── main.py                  # Single-command CLI launcher
└── README.md                # Project Documentation
```

---

## Quick Start & Commands

### 1. Run Complete End-to-End Pipeline
```bash
python3 main.py run-all
```

### 2. Train Hybrid Price Predictor (MAE, RMSE, MAPE, R²)
```bash
python3 main.py train-hybrid --symbol btc
```

### 3. Train 5-Action Deep Q-Network (DQN) Agent
```bash
python3 main.py train-rl --symbol btc --scenario adaptive --timesteps 50000
```

### 4. Run Backtest & Inferential Statistical Significance Suite
```bash
python3 main.py backtest --symbol all
```

### 5. Generate LaTeX & Markdown Paper Materials
```bash
python3 generate_paper_materials.py
```

---

## Scientific Verification & Key Empirical Findings

1. **Resolution of Zero-Trade Policy Stagnation**:
   By introducing a 5-action discrete unit space (BUY_HALF, BUY_ALL, HOLD, SELL_HALF, SELL_ALL), the DQN agent achieves **830 active trades on BTC**, **324 on ETH**, and **614 on XRP**, resolving standard 3-action DQN stagnation under 0.1% transaction fee friction.

2. **Capital Preservation & Drawdown Mitigation (Jiang et al., 2026)**:
   - On **XRP**: Neuro-Symbolic 5-Action DQN outperformed Pure Baseline DQN by **+7.94% in cumulative return** (-36.25% vs -44.19%) with 8 deterministic safety blocks.
   - On **ETH**: Neuro-Symbolic 5-Action DQN reduced Maximum Drawdown from **-40.90% (Baseline DQN) to -32.73%** (an 8.17% risk reduction).

3. **Asset-Dependent Safety Net Validation (Vergara & Kristjanpoller, 2024; Zhang, 2025)**:
   The symbolic safety net acts as a non-intrusive circuit breaker that triggers selectively during high-volatility spikes (8 blocks on XRP) while remaining dormant on smoother large-cap trends (BTC and ETH).
