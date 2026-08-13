# Neuro-Symbolic AI Cryptocurrency Trading Framework

A robust, risk-aware Neuro-Symbolic AI trading research framework for Bitcoin (BTC), Ethereum (ETH), and Ripple (XRP). It combines an adaptive Linear Regression-LSTM price forecasting ensemble, Deep Reinforcement Learning agents (DQN / QR-DQN), and a deterministic Rule-Based Symbolic Safety Net (ATR, RSI, SMA) acting as an un-overrideable veto circuit breaker against high-risk decisions.

---

## Clean Research Repository Structure

```
NeuroSymbolicAITrading/
├── data/                    # Processed CSV historical price datasets (BTC, ETH, XRP)
├── ml_models/               # Trained model checkpoints (.pkl, .zip)
├── final_results/           # Generated paper tables (Markdown & LaTeX) & charts
├── src/                     # Core Business Logic & Machine Learning Engine
│   ├── __init__.py
│   ├── features.py          # Technical indicator calculation engine (DRY)
│   ├── models.py            # Hybrid LR-LSTM & adaptive weighting engine
│   ├── stats_eval.py        # Inferential statistical tests (t-test, Wilcoxon, Bootstrap)
│   └── trading_env.py       # Neuro-Symbolic Gym environment & XAI logging
├── archive/                 # Legacy diagnostic scripts and backup archives
├── trading_env.py           # Top-level Gym environment wrapper
├── train_hybrid.py          # Hybrid forecasting training script
├── train_dqn.py             # DRL agent training script
├── train_all_agents.py      # DRL algorithm benchmark runner
├── run_test.py              # Main backtest runner & statistical suite
├── generate_paper_materials.py # Journal paper table & narrative generator
├── main.py                  # Single-command CLI launcher
└── README.md                # Documentation
```

---

## Quick Start & Commands

### 1. Run Complete End-to-End Pipeline
```bash
python3 main.py run-all
```

### 2. Train Hybrid Price Predictor (MAE, RMSE, MAPE, R2)
```bash
python3 main.py train-hybrid --symbol btc
```

### 3. Train Neuro-Symbolic DRL Agent
```bash
python3 main.py train-rl --symbol btc --scenario adaptive --timesteps 50000
```

### 4. Run Backtest & Inferential Statistical Significance Suite
```bash
python3 main.py backtest --symbol all
```

### 5. Generate LaTeX & Markdown Paper Tables
```bash
python3 main.py generate-paper
```

---

## Scientific Verification & Paper Outputs

All paper outputs are generated in `final_results/`:
- `table5_summary_results.md` & `table5_paper.tex`: Empirical strategy performance & statistical test tables (p < 0.05).
- `quantitative_abstract.md`: Quantitative Abstract in Indonesian & English.
- `related_work_table.md`: Comparative literature mapping (10 Q1/Q2 journal papers).
- `novelty_statement.md`: 3-Pillar Technical Novelty formulation.
- `eth_drawdown_discussion.md`: Drawdown & trade-frequency discussion under bear market regimes.
