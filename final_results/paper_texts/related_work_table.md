# Comprehensive Related Work Comparison Table (2021-2026)

| Study / Reference | Market / Asset | Architecture & Model | Safety / Risk Mechanism | Statistical Validation | Key Findings & Empirical Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kabbani & Duman (2022)** | Bitcoin | DRL (PPO, A2C, DQN) | None (Pure RL) | None (Raw Return) | Notes extreme instability and policy collapse in pure RL under transaction fee friction. |
| **Kochliaridis et al. (2023)** | Crypto & Stocks | Genetic Fuzzy Veto + DRL | Fuzzy rule-based veto | Wilcoxon test | Proves deterministic safety net overrides dangerous RL trades during high market turbulence. |
| **Muminov et al. (2024)** | Top 10 Cryptos | Attention-BiLSTM + DQN | Stop-loss heuristics | Paired t-test | Shows price prediction integration enhances reward stability in value-based RL. |
| **Huang & Su (2024)** | Crypto Market | Multi-Discrete Unit DQN | Discrete position sizing | Return / Sharpe | Demonstrates multi-discrete action units mitigate policy stagnation under transaction fee friction. |
| **Vergara & Kristjanpoller (2024)** | Crypto Portfolio | Regime-Switching DRL | Volatility regime filter | Bootstrap CI | Demonstrates rule-guided position sizing prevents severe drawdowns across non-stationary regimes. |
| **Otabek & Choi (2024)** | Bitcoin | Multi-Level DQN (M-DQN) | Multi-objective reward shaping | Sharpe > 2.7 | Demonstrates DQN reward shaping mitigates over-trading; sensitive to fee hyper-parameters. |
| **Zhang (2025)** | Multi-Crypto | Hybrid Transformer-D3QN | Threshold circuit breakers | t-test / Sharpe | Validates that safety triggers preserve equity during flash crash market regimes. |
| **Priya et al. (2025)** | Crypto Trading | Hybrid Market-Aware DQN | Profit-driven selling | Empirical PnL | Proves hybrid buy-hold-sell architecture enhances risk-adjusted returns during volatile regimes. |
| **Khujamatov et al. (2026)** | Bitcoin | Risk-Aware DRL | Adaptive drawdown reward | Max Drawdown | Shows risk-adjusted rewards reduce drawdown to 16.8% during bearish test regimes. |
| **Jiang et al. (2026)** | Crypto Trading | Neuro-Symbolic DRL | Trend-analysis logic shield | Sharpe / MDD | Demonstrates logic-guided DRL outperforms black-box models during market crashes. |
| **Proposed Framework** | BTC, ETH, XRP | NeSy 5-Action DQN (LR-LSTM + 5-DQN + Adaptive Veto) | Distribution-Calibrated ATR/RSI/SMA Action Shield | Paired t-test, Wilcoxon, Bootstrap | Resolves zero-trade policy stagnation: 783 trades on BTC, +10.62% PnL on XRP, 9.01% MDD reduction on ETH, 451 active veto triggers across assets. |
