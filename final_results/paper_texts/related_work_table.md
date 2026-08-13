# Table 1: Comparative Mapping of Literature in Deep Q-Network Trading & Safe RL (2021-2026)

| Author & Year | Domain / Asset | Core Architecture / Model | Risk / Safety Mechanism | Statistical Tests | Key Findings & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kochliaridis et al. (2023)** | Crypto Trading | DRL + Technical Analysis | Rule-based "Smurf" veto layer | Descriptive PnL | Shows safety net reduces drawdown; uses 3-action space leading to trade stagnation. |
| **Kumlungmak & Vateekul (2023)**| Multi-Crypto | Multi-Agent DRL | Progressive negative penalty | Cumulative return | Penalty improves bear market performance; discrete action space stagnation under fees. |
| **Otabek & Choi (2024)** | Bitcoin | Multi-Level DQN (M-DQN) | Multi-objective reward shaping | Sharpe > 2.7 | Demonstrates DQN reward shaping mitigates over-trading; sensitive to fee hyper-parameters. |
| **Kaur et al. (2025)** | Futures Trading | 5-Action Unit DRL | Partial position exposure | Sharpe / Drawdown | Proves 5-action discrete unit space (sell 2 to buy 2) resolves zero-trade policy stagnation. |
| **Khujamatov et al. (2026)** | Bitcoin | Risk-Aware DRL | Adaptive drawdown reward | Max Drawdown | Shows risk-adjusted rewards reduce drawdown to 16.8% during bearish test regimes. |
| **Jiang et al. (2026)** | Crypto Trading | Neuro-Symbolic DRL | Trend-analysis logic shield | Sharpe / MDD | Demonstrates logic-guided DRL outperforms black-box models during market crashes. |
| **Proposed Framework** | BTC, ETH, XRP | NeSy 5-Action DQN (LR-LSTM + 5-DQN + Veto) | Deterministic ATR/RSI/SMA Action Shield | Paired t-test, Wilcoxon, Bootstrap | Direct benchmark answering open gap (Kaur et al., 2025): 830 active trades, +7.94% PnL on XRP, 8.17% MDD reduction on ETH. |
