# Table 1: Comparative Mapping of Literature in AI Trading & Safe RL (2021-2026)

| Author & Year | Domain / Asset | Core Architecture / Model | Risk / Safety Mechanism | Statistical Tests | Key Findings & Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hitzler et al. (2022)** | General AI | NeSy Taxonomy (Neural + Symbolic) | Knowledge-driven constraints | N/A | Establishes NeSy paradigm: neural for data synthesis, symbolic for deterministic explainability. |
| **Wang et al. (2022)** | Cognitive Systems | Neuro-Symbolic MDPs | Hard logical constraints | N/A | Formulates relational MDPs with logical shielding; highlights scalability challenges. |
| **Kochliaridis et al. (2023)** | Crypto Trading | DRL + Technical Analysis | Rule-based "Smurf" veto layer | Descriptive PnL | Shows safety net reduces catastrophic drawdown; lacks formal statistical significance tests. |
| **Kumlungmak & Vateekul (2023)** | BTC / Altcoins | Multi-Agent MAPPO | Progressive negative penalty | Cumulative return | Penalty improves bear market performance; discrete action space limitation. |
| **Escudero et al. (2024)** | Portfolio Mgmt | DRL (PPO) + Post-hoc XAI | Feature importance / SHAP / LIME | Portfolio metrics | Provides post-hoc explainability for DRL allocations; non-deterministic execution safety. |
| **Otabek & Choi (2024)** | Bitcoin | Multi-Level DQN (M-DQN) | Multi-objective reward shaping | Sharpe > 2.7 | New reward function mitigates over-trading; sensitive to hyperparameter tuning. |
| **Qureshi et al. (2025)** | Crypto Price | ML / Deep Learning Regressors | Baseline statistical bounds | Paired t-test, Wilcoxon | Demonstrates statistical error metrics do not automatically translate to trading profitability. |
| **Bysik & Ślepaczuk (2026)** | Bitcoin | XGBoost / LSTM / iTransformer | Walk-forward cost filter | Circular block bootstrap | Confirms naive DRL fails under 0.1% transaction fee without deterministic execution bounds. |
| **Vasileva et al. (2026)** | 14 Crypto Assets | Walk-Forward Regressors | Nested cross-validation | Diebold-Mariano test | Shows predictive ranking MAE/RMSE decouples from post-fee Sharpe ratio across assets. |
| **Proposed Framework** | BTC, ETH, XRP | NeSy Hybrid (LR-LSTM + DRL + Veto) | Deterministic ATR/RSI/SMA Veto | Paired t-test, Wilcoxon, Bootstrap | Achieves high risk-adjusted return, statistically significant outperformance, and full XAI auditability. |
