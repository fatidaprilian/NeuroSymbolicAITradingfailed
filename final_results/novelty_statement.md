# Explicit Novelty Statement (3-Pillar Formulation)

Our proposed Neuro-Symbolic AI Cryptocurrency Trading Architecture introduces three distinct novel technical contributions:

1. **Dual-Model Hybrid Signal Fusion**: Unlike existing standalone DRL trading frameworks, our model integrates a non-linearly scalable Long Short-Term Memory (LSTM) network with transparent Linear Regression (LR) trend extrapolation, dynamically weighted via an adaptive validation error minimizer to provide robust price inputs to the state space.
2. **Deterministic Technical Safety Net Veto**: We engineer a deterministic symbolic veto layer operating directly over the RL agent's action space. By evaluating real-time Average True Range (ATR), Relative Strength Index (RSI), and Simple Moving Average (SMA) regime filters, the veto layer acts as an un-overrideable circuit-breaker against high-risk buy/sell signals in volatile or overbought markets.
3. **Auditable Explainable AI (XAI) Audit Logs with Statistical Rigor**: We bridge the explainability gap identified in quantitative finance literature by outputting real-time, human-readable symbolic veto logs for every overridden decision, combined with empirical statistical validation (Paired t-tests, Wilcoxon signed-rank tests, and Circular Block Bootstrap confidence intervals).
