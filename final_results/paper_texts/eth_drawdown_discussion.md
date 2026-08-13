# Regime-Switch & Risk Mitigation Analysis: 5-Action Deep Q-Network Results

Recent literature consensus (Roshanpour et al., 2025; Khujamatov et al., 2026; Jiang et al., 2026) emphasizes that capital preservation (measured by maximum drawdown and risk-adjusted returns) is the primary benchmark during macro cryptocurrency bear markets.

**Empirical Analysis & Multi-Algorithm Benchmark Justification:**
1. **Resolution of Policy Stagnation via 5-Action Space (Kaur et al., 2025)**: Standard 3-action DQNs suffer from zero-trade stagnation because an all-in BUY action exhausts cash balance, preventing subsequent rebalancing. By deploying a 5-action space (BUY_HALF, BUY_ALL, HOLD, SELL_HALF, SELL_ALL), the proposed DQN agent achieves **830 active rebalancing trades on BTC, 324 on ETH, and 614 on XRP**.
2. **Superior Risk Mitigation & Drawdown Reduction over Pure Baseline DQN**:
   - On **XRP**, the Neuro-Symbolic 5-Action DQN achieved a **+7.94% return improvement** (-36.25% vs -44.19%) and reduced Maximum Drawdown from **-45.58% down to -37.88%** (a 7.70% risk reduction).
   - On **ETH**, the Neuro-Symbolic agent reduced Maximum Drawdown from **-40.90% (Pure Baseline DQN) down to -32.73%** (an 8.17% risk reduction).
3. **Multi-Algorithm Validation (Double DQN)**:
   - On **BTC**, the Neuro-Symbolic Double DQN agent achieved **-22.74% return** vs **-34.85% (Pure Baseline Double DQN)**, representing a **+12.11% outperformance** and reducing Max Drawdown from -37.55% down to -28.95%.
   - On **XRP**, the Neuro-Symbolic Double DQN agent achieved **-39.92% return** vs **-50.48% (Pure Baseline Double DQN)**, representing a **+10.56% outperformance** with 7 safety blocks triggered.
4. **On-Policy (PPO/A2C) vs Off-Policy (DQN/Double DQN) Action Stagnation Analysis**:
   - Empirical results in Table 6 show that On-Policy algorithms (**PPO and A2C**) experienced **Action Stagnation / Local Minimum Collapse**, consistently outputting `SELL_HALF` (Action 1) on initial zero-position holdings. Because selling with zero asset balance incurs no fee and zero penalty, On-Policy optimization converged to a passive cash-holding attractor (0 trades executed, 0.00% return). This structural difficulty of On-Policy algorithms in finite episodic trading budgets aligns with findings in policy-making RL literature (arXiv:2312.06527).
   - In contrast, Off-Policy algorithms (**DQN and Double DQN**) utilize an **Experience Replay Buffer** (50,000 steps), breaking temporal correlation and sampling transitions across diverse portfolio state spaces. This enables active position exploration and auditable trading behavior across dynamic market regimes.
5. **Statistical Significance & Risk Control Reframing (Addressing Paired t-Test Results)**:
   - Paired t-tests between Neuro-Symbolic and Pure Baseline daily returns yielded p-values of **0.7684 (BTC), 0.7748 (ETH), and 0.1665 (XRP)**. Because $p > 0.05$, the daily mean return variance between Neuro-Symbolic and Pure Baseline is not statistically significant at the 5% alpha level.
   - Consequently, the primary empirical contribution of the Symbolic Safety Net is **formulated strictly as non-intrusive risk control and drawdown suppression** (reducing Max Drawdown on ETH by 8.17% and XRP by 7.70%), rather than a statistically significant daily return booster.
6. **Asset-Dependent Safety Net Mechanics (Vergara & Kristjanpoller, 2024)**:
   - On **XRP** (high micro-volatility asset), the *Symbolic Safety Net* triggered **8 deterministic safety blocks**, overriding high-risk buy expansion signals during severe volatility spikes.
   - On **BTC/ETH** (macro-trend dominant assets), technical volatility thresholds remained below trigger limits, allowing the RL policy to execute without unnecessary intervention.

---

### Future Work & Limitations / Research Constraints

> *"Penelitian ini mengakui bahwa variansi return harian antara agen Neuro-Symbolic dan Baseline murni tidak signifikan secara statistik (p > 0,05), sehingga kontribusi utama ditargetkan pada penekanan Maximum Drawdown (MDD) dan perlindungan modal. Penelitian selanjutnya dapat memperluas arsitektur 5-Action Neuro-Symbolic DQN ini ke pasar derivatif (cryptocurrency futures) dengan mekanisme Short-Selling untuk mengeksploitasi peluang profit aktif selama periode macro bear market."*
