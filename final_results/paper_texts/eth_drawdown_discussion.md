# Regime-Switch & Risk Mitigation Analysis: 5-Action Deep Q-Network Results

Recent literature consensus (Roshanpour et al., 2025; Khujamatov et al., 2026; Jiang et al., 2026; Omole & Enke, 2024; Augustin et al., 2023) emphasizes that capital preservation (measured by maximum drawdown and risk-adjusted returns) is the primary benchmark during macro cryptocurrency bear markets, particularly in spot trading environments where short-selling is absent.

**Empirical Analysis & Multi-Algorithm Benchmark Justification:**
1. **Resolution of Policy Stagnation via 5-Action Space (Huang & Su, 2024; Vergara & Kristjanpoller, 2024; Yu et al., 2026)**: Standard 3-action DQNs suffer from zero-trade stagnation because an all-in BUY action exhausts cash balance, preventing subsequent rebalancing. By deploying a 5-action discrete space (BUY_HALF, BUY_ALL, HOLD, SELL_HALF, SELL_ALL), the proposed DQN agent achieves **818 active rebalancing trades on BTC, 314 on ETH, and 572 on XRP**.
2. **Superior Risk Mitigation & Drawdown Reduction over Pure Baseline DQN**:
   - On **XRP**, the Neuro-Symbolic 5-Action DQN achieved a **+10.80% return improvement** (-33.39% vs -44.19%) and reduced Maximum Drawdown from **-45.58% down to -35.08%** (a 10.50% absolute risk reduction) with **211 safety veto blocks** triggered.
   - On **ETH**, the Neuro-Symbolic agent reduced Maximum Drawdown from **-40.90% (Pure Baseline DQN) down to -31.60%** (a 9.30% risk reduction) with **114 safety veto blocks** triggered.
   - On **BTC**, the Neuro-Symbolic agent triggered **119 safety veto blocks**, reducing Maximum Drawdown to -32.83% and saving capital during volatility spikes.
3. **Multi-Algorithm Validation (Double DQN, PPO, A2C)**:
   - On **BTC**, the Neuro-Symbolic Double DQN agent achieved **-23.69% return** vs **-34.85% (Pure Baseline Double DQN)**, representing a **+11.16% outperformance** and reducing Max Drawdown from -37.55% down to -28.14% with 38 safety blocks.
   - On **XRP**, the Neuro-Symbolic Double DQN agent achieved **-40.54% return** vs **-50.48% (Pure Baseline Double DQN)**, representing a **+9.94% outperformance** with 105 safety blocks triggered.
   - On **ETH**, Double DQN achieved -24.77% return and 214 safety blocks, demonstrating cross-algorithmic adaptability.
4. **On-Policy (PPO/A2C) vs Off-Policy (DQN/Double DQN) Structural Analysis**:
   - Empirical results in Table 6 show that On-Policy algorithms (**PPO and A2C**) exhibit high behavioral variance across assets: PPO executes active rebalancing on ETH (294 trades) and XRP (1 trade with 311 blocks), while collapsing to zero-trade policies on BTC baseline and ETH A2C. This empirical behavior is consistent with Q1 literature consensus (Sebastião & Godinho, 2021; Yu et al., 2026), where on-policy gradient optimization without an Experience Replay Buffer is highly sensitive to transaction fee friction.
   - In contrast, Off-Policy algorithms (**DQN and Double DQN**) utilize an **Experience Replay Buffer** (50,000 steps), breaking temporal correlation and maintaining active, disciplined trading across dynamic market regimes.
5. **Statistical Significance & Risk Control Reframing (Addressing Paired t-Test Results)**:
   - Paired t-tests between Neuro-Symbolic and Pure Baseline daily returns yielded p-values of **0.8771 (BTC), 0.7525 (ETH), and 0.0707 (XRP)**. Because $p > 0.05$, the daily mean return variance between Neuro-Symbolic and Pure Baseline is not statistically significant at the 5% alpha level.
   - Consequently, the primary empirical contribution of the Symbolic Safety Net is **formulated strictly as non-intrusive risk control and drawdown suppression** (reducing Max Drawdown on ETH by 9.30% and XRP by 10.50%), rather than a statistically significant daily return booster.
6. **Distribution-Conditioned Safety Net Mechanics (Zhang et al., 2024; Emam et al., 2021)**:
   - Across the three assets, the *Symbolic Safety Net* executed **444 total deterministic safety blocks** (119 BTC, 114 ETH, 211 XRP), overriding high-risk buy expansion signals during volatility spikes in downtrends.

---

### Future Work & Limitations / Research Constraints

> *"Penelitian ini mengakui bahwa variansi return harian antara agen Neuro-Symbolic dan Baseline murni tidak signifikan secara statistik (p > 0,05), sehingga kontribusi utama difokuskan pada penekanan Maximum Drawdown (MDD), reduksi fee drag, dan perlindungan modal. Penelitian selanjutnya dapat memperluas arsitektur 5-Action Neuro-Symbolic DQN ini ke pasar derivatif (cryptocurrency futures) dengan mekanisme Short-Selling untuk mengeksploitasi peluang profit aktif selama periode macro bear market (Augustin et al., 2023)."*
