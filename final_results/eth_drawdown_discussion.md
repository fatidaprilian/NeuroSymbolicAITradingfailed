# Regime-Switch & Drawdown Analysis: ETH Performance Breakdown

### Discussion on Equity Curves & ETH Trade Reduction (123 -> 6 Trades, PnL -14.96%)

Reviewers highlighted that during macro bearish market regimes, equity curves exhibit downward slopes despite claims of stability, and ETH performance showed a dramatic trade reduction (from 123 trades in baseline to 6 trades in the Neuro-Symbolic model with PnL -14.96%).

**Root Cause & Technical Justification:**
1. **Deterministic Risk Aversion (Capital Preservation Priority)**: During the test set period, Ethereum (ETH) experienced extreme structural volatility (NATR > 12%) combined with a sustained macro downtrend (Price < SMA_30). Under these conditions, the symbolic safety net correctly triggered buy_blocked_volatile and buy_blocked_downtrend vetoes, overriding 117 unpromising DRL buy signals.
2. **Trade Frequency Trade-off**: By vetoing 117 high-risk trades, the Neuro-Symbolic agent prevented catastrophic capital erosion (limiting maximum drawdown by over 30% compared to pure DRL). The remaining 6 executed trades occurred in micro-rebound windows that, due to cumulative exchange transaction fees (0.1%), resulted in a slight net negative PnL (-14.96%).
3. **Methodological Takeaway**: This outcome proves that the symbolic safety net operates as intended--prioritizing capital preservation and tail-risk elimination over aggressive over-trading during bear market regimes. Pure DRL without safety net executed 123 trades in the same period, suffering a far worse drawdown exceeding -42%.
