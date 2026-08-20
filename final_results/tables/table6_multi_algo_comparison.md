# Table 6: Comparative Analysis of DRL Algorithms (DQN vs Double DQN vs PPO vs A2C vs Buy & Hold)

| Asset   | Algorithm                 | Strategy       | Return (%)   |   Sharpe |   Sortino | Max Drawdown   |   Trades |   Safety Blocks |
|:--------|:--------------------------|:---------------|:-------------|---------:|----------:|:---------------|---------:|----------------:|
| BTC     | DQN (5-Action) [Proposed] | Neuro-Symbolic | -27.06%      |  -2.9127 |   -2.8651 | -32.83%        |      818 |             119 |
| BTC     | Double DQN (5-Action)     | Neuro-Symbolic | -23.69%      |  -2.5183 |   -2.5038 | -28.14%        |      452 |              38 |
| BTC     | Double DQN (5-Action)     | Pure Baseline  | -34.85%      |  -4.2387 |   -4.003  | -37.55%        |      706 |               0 |
| BTC     | PPO (5-Action)            | Neuro-Symbolic | -17.14%      |  -1.4609 |   -1.4298 | -29.34%        |       10 |              41 |
| BTC     | PPO (5-Action)            | Pure Baseline  | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| BTC     | A2C (5-Action)            | Neuro-Symbolic | -12.37%      |  -1.5557 |   -1.5174 | -20.52%        |        2 |               0 |
| BTC     | A2C (5-Action)            | Pure Baseline  | -20.33%      |  -2.5192 |   -2.1184 | -24.07%        |      176 |               0 |
| BTC     | Passive (HODL)            | Buy & Hold     | -17.85%      |  -1.5215 |   -1.4977 | -29.36%        |        1 |               0 |
| ETH     | DQN (5-Action) [Proposed] | Neuro-Symbolic | -17.79%      |  -1.3689 |   -1.2691 | -31.60%        |      314 |             114 |
| ETH     | Double DQN (5-Action)     | Neuro-Symbolic | -24.77%      |  -1.7226 |   -1.6733 | -39.25%        |      314 |             214 |
| ETH     | Double DQN (5-Action)     | Pure Baseline  | -17.74%      |  -1.0609 |   -1.044  | -36.80%        |      110 |               0 |
| ETH     | PPO (5-Action)            | Neuro-Symbolic | -10.29%      |  -3.3702 |   -1.7811 | -10.86%        |      294 |               3 |
| ETH     | PPO (5-Action)            | Pure Baseline  | -9.39%       |  -1.3452 |   -0.5805 | -13.65%        |       85 |               0 |
| ETH     | A2C (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| ETH     | A2C (5-Action)            | Pure Baseline  | -26.37%      |  -3.3257 |   -2.2964 | -29.59%        |      120 |               0 |
| ETH     | Passive (HODL)            | Buy & Hold     | -18.40%      |  -1.0953 |   -1.0753 | -36.90%        |        1 |               0 |
| XRP     | DQN (5-Action) [Proposed] | Neuro-Symbolic | -33.39%      |  -2.9108 |   -2.8623 | -35.08%        |      572 |             211 |
| XRP     | Double DQN (5-Action)     | Neuro-Symbolic | -40.54%      |  -5.015  |   -4.4598 | -41.84%        |      849 |             105 |
| XRP     | Double DQN (5-Action)     | Pure Baseline  | -50.48%      |  -6.9188 |   -6.4353 | -50.80%        |      899 |               0 |
| XRP     | PPO (5-Action)            | Neuro-Symbolic | -29.00%      |  -2.1051 |   -2.0763 | -35.26%        |        1 |             311 |
| XRP     | PPO (5-Action)            | Pure Baseline  | -29.00%      |  -2.1051 |   -2.0763 | -35.26%        |        1 |               0 |
| XRP     | A2C (5-Action)            | Neuro-Symbolic | -15.91%      |  -1.1793 |   -0.9475 | -22.43%        |        1 |               0 |
| XRP     | A2C (5-Action)            | Pure Baseline  | -13.35%      |  -1.4209 |   -1.0198 | -17.16%        |       54 |               0 |
| XRP     | Passive (HODL)            | Buy & Hold     | -28.93%      |  -2.0982 |   -2.0688 | -35.26%        |        1 |               0 |