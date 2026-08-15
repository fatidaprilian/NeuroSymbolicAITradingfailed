# Table 6: Comparative Analysis of DRL Algorithms (DQN vs Double DQN vs PPO vs A2C vs Buy & Hold)

| Asset   | Algorithm                 | Strategy       | Return (%)   |   Sharpe |   Sortino | Max Drawdown   |   Trades |   Safety Blocks |
|:--------|:--------------------------|:---------------|:-------------|---------:|----------:|:---------------|---------:|----------------:|
| BTC     | DQN (5-Action) [Proposed] | Neuro-Symbolic | -27.94%      |  -3.0202 |   -2.9875 | -33.15%        |      830 |               0 |
| BTC     | Double DQN (5-Action)     | Neuro-Symbolic | -22.74%      |  -2.442  |   -2.4366 | -28.95%        |      434 |               0 |
| BTC     | Double DQN (5-Action)     | Pure Baseline  | -34.85%      |  -4.2387 |   -4.003  | -37.55%        |      706 |               0 |
| BTC     | PPO (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| BTC     | A2C (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| BTC     | Passive (HODL)            | Buy & Hold     | -17.85%      |  -1.5215 |   -1.4977 | -29.36%        |        1 |               0 |
| ETH     | DQN (5-Action) [Proposed] | Neuro-Symbolic | -18.22%      |  -1.3996 |   -1.3049 | -32.73%        |      324 |               0 |
| ETH     | Double DQN (5-Action)     | Neuro-Symbolic | -23.62%      |  -1.6124 |   -1.5818 | -38.84%        |      363 |               0 |
| ETH     | Double DQN (5-Action)     | Pure Baseline  | -17.74%      |  -1.0609 |   -1.044  | -36.80%        |      110 |               0 |
| ETH     | PPO (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| ETH     | A2C (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| ETH     | Passive (HODL)            | Buy & Hold     | -18.40%      |  -1.0953 |   -1.0753 | -36.90%        |        1 |               0 |
| XRP     | DQN (5-Action) [Proposed] | Neuro-Symbolic | -36.25%      |  -3.2188 |   -3.1569 | -37.88%        |      614 |               8 |
| XRP     | Double DQN (5-Action)     | Neuro-Symbolic | -39.92%      |  -4.8885 |   -4.3212 | -41.24%        |      864 |               7 |
| XRP     | Double DQN (5-Action)     | Pure Baseline  | -50.48%      |  -6.9188 |   -6.4353 | -50.80%        |      899 |               0 |
| XRP     | PPO (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| XRP     | A2C (5-Action)            | Neuro-Symbolic | +0.00%       |   0      |    0      | 0.00%          |        0 |               0 |
| XRP     | Passive (HODL)            | Buy & Hold     | -28.93%      |  -2.0982 |   -2.0688 | -35.26%        |        1 |               0 |