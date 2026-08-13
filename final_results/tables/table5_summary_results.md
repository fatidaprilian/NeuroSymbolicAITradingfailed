# Table 5: Multi-Algorithm DRL Performance & Statistical Significance

| Asset   | Algorithm        | Strategy       | Return (%)   |   Sharpe |   Sortino | Max Drawdown   |   Trades |   Safety Blocks | t-stat   | p-val   | Sig (p<0.05)   |
|:--------|:-----------------|:---------------|:-------------|---------:|----------:|:---------------|---------:|----------------:|:---------|:--------|:---------------|
| BTC     | DQN (5-Action)   | Neuro-Symbolic | -27.94%      |  -3.0202 |   -2.9875 | -33.15%        |      830 |               0 | -0.295   | 0.7684  | NO             |
| BTC     | DQN (5-Action)   | Pure Baseline  | -26.24%      |  -2.6459 |   -2.6259 | -30.52%        |      561 |               0 | -        | -       | -              |
| BTC     | PPO (Continuous) | Neuro-Symbolic | -27.94%      |  -3.0202 |   -2.9875 | -33.15%        |      830 |               0 | -0.295   | 0.7684  | NO             |
| BTC     | PPO (Continuous) | Pure Baseline  | -26.24%      |  -2.6459 |   -2.6259 | -30.52%        |      561 |               0 | -        | -       | -              |
| BTC     | -                | Buy & Hold     | -17.85%      |  -1.5215 |   -1.4977 | -29.36%        |        1 |               0 | -1.782   | 0.0748  | NO             |
| ETH     | DQN (5-Action)   | Neuro-Symbolic | -18.22%      |  -1.3996 |   -1.3049 | -32.73%        |      324 |               0 | 0.286    | 0.7748  | NO             |
| ETH     | DQN (5-Action)   | Pure Baseline  | -22.16%      |  -1.5825 |   -1.5348 | -40.90%        |      156 |               0 | -        | -       | -              |
| ETH     | PPO (Continuous) | Neuro-Symbolic | -18.22%      |  -1.3996 |   -1.3049 | -32.73%        |      324 |               0 | 0.286    | 0.7748  | NO             |
| ETH     | PPO (Continuous) | Pure Baseline  | -22.16%      |  -1.5825 |   -1.5348 | -40.90%        |      156 |               0 | -        | -       | -              |
| ETH     | -                | Buy & Hold     | -18.40%      |  -1.0953 |   -1.0753 | -36.90%        |        1 |               0 | -0.066   | 0.9475  | NO             |
| XRP     | DQN (5-Action)   | Neuro-Symbolic | -36.25%      |  -3.2188 |   -3.1569 | -37.88%        |      614 |               8 | 1.384    | 0.1665  | NO             |
| XRP     | DQN (5-Action)   | Pure Baseline  | -44.19%      |  -4.0686 |   -3.9791 | -45.58%        |      327 |               0 | -        | -       | -              |
| XRP     | PPO (Continuous) | Neuro-Symbolic | -36.25%      |  -3.2188 |   -3.1569 | -37.88%        |      614 |               8 | 1.384    | 0.1665  | NO             |
| XRP     | PPO (Continuous) | Pure Baseline  | -44.19%      |  -4.0686 |   -3.9791 | -45.58%        |      327 |               0 | -        | -       | -              |
| XRP     | -                | Buy & Hold     | -28.93%      |  -2.0982 |   -2.0688 | -35.26%        |        1 |               0 | -1.644   | 0.1003  | NO             |