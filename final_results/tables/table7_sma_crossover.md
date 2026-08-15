# Table 7: Performance Comparison with Rule-Based Strategy (SMA Crossover)

| Asset   | Algorithm            | Strategy       | Return (%)   |   Sharpe |   Sortino | Max Drawdown   |   Trades |   Safety Blocks |
|:--------|:---------------------|:---------------|:-------------|---------:|----------:|:---------------|---------:|----------------:|
| BTC     | DQN (5-Action)       | Neuro-Symbolic | -27.94%      |  -3.0202 |   -2.9875 | -33.15%        |      830 |               0 |
| BTC     | DQN (5-Action)       | Pure Baseline  | -26.24%      |  -2.6459 |   -2.6259 | -30.52%        |      561 |               0 |
| BTC     | SMA Crossover (7/30) | Rule-Based     | -14.93%      |  -2.1534 |   -1.5298 | -20.09%        |      112 |               0 |
| BTC     | Passive (HODL)       | Buy & Hold     | -17.85%      |  -1.5215 |   -1.4977 | -29.36%        |        1 |               0 |
| ETH     | DQN (5-Action)       | Neuro-Symbolic | -18.22%      |  -1.3996 |   -1.3049 | -32.73%        |      324 |               0 |
| ETH     | DQN (5-Action)       | Pure Baseline  | -22.16%      |  -1.5825 |   -1.5348 | -40.90%        |      156 |               0 |
| ETH     | SMA Crossover (7/30) | Rule-Based     | -17.85%      |  -1.7754 |   -1.3412 | -26.35%        |      118 |               0 |
| ETH     | Passive (HODL)       | Buy & Hold     | -18.40%      |  -1.0953 |   -1.0753 | -36.90%        |        1 |               0 |
| XRP     | DQN (5-Action)       | Neuro-Symbolic | -36.25%      |  -3.2188 |   -3.1569 | -37.88%        |      614 |               8 |
| XRP     | DQN (5-Action)       | Pure Baseline  | -44.19%      |  -4.0686 |   -3.9791 | -45.58%        |      327 |               0 |
| XRP     | SMA Crossover (7/30) | Rule-Based     | -12.46%      |  -1.2428 |   -0.8683 | -16.17%        |      104 |               0 |
| XRP     | Passive (HODL)       | Buy & Hold     | -28.93%      |  -2.0982 |   -2.0688 | -35.26%        |        1 |               0 |
