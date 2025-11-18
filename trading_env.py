import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class CryptoTradingEnv(gym.Env):
    """
    V10.2 FINAL: POSITIVE REWARD DOMINANCE

    Strategy Shift: Instead of bigger penalties, give BIGGER REWARDS!

    Key Changes:
    1. Unrealized profit reward (holding winning position)
    2. Explicit sell-at-profit bonus
    3. Stronger opportunity rewards (0.01 → 0.02)
    4. Trade action bonus (0.005 → 0.01)

    Logic: Make trading MORE REWARDING than idling with penalty!

    Math:
    - Good trade reward: +0.05 to +0.20
    - Idle penalty: -0.10 to -2.0
    - Agent: "If I can get +0.20, worth more than avoiding -0.10!"
    """

    def __init__(self, df, initial_balance=10000, fee=0.001,
                 symbol='btc', enable_safety_net=True, log_trades=False):
        super(CryptoTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.fee = fee
        self.symbol = symbol.lower()
        self.enable_safety_net = enable_safety_net
        self.log_trades = log_trades

        # === PER-ASSET SAFETY NET THRESHOLDS ===
        if self.symbol == 'btc':
            self.RSI_OVERBOUGHT = 80.0
            self.RSI_OVERSOLD = 20.0
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
            self.min_trade_gap = 10
            self.hyper_penalty_multiplier = 0.001

        elif self.symbol == 'eth':
            self.RSI_OVERBOUGHT = 78.0
            self.RSI_OVERSOLD = 22.0
            self.VOLATILITY_THRESHOLD_PERCENT = 12.0
            self.min_trade_gap = 8
            self.hyper_penalty_multiplier = 0.001

        elif self.symbol == 'xrp':
            self.RSI_OVERBOUGHT = 82.0
            self.RSI_OVERSOLD = 18.0
            self.VOLATILITY_THRESHOLD_PERCENT = 18.0
            self.min_trade_gap = 4
            self.hyper_penalty_multiplier = 0.002

        else:  # default
            self.RSI_OVERBOUGHT = 80.0
            self.RSI_OVERSOLD = 20.0
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
            self.min_trade_gap = 6
            self.hyper_penalty_multiplier = 0.001

        self.SAFETY_NET_PENALTY = -0.005

        # === V10.1 PARAMS: MASSIVE IDLE PENALTY (KEEP) ===
        self.base_idle_penalty = 0.10
        self.idle_growth_factor = 1.05
        self.max_idle_penalty = 2.0

        # === V10.2 NEW: BIGGER POSITIVE REWARDS! ===
        self.opportunity_reward = 0.02          # Was 0.01 → 2x bigger!
        self.missed_opportunity_penalty = 0.005  # Keep same
        self.exit_reward = 0.02                 # Was 0.01 → 2x bigger!
        self.forced_trade_bonus = 0.01          # Was 0.005 → 2x bigger!
        self.action_bonus = 0.01                # Was 0.005 → 2x bigger!

        # V10.2 NEW: Unrealized profit reward
        self.unrealized_profit_multiplier = 0.001  # Small but positive!

        self.consecutive_idle_threshold = 100

        self.safety_net_triggers = {
            'buy_blocked_downtrend': 0,
            'buy_blocked_overbought': 0,
            'buy_blocked_volatile': 0,
            'sell_blocked_volatile': 0,
            'hyper_trading_penalized': 0,
            'opportunity_used': 0,
            'opportunity_missed': 0,
            'exit_used': 0,
            'total_blocks': 0
        }

        self.trade_history = []
        self.steps_since_last_trade = 0
        self.trade_count = 0
        self.exploration_trades_exemption = 5

        # V10.2: Track entry price for profit calculation
        self.entry_price = 0.0

        # Action: 0 = SELL ALL, 1 = HOLD, 2 = BUY ALL
        self.action_space = spaces.Discrete(3)

        # Observation: [close, prediction, RSI, MACD, SMA_7, ATR, balance_usdt, balance_btc, net_worth]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.balance_usdt = self.initial_balance
        self.balance_btc = 0.0

        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.current_step = 0
        self.steps_since_last_trade = 0
        self.trade_count = 0
        self.entry_price = 0.0

        for key in self.safety_net_triggers:
            self.safety_net_triggers[key] = 0
        self.trade_history = []

        return self._next_observation(), {}

    def _next_observation(self):
        current_data = self.df.iloc[self.current_step]
        obs = np.array([
            current_data['close'],
            current_data['prediction'],
            current_data['RSI'],
            current_data['MACD'],
            current_data['SMA_7'],
            current_data['ATR'],
            self.balance_usdt,
            self.balance_btc,
            self.net_worth
        ], dtype=np.float32)
        return obs

    def _compute_market_flags(self, current_data, current_price):
        rsi = current_data['RSI']
        macd = current_data['MACD']
        sma7 = current_data['SMA_7']
        sma30 = current_data.get('SMA_30', sma7)

        natr = (current_data['ATR'] / current_price) * \
            100 if current_price > 0 else 0.0

        is_downtrend = current_price < sma30
        is_uptrend = current_price > sma30
        is_overbought = rsi > self.RSI_OVERBOUGHT
        is_oversold = rsi < self.RSI_OVERSOLD
        is_too_volatile = natr > self.VOLATILITY_THRESHOLD_PERCENT

        prediction = current_data['prediction']

        opp_buy = (
            is_uptrend
            and prediction > 0
            and rsi < self.RSI_OVERBOUGHT
        )

        opp_sell = (
            (is_downtrend or macd < 0)
            or is_overbought
            or prediction < 0
        )

        strong_exit = (
            is_overbought
            and prediction < 0
        )

        flags = {
            'is_downtrend': is_downtrend,
            'is_uptrend': is_uptrend,
            'is_overbought': is_overbought,
            'is_oversold': is_oversold,
            'is_too_volatile': is_too_volatile,
            'natr': natr,
            'opp_buy': opp_buy,
            'opp_sell': opp_sell,
            'strong_exit': strong_exit
        }
        return flags

    def step(self, action):
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']

        flags = self._compute_market_flags(current_data, current_price)

        final_action = action
        reward_penalty = 0.0
        blocked = False
        block_reason = ""
        trade_gap = None

        # === SYMBOLIC SAFETY NET ===
        if self.enable_safety_net:
            is_downtrend = flags['is_downtrend']
            is_too_volatile = flags['is_too_volatile']

            if action == 2 and is_too_volatile and is_downtrend:
                final_action = 1
                reward_penalty += self.SAFETY_NET_PENALTY
                blocked = True
                block_reason = "EXTREME_VOLATILE+DOWNTREND"
                self.safety_net_triggers['buy_blocked_volatile'] += 1
                self.safety_net_triggers['total_blocks'] += 1

            if action == 0 and is_too_volatile and not is_downtrend:
                final_action = 1
                reward_penalty += self.SAFETY_NET_PENALTY
                blocked = True
                block_reason = "EXTREME_VOLATILE_UPTREND"
                self.safety_net_triggers['sell_blocked_volatile'] += 1
                self.safety_net_triggers['total_blocks'] += 1

        # === TRADE EXECUTION ===
        executed_action = "HOLD"

        if final_action == 2 and self.balance_usdt > 0:  # BUY ALL
            trade_gap = self.steps_since_last_trade

            btc_bought = self.balance_usdt / current_price
            fee_cost = btc_bought * self.fee
            self.balance_btc += (btc_bought - fee_cost)
            self.balance_usdt = 0.0
            self.entry_price = current_price  # V10.2: Track entry
            executed_action = "BUY"
            self.steps_since_last_trade = 0
            self.trade_count += 1

            if self.log_trades:
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'price': float(current_price),
                    'amount': float(btc_bought - fee_cost),
                    'blocked': blocked,
                    'gap': trade_gap,
                    'time': self.df.index[self.current_step]
                })

        elif final_action == 0 and self.balance_btc > 0:  # SELL ALL
            trade_gap = self.steps_since_last_trade

            usdt_received = self.balance_btc * current_price
            fee_cost = usdt_received * self.fee
            self.balance_usdt += (usdt_received - fee_cost)
            self.balance_btc = 0.0
            executed_action = "SELL"
            self.steps_since_last_trade = 0
            self.trade_count += 1
            self.entry_price = 0.0  # Reset

            if self.log_trades:
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': float(current_price),
                    'amount': float(usdt_received - fee_cost),
                    'blocked': blocked,
                    'gap': trade_gap,
                    'time': self.df.index[self.current_step]
                })
        else:
            self.steps_since_last_trade += 1

        # === UPDATE STATE ===
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)

        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance_usdt + (self.balance_btc * next_price)

        # === BASE REWARD: LOG RETURN ===
        if self.net_worth > 0 and self.prev_net_worth > 0:
            reward = np.log(self.net_worth / self.prev_net_worth)
        else:
            reward = 0.0

        reward += reward_penalty

        # === V10.1: MASSIVE IDLE PENALTY (BOTH CASH AND POSITION) ===
        if final_action == 1 and not blocked:
            idle_steps = self.steps_since_last_trade

            idle_penalty = self.base_idle_penalty * \
                (self.idle_growth_factor ** min(idle_steps, 200))
            idle_penalty = min(idle_penalty, self.max_idle_penalty)

            if idle_steps > self.consecutive_idle_threshold:
                catastrophic_penalty = 0.5 * \
                    (idle_steps - self.consecutive_idle_threshold) / 100
                idle_penalty += min(catastrophic_penalty, 0.5)

            if self.balance_usdt > 0:
                reward -= idle_penalty
            elif self.balance_btc > 0:
                reward -= idle_penalty * 0.5

        # === V10.2 NEW: UNREALIZED PROFIT REWARD! ===
        if self.balance_btc > 0 and final_action == 1 and self.entry_price > 0:
            unrealized_profit_pct = (
                current_price - self.entry_price) / self.entry_price
            if unrealized_profit_pct > 0:
                reward += unrealized_profit_pct * self.unrealized_profit_multiplier
                # Encourage holding winning positions!

        # === TRADE PROFIT/LOSS REWARD ===
        if executed_action == "SELL":
            profit_pct = (self.net_worth - self.prev_net_worth) / \
                self.prev_net_worth
            if profit_pct > 0:
                reward += profit_pct * 20.0
                # V10.2: Extra bonus for selling at profit!
                if profit_pct > 0.01:  # > 1% profit
                    reward += 0.05  # Big bonus!
            else:
                reward += profit_pct * 5.0

        # === V10.2: BIGGER ACTION BONUS ===
        if executed_action in ["BUY", "SELL"]:
            reward += self.action_bonus  # 0.01 (was 0.005)

        # === V10.2: STRONGER OPPORTUNITY REWARDS ===
        opp_buy = flags['opp_buy']
        opp_sell = flags['opp_sell']
        strong_exit = flags['strong_exit']

        if opp_buy and not blocked:
            if executed_action == "BUY":
                reward += self.opportunity_reward  # 0.02 (was 0.01)
                reward += self.forced_trade_bonus  # 0.01 (was 0.005)
                self.safety_net_triggers['opportunity_used'] += 1
            elif final_action == 1 and self.balance_usdt > 0:
                reward -= self.missed_opportunity_penalty

        if opp_sell and self.balance_btc > 0 and not blocked:
            if executed_action == "SELL":
                reward += self.exit_reward  # 0.02 (was 0.01)
                reward += self.forced_trade_bonus  # 0.01 (was 0.005)
                self.safety_net_triggers['exit_used'] += 1
            elif final_action == 1:
                reward -= self.missed_opportunity_penalty * 0.5

        if strong_exit and self.balance_btc > 0 and not blocked:
            if executed_action == "SELL":
                reward += self.exit_reward + self.forced_trade_bonus
            elif final_action == 1:
                reward -= self.missed_opportunity_penalty

        # === V10.2: EXPLICIT BONUS FOR TRADING AT OPPORTUNITY ===
        if opp_buy and final_action == 2 and not blocked:
            reward += 0.02  # Direct reward for buying at good time!

        if opp_sell and final_action == 0 and not blocked:
            reward += 0.02  # Direct reward for selling at good time!

        # === ANTI-HYPER-TRADING ===
        if executed_action in ["BUY", "SELL"] and trade_gap is not None:
            if self.trade_count > self.exploration_trades_exemption:
                if trade_gap < self.min_trade_gap:
                    gap_violation = self.min_trade_gap - trade_gap
                    hyper_penalty = self.hyper_penalty_multiplier * gap_violation
                    reward -= hyper_penalty
                    self.safety_net_triggers['hyper_trading_penalized'] += 1

                    if self.log_trades and len(self.trade_history) > 0:
                        self.trade_history[-1]['hyper_penalty'] = float(
                            hyper_penalty)

        # === BOOKKEEPING ===
        self.prev_net_worth = self.net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        obs = self._next_observation()
        info = {
            'net_worth': self.net_worth,
            'executed_action': executed_action,
            'was_blocked': blocked,
            'block_reason': block_reason,
            'safety_triggers': self.safety_net_triggers.copy(),
            'steps_since_trade': self.steps_since_last_trade,
            'trade_count': self.trade_count,
        }

        return obs, reward, done, False, info

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(
            f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, "
            f"Profit: {profit:.2f}, Trades: {self.trade_count}, "
            f"Hyper-Penalties: {self.safety_net_triggers['hyper_trading_penalized']}, "
            f"Idle Steps: {self.steps_since_last_trade}"
        )
