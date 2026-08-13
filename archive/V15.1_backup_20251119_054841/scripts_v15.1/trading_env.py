import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CryptoTradingEnv(gym.Env):
    """
    V15.1 PRODUCTION — FINANCIAL-FOCUSED + LIGHT BOOST

    - Pure financial reward (log-return + realized PnL)
    - Idle penalty softened for 70–200 trades
    - Hyper-trading penalty softened (asset-specific)
    - Action bonus small (0.03)
    - Safety net aligned with diagnostic environment

    Obs (9 features):
      [close, prediction, RSI, MACD, SMA_7, ATR,
       balance_usdt, balance_btc, net_worth]
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, df, initial_balance=10000, fee=0.001,
                 symbol='btc', enable_safety_net=True, log_trades=False):

        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.symbol = symbol.lower()
        self.enable_safety_net = enable_safety_net
        self.log_trades = log_trades

        # ===== SYMBOL-specific settings =====
        if self.symbol == 'btc':
            self.RSI_OVERBOUGHT = 80
            self.RSI_OVERSOLD = 20
            self.VOLATILITY_THRESHOLD_PERCENT = 15
            self.min_trade_gap = 20
            self.hyper_penalty_multiplier = 0.0012     # softer (V15.1)
        elif self.symbol == 'eth':
            self.RSI_OVERBOUGHT = 78
            self.RSI_OVERSOLD = 22
            self.VOLATILITY_THRESHOLD_PERCENT = 12
            self.min_trade_gap = 16
            self.hyper_penalty_multiplier = 0.0012
        elif self.symbol == 'xrp':
            self.RSI_OVERBOUGHT = 82
            self.RSI_OVERSOLD = 18
            self.VOLATILITY_THRESHOLD_PERCENT = 18
            self.min_trade_gap = 10
            self.hyper_penalty_multiplier = 0.0009
        else:
            self.RSI_OVERBOUGHT = 80
            self.RSI_OVERSOLD = 20
            self.VOLATILITY_THRESHOLD_PERCENT = 15
            self.min_trade_gap = 18
            self.hyper_penalty_multiplier = 0.0012

        self.SAFETY_NET_PENALTY = -0.006  # softer from -0.008

        # ===== Reward shaping (V15.1) =====
        self.base_idle_penalty = 0.007
        self.idle_growth_factor = 1.02
        self.max_idle_penalty = 0.14

        self.trade_profit_multiplier = 24.0
        self.big_win_bonus = 0.06
        self.big_win_threshold = 0.01

        self.action_bonus = 0.03
        self.unrealized_profit_multiplier = 0.006

        self.consecutive_idle_threshold = 90
        self.exploration_trades_exemption = 5

        # === NEW: trade history untuk backtest ===
        self.trade_history = []

        # === NEW: safety net triggers lengkap (biar run_test nggak KeyError) ===
        self.safety_net_triggers = {
            'buy_blocked_downtrend': 0,    # ini mungkin tetap 0, gapapa
            'buy_blocked_overbought': 0,
            'buy_blocked_volatile': 0,
            'sell_blocked_volatile': 0,
            'hyper_trading_penalized': 0,
            'total_blocks': 0
        }

        # ===== Spaces =====
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        # ===== Init state =====
        self.reset()

    # ========== RESET ==========
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
        self.entry_price = 0

        # reset triggers & trade history
        for k in self.safety_net_triggers:
            self.safety_net_triggers[k] = 0
        self.trade_history = []

        return self._next_observation(), {}

    # ========== OBSERVATION ==========
    def _next_observation(self):
        d = self.df.iloc[self.current_step]

        return np.array([
            d['close'],
            d['prediction'],
            d['RSI'],
            d['MACD'],
            d['SMA_7'],
            d['ATR'],
            self.balance_usdt,
            self.balance_btc,
            self.net_worth
        ], dtype=np.float32)

    # ========== STEP ==========
    def step(self, action):
        d = self.df.iloc[self.current_step]
        price = d['close']

        # ===== Safety flags =====
        sma30 = d.get('SMA_30', d['SMA_7'])
        natr = (d['ATR'] / price) * 100

        is_downtrend = price < sma30
        is_overbought = d['RSI'] > self.RSI_OVERBOUGHT
        is_volatile = natr > self.VOLATILITY_THRESHOLD_PERCENT

        final_action = action
        trade_gap = None
        blocked = False

        # ===== Safety Net =====
        if self.enable_safety_net:
            if action == 2:
                if is_volatile and is_downtrend:
                    final_action = 1
                    blocked = True
                    self.safety_net_triggers['buy_blocked_volatile'] += 1
                    self.safety_net_triggers['total_blocks'] += 1
                elif is_overbought:
                    final_action = 1
                    blocked = True
                    self.safety_net_triggers['buy_blocked_overbought'] += 1
                    self.safety_net_triggers['total_blocks'] += 1

            if action == 0 and is_volatile and not is_downtrend:
                final_action = 1
                blocked = True
                self.safety_net_triggers['sell_blocked_volatile'] += 1
                self.safety_net_triggers['total_blocks'] += 1

        # ===== Execute trade =====
        executed = "HOLD"

        if final_action == 2 and self.balance_usdt > 0:
            trade_gap = self.steps_since_last_trade

            btc = self.balance_usdt / price
            fee = btc * self.fee

            self.balance_btc += (btc - fee)
            self.balance_usdt = 0
            self.entry_price = price

            executed = "BUY"
            self.trade_count += 1
            self.steps_since_last_trade = 0

            if self.log_trades:
                self.trade_history.append({
                    'action': 'BUY',
                    'price': float(price),
                    'amount': float(btc - fee),
                    'time': self.df.index[self.current_step]
                })

        elif final_action == 0 and self.balance_btc > 0:
            trade_gap = self.steps_since_last_trade

            usdt = self.balance_btc * price
            fee = usdt * self.fee

            self.balance_usdt += (usdt - fee)
            self.balance_btc = 0

            executed = "SELL"
            self.trade_count += 1
            self.steps_since_last_trade = 0

            if self.log_trades:
                self.trade_history.append({
                    'action': 'SELL',
                    'price': float(price),
                    'amount': float(usdt - fee),
                    'time': self.df.index[self.current_step]
                })

            self.entry_price = 0

        else:
            self.steps_since_last_trade += 1

        # ===== Update price & net worth =====
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance_usdt + (self.balance_btc * next_price)

        # ===== BASE REWARD (financial) =====
        if self.net_worth > 0 and self.prev_net_worth > 0:
            reward = np.log(self.net_worth / self.prev_net_worth)
        else:
            reward = 0

        # ===== SAFETY PENALTY =====
        if blocked:
            reward += self.SAFETY_NET_PENALTY

        # ===== IDLE PENALTY =====
        if final_action == 1 and not blocked:
            idle_steps = self.steps_since_last_trade

            idle_pen = self.base_idle_penalty * (
                self.idle_growth_factor ** min(idle_steps, 200)
            )
            idle_pen = min(idle_pen, self.max_idle_penalty)

            if self.balance_usdt > 0:
                reward -= idle_pen
            elif self.balance_btc > 0:
                reward -= idle_pen * 0.6

        # ===== Unrealized profit reward =====
        if self.balance_btc > 0 and final_action == 1 and self.entry_price > 0:
            upct = (price - self.entry_price) / self.entry_price
            if upct > 0:
                reward += upct * self.unrealized_profit_multiplier

        # ===== Realized PnL =====
        if executed == "SELL":
            pct = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
            if pct > 0:
                reward += pct * self.trade_profit_multiplier
                if pct > self.big_win_threshold:
                    reward += self.big_win_bonus
            else:
                reward += pct * 5.0

        # ===== Action bonus =====
        if executed in ["BUY", "SELL"]:
            reward += self.action_bonus

        # ===== Hyper trading =====
        if executed in ["BUY", "SELL"] and trade_gap is not None:
            if self.trade_count > self.exploration_trades_exemption:
                if trade_gap < self.min_trade_gap:
                    penalty = self.hyper_penalty_multiplier * (
                        self.min_trade_gap - trade_gap
                    )

                    reward -= penalty
                    self.safety_net_triggers['hyper_trading_penalized'] += 1

                    if self.log_trades and len(self.trade_history) > 0:
                        self.trade_history[-1]['hyper_penalty'] = float(
                            penalty)

        # ===== Store prev =====
        self.prev_net_worth = self.net_worth

        return self._next_observation(), reward, done, False, {
            "net_worth": self.net_worth,
            "executed": executed
        }
