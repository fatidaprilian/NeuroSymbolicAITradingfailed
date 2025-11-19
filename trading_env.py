import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CryptoTradingEnv(gym.Env):
    """
    V17.1 PRODUCTION — COST-OF-ACTION + POSITIONAL HOLD REWARD

    Perbedaan utama dari V17:

    - HOLD_REWARD hanya diberikan ketika:
        -> final_action == HOLD
        -> DAN sedang memegang BTC (balance_btc > 0)
      => Jadi HOLD di cash (full USDT) tidak diganjar reward.
    - Idle penalty kecil hanya ketika full USDT (belum/ tidak pegang posisi):
        -> Mendorong agent untuk sesekali masuk market.
    - BUY/SELL tetap kena TRADE_COST < 0.
    - Log-return net worth tetap jadi komponen reward utama.
    - Safety net & hyper-trading penalty tetap aktif.

    Obs (9 features):
      [close, prediction, RSI, MACD, SMA_7, ATR,
       balance_usdt, balance_btc, net_worth]

    Actions:
      0 = SELL
      1 = HOLD
      2 = BUY
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        df,
        initial_balance=10000,
        fee=0.001,
        symbol='btc',
        enable_safety_net=True,
        log_trades=False,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.symbol = symbol.lower()
        self.enable_safety_net = enable_safety_net
        self.log_trades = log_trades

        # ===== SYMBOL-specific settings (selaras V17 diagnostic) =====
        if self.symbol == 'btc':
            self.RSI_OVERBOUGHT = 80
            self.RSI_OVERSOLD = 20
            self.VOLATILITY_THRESHOLD_PERCENT = 15

            self.min_trade_gap = 30
            self.hyper_penalty_multiplier = 0.006

        elif self.symbol == 'eth':
            self.RSI_OVERBOUGHT = 78
            self.RSI_OVERSOLD = 22
            self.VOLATILITY_THRESHOLD_PERCENT = 12

            self.min_trade_gap = 25
            self.hyper_penalty_multiplier = 0.008

        elif self.symbol == 'xrp':
            self.RSI_OVERBOUGHT = 82
            self.RSI_OVERSOLD = 18
            self.VOLATILITY_THRESHOLD_PERCENT = 18

            self.min_trade_gap = 15
            self.hyper_penalty_multiplier = 0.003
        else:
            self.RSI_OVERBOUGHT = 80
            self.RSI_OVERSOLD = 20
            self.VOLATILITY_THRESHOLD_PERCENT = 15
            self.min_trade_gap = 25
            self.hyper_penalty_multiplier = 0.006

        # ===== Safety Net Penalty =====
        self.SAFETY_NET_PENALTY = -0.05

        # ===== Reward params V17.1 =====

        # Idle penalty: kecil, cuma ketika full USDT
        self.base_idle_penalty = 0.001
        self.idle_growth_factor = 1.005
        self.max_idle_penalty = 0.02

        # HOLD reward hanya saat memegang posisi (BTC > 0)
        self.HOLD_REWARD_POSITION = 0.002  # lebih kecil dari 0.003 global

        # Biaya aksi BUY/SELL → trading tetap mahal
        self.TRADE_COST = -0.01

        # Reward trade (realized PnL)
        self.trade_profit_multiplier = 18.0
        self.big_win_bonus = 0.06
        self.big_win_threshold = 0.01

        # Tidak ada action bonus brute-force
        self.action_bonus = 0.0

        # Unrealized profit reward ketika HOLD posisi untung
        self.unrealized_profit_multiplier = 0.008

        # Hyper-trading control
        self.consecutive_idle_threshold = 90
        self.exploration_trades_exemption = 5

        # Trade history untuk analisis/backtest
        self.trade_history = []

        # Safety net triggers
        self.safety_net_triggers = {
            'buy_blocked_downtrend': 0,
            'buy_blocked_overbought': 0,
            'buy_blocked_volatile': 0,
            'sell_blocked_volatile': 0,
            'hyper_trading_penalized': 0,
            'total_blocks': 0
        }

        # ===== Spaces =====
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

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
        self.entry_price = 0.0

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

        sma30 = d.get('SMA_30', d['SMA_7'])
        natr = (d['ATR'] / price) * 100 if price > 0 else 0.0

        is_downtrend = price < sma30
        is_overbought = d['RSI'] > self.RSI_OVERBOUGHT
        is_volatile = natr > self.VOLATILITY_THRESHOLD_PERCENT

        final_action = action
        trade_gap = None
        blocked = False

        # ===== Safety Net =====
        if self.enable_safety_net:
            if action == 2:  # BUY
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
            self.balance_usdt = 0.0
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
            self.balance_btc = 0.0

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

            self.entry_price = 0.0
        else:
            self.steps_since_last_trade += 1

        # ===== Update step & net worth =====
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance_usdt + (self.balance_btc * next_price)

        # ===== BASE REWARD (log-return net worth) =====
        if self.net_worth > 0 and self.prev_net_worth > 0:
            reward = np.log(self.net_worth / self.prev_net_worth)
        else:
            reward = 0.0

        # ===== SAFETY PENALTY =====
        if blocked:
            reward += self.SAFETY_NET_PENALTY

        # ===== HOLD REWARD (hanya kalau ADA posisi) =====
        if final_action == 1 and not blocked and self.balance_btc > 0:
            reward += self.HOLD_REWARD_POSITION

        # ===== Idle penalty kecil kalau full USDT =====
        if final_action == 1 and not blocked and self.balance_btc == 0:
            idle_steps = self.steps_since_last_trade
            idle_pen = self.base_idle_penalty * (
                self.idle_growth_factor ** min(idle_steps, 200)
            )
            idle_pen = min(idle_pen, self.max_idle_penalty)
            reward -= idle_pen

        # ===== Unrealized profit reward (posisi HOLD yang untung) =====
        if self.balance_btc > 0 and final_action == 1 and self.entry_price > 0:
            upct = (price - self.entry_price) / self.entry_price
            if upct > 0:
                reward += upct * self.unrealized_profit_multiplier

        # ===== Realized PnL (SELL) =====
        if executed == "SELL":
            pct = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
            if pct > 0:
                reward += pct * self.trade_profit_multiplier
                if pct > self.big_win_threshold:
                    reward += self.big_win_bonus
            else:
                reward += pct * 5.0

        # ===== Cost-of-Action (BUY/SELL) =====
        if executed in ["BUY", "SELL"]:
            reward += self.TRADE_COST

        # ===== Hyper trading control =====
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

        self.prev_net_worth = self.net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        info = {
            "net_worth": self.net_worth,
            "executed": executed,
            "trade_count": self.trade_count,
            "safety_triggers": dict(self.safety_net_triggers),
        }

        return self._next_observation(), reward, done, False, info
