"""
V18.0 PRODUCTION - NEURO-SYMBOLIC CRYPTO TRADING ENVIRONMENT
Integrates technical feature processing, deterministic symbolic veto net (ATR, RSI, SMA),
normalized observation space, and Explainable AI (XAI) audit logging for trade decisions.

V18 Changes:
- Normalized observation space to [-1, 1] range for stable DQN gradient flow.
- Rebalanced reward function: reduced TRADE_COST, increased idle_penalty,
  added first_buy_bonus to overcome cold-start degenerate HOLD policy.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.features import compute_technical_indicators


class CryptoTradingEnv(gym.Env):
    """
    Neuro-Symbolic Crypto Trading Environment with Rule-Based Deterministic Safety Net Veto.

    Obs (9 features, normalized to [-1, 1]):
      [close_norm, prediction_norm, RSI_norm, MACD_norm, SMA7_norm,
       ATR_norm, balance_usdt_norm, balance_btc_norm, net_worth_norm]

    Actions:
      0 = SELL
      1 = HOLD
      2 = BUY
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance: float = 10000.0,
        fee: float = 0.001,
        symbol: str = 'btc',
        enable_safety_net: bool = True,
        log_trades: bool = False,
    ):
        super().__init__()

        # Ensure features are computed
        if 'RSI' not in df.columns or 'ATR' not in df.columns:
            df = compute_technical_indicators(df)

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.symbol = symbol.lower()
        self.enable_safety_net = enable_safety_net
        self.log_trades = log_trades

        # ===== Asset-Specific Deterministic Veto Rules =====
        if self.symbol == 'btc':
            self.RSI_OVERBOUGHT = 80
            self.RSI_OVERSOLD = 20
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
            self.min_trade_gap = 30
            self.hyper_penalty_multiplier = 0.006

        elif self.symbol == 'eth':
            self.RSI_OVERBOUGHT = 78
            self.RSI_OVERSOLD = 22
            self.VOLATILITY_THRESHOLD_PERCENT = 12.0
            self.min_trade_gap = 25
            self.hyper_penalty_multiplier = 0.008

        elif self.symbol == 'xrp':
            self.RSI_OVERBOUGHT = 82
            self.RSI_OVERSOLD = 18
            self.VOLATILITY_THRESHOLD_PERCENT = 18.0
            self.min_trade_gap = 15
            self.hyper_penalty_multiplier = 0.003
        else:
            self.RSI_OVERBOUGHT = 80
            self.RSI_OVERSOLD = 20
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
            self.min_trade_gap = 25
            self.hyper_penalty_multiplier = 0.006

        # ===== Reward & Penalty Parameters (V18: rebalanced) =====
        self.SAFETY_NET_PENALTY = -0.05
        # Idle penalty > TRADE_COST so agent is incentivized to explore trading
        self.base_idle_penalty = 0.003
        self.idle_growth_factor = 1.02
        self.max_idle_penalty = 0.05
        self.HOLD_REWARD_POSITION = 0.002
        # Reduced from -0.01: old value was 10x idle_penalty, discouraging all trades
        self.TRADE_COST = -0.002
        self.trade_profit_multiplier = 18.0
        self.big_win_bonus = 0.06
        self.big_win_threshold = 0.01
        self.unrealized_profit_multiplier = 0.008
        self.exploration_trades_exemption = 5
        # V18: positive gradient signal for first BUY to break cold-start HOLD lock
        self.first_buy_bonus = 0.005

        # Tracking state
        self.trade_history = []
        self.xai_veto_logs = []
        self.safety_net_triggers = {
            'buy_blocked_downtrend': 0,
            'buy_blocked_overbought': 0,
            'buy_blocked_volatile': 0,
            'sell_blocked_volatile': 0,
            'hyper_trading_penalized': 0,
            'total_blocks': 0
        }

        # Spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )

        self.reset()

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

        # V18: Store initial close price for per-episode normalization reference
        self.initial_close = float(self.df.iloc[0]['close'])

        for k in self.safety_net_triggers:
            self.safety_net_triggers[k] = 0
        self.trade_history = []
        self.xai_veto_logs = []

        return self._next_observation(), {}

    def _next_observation(self):
        """
        V18: Returns normalized observation in [-1, 1] range.
        Price features normalized against episode initial close price.
        Portfolio features normalized against initial balance.
        RSI centered at 50, MACD normalized by ATR.
        """
        d = self.df.iloc[self.current_step]
        pred_val = d.get('prediction', d['close'])
        atr_val = d['ATR'] if d['ATR'] > 0 else 1e-8

        return np.array([
            (d['close'] / self.initial_close) - 1.0,            # price change from start
            (pred_val / self.initial_close) - 1.0,               # prediction change from start
            (d['RSI'] - 50.0) / 50.0,                            # RSI centered [-1, 1]
            d['MACD'] / (atr_val + 1e-8),                        # MACD relative to volatility
            (d['SMA_7'] / self.initial_close) - 1.0,             # SMA_7 change from start
            (atr_val / d['close']) * 10.0 if d['close'] > 0 else 0.0,  # NATR scaled ~[0, 1]
            (self.balance_usdt / self.initial_balance) - 1.0,    # cash position relative
            (self.balance_btc * d['close']) / self.initial_balance,  # crypto value relative
            (self.net_worth / self.initial_balance) - 1.0,       # portfolio change relative
        ], dtype=np.float32)

    def step(self, action: int):
        d = self.df.iloc[self.current_step]
        price = d['close']

        sma30 = d.get('SMA_30', d['SMA_7'])
        natr = (d['ATR'] / price) * 100.0 if price > 0 else 0.0

        is_downtrend = price < sma30
        is_overbought = d['RSI'] > self.RSI_OVERBOUGHT
        is_volatile = natr > self.VOLATILITY_THRESHOLD_PERCENT

        final_action = action
        trade_gap = None
        blocked = False
        veto_reason = None

        # ===== Symbolic Deterministic Safety Net Veto =====
        if self.enable_safety_net:
            if action == 2:  # BUY
                if is_volatile and is_downtrend:
                    final_action = 1
                    blocked = True
                    veto_reason = f"VETO_BUY: High Volatility (NATR {natr:.1f}% > {self.VOLATILITY_THRESHOLD_PERCENT}%) in Downtrend"
                    self.safety_net_triggers['buy_blocked_volatile'] += 1
                    self.safety_net_triggers['total_blocks'] += 1
                elif is_overbought:
                    final_action = 1
                    blocked = True
                    veto_reason = f"VETO_BUY: RSI Overbought ({d['RSI']:.1f} > {self.RSI_OVERBOUGHT})"
                    self.safety_net_triggers['buy_blocked_overbought'] += 1
                    self.safety_net_triggers['total_blocks'] += 1

            elif action == 0 and is_volatile and not is_downtrend:  # SELL
                final_action = 1
                blocked = True
                veto_reason = f"VETO_SELL: High Volatility (NATR {natr:.1f}%) in Uptrend"
                self.safety_net_triggers['sell_blocked_volatile'] += 1
                self.safety_net_triggers['total_blocks'] += 1

        if blocked and veto_reason:
            self.xai_veto_logs.append({
                'step': self.current_step,
                'time': self.df.index[self.current_step],
                'raw_action': action,
                'veto_reason': veto_reason
            })

        # ===== Trade Execution =====
        executed = "HOLD"

        if final_action == 2 and self.balance_usdt > 0:
            trade_gap = self.steps_since_last_trade
            btc = self.balance_usdt / price
            fee_amount = btc * self.fee

            self.balance_btc += (btc - fee_amount)
            self.balance_usdt = 0.0
            self.entry_price = price

            executed = "BUY"
            self.trade_count += 1
            self.steps_since_last_trade = 0

            if self.log_trades:
                self.trade_history.append({
                    'action': 'BUY',
                    'price': float(price),
                    'amount': float(btc - fee_amount),
                    'time': self.df.index[self.current_step]
                })

        elif final_action == 0 and self.balance_btc > 0:
            trade_gap = self.steps_since_last_trade
            usdt = self.balance_btc * price
            fee_amount = usdt * self.fee

            self.balance_usdt += (usdt - fee_amount)
            self.balance_btc = 0.0

            executed = "SELL"
            self.trade_count += 1
            self.steps_since_last_trade = 0

            if self.log_trades:
                self.trade_history.append({
                    'action': 'SELL',
                    'price': float(price),
                    'amount': float(usdt - fee_amount),
                    'time': self.df.index[self.current_step]
                })

            self.entry_price = 0.0
        else:
            self.steps_since_last_trade += 1

        # ===== Update State =====
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance_usdt + (self.balance_btc * next_price)

        # ===== Reward Function (V18: rebalanced) =====
        if self.net_worth > 0 and self.prev_net_worth > 0:
            reward = np.log(self.net_worth / self.prev_net_worth)
        else:
            reward = 0.0

        if blocked:
            reward += self.SAFETY_NET_PENALTY

        if final_action == 1 and not blocked and self.balance_btc > 0:
            reward += self.HOLD_REWARD_POSITION

        # V18: Stronger idle penalty when holding cash, incentivizes exploration
        if final_action == 1 and not blocked and self.balance_btc == 0:
            idle_steps = self.steps_since_last_trade
            idle_pen = min(
                self.base_idle_penalty * (self.idle_growth_factor ** min(idle_steps, 200)),
                self.max_idle_penalty
            )
            reward -= idle_pen

        if self.balance_btc > 0 and final_action == 1 and self.entry_price > 0:
            upct = (price - self.entry_price) / self.entry_price
            if upct > 0:
                reward += upct * self.unrealized_profit_multiplier

        # V18: First buy bonus to break cold-start HOLD lock
        if executed == "BUY" and self.trade_count <= 2:
            reward += self.first_buy_bonus

        if executed == "SELL":
            pct = (self.net_worth - self.prev_net_worth) / self.prev_net_worth
            if pct > 0:
                reward += pct * self.trade_profit_multiplier
                if pct > self.big_win_threshold:
                    reward += self.big_win_bonus
            else:
                reward += pct * 5.0

        if executed in ["BUY", "SELL"]:
            reward += self.TRADE_COST

        if executed in ["BUY", "SELL"] and trade_gap is not None:
            if self.trade_count > self.exploration_trades_exemption:
                if trade_gap < self.min_trade_gap:
                    penalty = self.hyper_penalty_multiplier * (self.min_trade_gap - trade_gap)
                    reward -= penalty
                    self.safety_net_triggers['hyper_trading_penalized'] += 1

        self.prev_net_worth = self.net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        info = {
            "net_worth": self.net_worth,
            "executed": executed,
            "blocked": blocked,
            "veto_reason": veto_reason,
            "trade_count": self.trade_count,
            "safety_triggers": dict(self.safety_net_triggers),
        }

        return self._next_observation(), reward, done, False, info
