"""
V21.0 PRODUCTION - NEURO-SYMBOLIC 5-ACTION DISCRETE CRYPTO TRADING ENVIRONMENT
Implements 5-Action Discrete Space (Kaur et al., 2025) for Deep Q-Network (DQN) trading:
  0: SELL_ALL  (100% Crypto -> USDT)
  1: SELL_HALF (50% Crypto -> USDT)
  2: HOLD      (Do Nothing)
  3: BUY_HALF  (50% USDT -> Crypto)
  4: BUY_ALL   (100% USDT -> Crypto)

Includes Deterministic Symbolic Safety Net Veto (ATR, RSI, SMA) for action shielding (Kochliaridis et al., 2023).
Matches paper title: "A NEURO-SYMBOLIC AI TRADING ARCHITECTURE COMBINING HYBRID LR-LSTM PREDICTION, DEEP Q-NETWORK, AND SYMBOLIC SAFETY NETS"
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from src.features import compute_technical_indicators


class CryptoTradingEnv5Action(gym.Env):
    """
    Neuro-Symbolic 5-Action Discrete Crypto Trading Environment.

    Obs (9 features, normalized to [-1, 1]):
      [close_norm, prediction_norm, RSI_norm, MACD_norm, SMA7_norm,
       ATR_norm, balance_usdt_norm, balance_btc_norm, net_worth_norm]

    Actions:
      0: SELL 100% Crypto
      1: SELL 50% Crypto
      2: HOLD
      3: BUY 50% Available USDT
      4: BUY 100% Available USDT
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

        if 'RSI' not in df.columns or 'ATR' not in df.columns:
            df = compute_technical_indicators(df)

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.symbol = symbol.lower()
        self.enable_safety_net = enable_safety_net
        self.log_trades = log_trades

        # Asset-specific veto thresholds
        if self.symbol == 'btc':
            self.RSI_OVERBOUGHT = 80
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
        elif self.symbol == 'eth':
            self.RSI_OVERBOUGHT = 78
            self.VOLATILITY_THRESHOLD_PERCENT = 12.0
        elif self.symbol == 'xrp':
            self.RSI_OVERBOUGHT = 82
            self.VOLATILITY_THRESHOLD_PERCENT = 18.0
        else:
            self.RSI_OVERBOUGHT = 80
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0

        self.SAFETY_NET_PENALTY = -0.02

        # 5 Discrete Actions (Kaur et al., 2025)
        self.action_space = spaces.Discrete(5)

        # 9 Normalized Observation Features
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
        self.trade_count = 0
        self.steps_since_last_trade = 0

        self.initial_close = float(self.df.iloc[0]['close'])

        self.safety_net_triggers = {
            'buy_blocked_downtrend': 0,
            'buy_blocked_overbought': 0,
            'buy_blocked_volatile': 0,
            'total_blocks': 0
        }
        self.trade_history = []
        self.xai_veto_logs = []

        return self._next_observation(), {}

    def _next_observation(self):
        d = self.df.iloc[self.current_step]
        pred_val = d.get('prediction', d['close'])
        atr_val = d['ATR'] if d['ATR'] > 0 else 1e-8

        return np.array([
            (d['close'] / self.initial_close) - 1.0,
            (pred_val / self.initial_close) - 1.0,
            (d['RSI'] - 50.0) / 50.0,
            d['MACD'] / (atr_val + 1e-8),
            (d['SMA_7'] / self.initial_close) - 1.0,
            (atr_val / d['close']) * 10.0 if d['close'] > 0 else 0.0,
            (self.balance_usdt / self.initial_balance) - 1.0,
            (self.balance_btc * d['close']) / self.initial_balance,
            (self.net_worth / self.initial_balance) - 1.0,
        ], dtype=np.float32)

    def step(self, action):
        act_int = int(action)
        d = self.df.iloc[self.current_step]
        price = float(d['close'])

        sma30 = float(d.get('SMA_30', d['SMA_7']))
        natr = (float(d['ATR']) / price) * 100.0 if price > 0 else 0.0

        is_downtrend = price < sma30
        is_overbought = float(d['RSI']) > self.RSI_OVERBOUGHT
        is_volatile = natr > self.VOLATILITY_THRESHOLD_PERCENT

        final_action = act_int
        blocked = False
        veto_reason = None

        # ===== Symbolic Deterministic Safety Net Veto (Kochliaridis et al., 2023) =====
        if self.enable_safety_net and act_int in [3, 4]:  # BUY_HALF or BUY_ALL
            if is_volatile and is_downtrend:
                final_action = 2  # Overridden to HOLD
                blocked = True
                veto_reason = f"VETO_BUY: High Volatility (NATR {natr:.1f}%) in Downtrend"
                self.safety_net_triggers['buy_blocked_volatile'] += 1
                self.safety_net_triggers['total_blocks'] += 1
            elif is_overbought:
                final_action = 2
                blocked = True
                veto_reason = f"VETO_BUY: RSI Overbought ({d['RSI']:.1f} > {self.RSI_OVERBOUGHT})"
                self.safety_net_triggers['buy_blocked_overbought'] += 1
                self.safety_net_triggers['total_blocks'] += 1

        if blocked and veto_reason:
            self.xai_veto_logs.append({
                'step': self.current_step,
                'time': str(d.name if isinstance(d.name, (str, pd.Timestamp)) else d.get('timestamp', str(self.current_step))),
                'raw_action': act_int,
                'veto_reason': veto_reason
            })

        # ===== Trade Execution Logic (5 Discrete Actions) =====
        executed = "HOLD"

        if final_action == 4 and self.balance_usdt > 10.0:  # BUY ALL (100% USDT)
            buy_val = self.balance_usdt
            fee_usdt = buy_val * self.fee
            net_usdt = buy_val - fee_usdt
            bought_btc = net_usdt / price

            self.balance_usdt = 0.0
            self.balance_btc += bought_btc
            executed = "BUY_ALL"
            self.trade_count += 1

            if self.log_trades:
                time_val = d.name if isinstance(d.name, (str, pd.Timestamp)) else d.get('timestamp', str(self.current_step))
                self.trade_history.append({'action': 'BUY', 'price': price, 'amount': float(bought_btc), 'time': str(time_val)})

        elif final_action == 3 and self.balance_usdt > 10.0:  # BUY HALF (50% USDT)
            buy_val = self.balance_usdt * 0.5
            fee_usdt = buy_val * self.fee
            net_usdt = buy_val - fee_usdt
            bought_btc = net_usdt / price

            self.balance_usdt -= buy_val
            self.balance_btc += bought_btc
            executed = "BUY_HALF"
            self.trade_count += 1

            if self.log_trades:
                time_val = d.name if isinstance(d.name, (str, pd.Timestamp)) else d.get('timestamp', str(self.current_step))
                self.trade_history.append({'action': 'BUY', 'price': price, 'amount': float(bought_btc), 'time': str(time_val)})

        elif final_action == 0 and self.balance_btc > 1e-6:  # SELL ALL (100% BTC)
            sell_btc = self.balance_btc
            gross_usdt = sell_btc * price
            fee_usdt = gross_usdt * self.fee
            net_usdt = gross_usdt - fee_usdt

            self.balance_btc = 0.0
            self.balance_usdt += net_usdt
            executed = "SELL_ALL"
            self.trade_count += 1

            if self.log_trades:
                time_val = d.name if isinstance(d.name, (str, pd.Timestamp)) else d.get('timestamp', str(self.current_step))
                self.trade_history.append({'action': 'SELL', 'price': price, 'amount': float(sell_btc), 'time': str(time_val)})

        elif final_action == 1 and self.balance_btc > 1e-6:  # SELL HALF (50% BTC)
            sell_btc = self.balance_btc * 0.5
            gross_usdt = sell_btc * price
            fee_usdt = gross_usdt * self.fee
            net_usdt = gross_usdt - fee_usdt

            self.balance_btc -= sell_btc
            self.balance_usdt += net_usdt
            executed = "SELL_HALF"
            self.trade_count += 1

            if self.log_trades:
                time_val = d.name if isinstance(d.name, (str, pd.Timestamp)) else d.get('timestamp', str(self.current_step))
                self.trade_history.append({'action': 'SELL', 'price': price, 'amount': float(sell_btc), 'time': str(time_val)})

        # Update State
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        next_price = float(self.df.iloc[self.current_step]['close'])
        self.net_worth = self.balance_usdt + (self.balance_btc * next_price)

        # ===== Reward Function =====
        if self.net_worth > 0 and self.prev_net_worth > 0:
            reward = float(np.log(self.net_worth / self.prev_net_worth))
        else:
            reward = 0.0

        if blocked:
            reward += self.SAFETY_NET_PENALTY

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
