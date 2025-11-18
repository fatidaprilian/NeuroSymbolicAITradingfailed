import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import json
import os


class CryptoTradingEnvDiagnostic(gym.Env):
    """
    V11 DIAGNOSTIC: OPPORTUNITY-AWARE REWARD + SOFTER PENALTIES

    Perubahan utama dari V10.x:
    - Tambah 2 fitur biner di state:
        * is_opportunity_buy
        * is_opportunity_sell
      → DQN bisa "melihat" kapan kondisi peluang BUY/SELL aktif.

    - opportunity_reward & exit_reward dinaikkan:
        0.07 -> 0.15  (mendorong trading saat sinyal kuat)

    - idle penalty dilunakkan:
        base_idle_penalty: 0.018 -> 0.006
        max_idle_penalty  : 0.35  -> 0.12
      → HOLD tidak lagi dihukum terlalu keras, tapi tetap ada cost.

    - hyper_trading penalty dipotong:
        hyper_penalty_multiplier: ~0.002 -> 0.0008
      → BUY/SELL yang terlalu rapat masih dihukum, tapi tidak
        menghapus semua reward trade yang bagus.

    Tujuan V11:
    - Agent mau mengambil peluang (opportunity usage naik)
    - HOLD jadi pilihan netral (bukan dosa besar, bukan gratis)
    - Reward trading (BUY/SELL) secara statistik > HOLD
    - Tetap ada kontrol anti-hyper-trading & safety net simbolik
    """

    def __init__(self, df, initial_balance=10000, fee=0.001,
                 symbol='btc', enable_safety_net=True, log_trades=False,
                 diagnostic_mode=True, diagnostic_log_path='diagnostics'):
        super(CryptoTradingEnvDiagnostic, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.fee = fee
        self.symbol = symbol.lower()
        self.enable_safety_net = enable_safety_net
        self.log_trades = log_trades

        # DIAGNOSTIC MODE
        self.diagnostic_mode = diagnostic_mode
        self.diagnostic_log_path = diagnostic_log_path
        if self.diagnostic_mode:
            os.makedirs(self.diagnostic_log_path, exist_ok=True)

        # Diagnostic logs
        self.reward_components_log = []
        self.action_log = []
        self.state_log = []
        self.episode_count = 0

        # === PER-ASSET SAFETY NET THRESHOLDS ===
        if self.symbol == 'btc':
            self.RSI_OVERBOUGHT = 80.0
            self.RSI_OVERSOLD = 20.0
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
            self.min_trade_gap = 10
            self.hyper_penalty_multiplier = 0.0008

        elif self.symbol == 'eth':
            self.RSI_OVERBOUGHT = 78.0
            self.RSI_OVERSOLD = 22.0
            self.VOLATILITY_THRESHOLD_PERCENT = 12.0
            self.min_trade_gap = 8
            self.hyper_penalty_multiplier = 0.0008

        elif self.symbol == 'xrp':
            self.RSI_OVERBOUGHT = 82.0
            self.RSI_OVERSOLD = 18.0
            self.VOLATILITY_THRESHOLD_PERCENT = 18.0
            self.min_trade_gap = 4
            self.hyper_penalty_multiplier = 0.0012

        else:  # default
            self.RSI_OVERBOUGHT = 80.0
            self.RSI_OVERSOLD = 20.0
            self.VOLATILITY_THRESHOLD_PERCENT = 15.0
            self.min_trade_gap = 6
            self.hyper_penalty_multiplier = 0.0008

        self.SAFETY_NET_PENALTY = -0.005

        # === V11: FINALIZED PARAMETERS ===

        # Idle penalty: lembut tapi tetap ada biaya diam
        self.base_idle_penalty = 0.006
        self.idle_growth_factor = 1.02
        self.max_idle_penalty = 0.12

        # Opportunity & exit rewards: kuat supaya agent mau trading
        self.opportunity_reward = 0.15
        self.missed_opportunity_penalty = 0.005
        self.exit_reward = 0.15
        self.forced_trade_bonus = 0.05
        self.action_bonus = 0.05
        self.unrealized_profit_multiplier = 0.005

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
        self.entry_price = 0.0

        # Action: 0 = SELL ALL, 1 = HOLD, 2 = BUY ALL
        self.action_space = spaces.Discrete(3)

        # Observation:
        # [close, prediction, RSI, MACD, SMA_7, ATR,
        #  balance_usdt, balance_btc, net_worth,
        #  is_opportunity_buy, is_opportunity_sell]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # CORE GYM METHODS
    # ------------------------------------------------------------------ #

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Save previous episode logs
        if self.diagnostic_mode and len(self.reward_components_log) > 0:
            self._save_episode_diagnostics()

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

        # Reset diagnostic logs
        self.reward_components_log = []
        self.action_log = []
        self.state_log = []
        self.episode_count += 1

        return self._next_observation(), {}

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

    def _next_observation(self):
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']

        # Hitung flags untuk step ini supaya bisa dipakai di state
        flags = self._compute_market_flags(current_data, current_price)
        opp_buy_flag = 1.0 if flags['opp_buy'] else 0.0
        opp_sell_flag = 1.0 if flags['opp_sell'] else 0.0

        obs = np.array([
            current_data['close'],
            current_data['prediction'],
            current_data['RSI'],
            current_data['MACD'],
            current_data['SMA_7'],
            current_data['ATR'],
            self.balance_usdt,
            self.balance_btc,
            self.net_worth,
            opp_buy_flag,
            opp_sell_flag
        ], dtype=np.float32)
        return obs

    def step(self, action):
        current_data = self.df.iloc[self.current_step]
        current_price = current_data['close']

        flags = self._compute_market_flags(current_data, current_price)

        # DIAGNOSTIC: Log state
        if self.diagnostic_mode:
            self.state_log.append({
                'step': self.current_step,
                'close': float(current_price),
                'prediction': float(current_data['prediction']),
                'rsi': float(current_data['RSI']),
                'macd': float(current_data['MACD']),
                'balance_usdt': float(self.balance_usdt),
                'balance_btc': float(self.balance_btc),
                'net_worth': float(self.net_worth),
                'flags': {
                    k: bool(v) if isinstance(v, (bool, np.bool_)) else float(v)
                    for k, v in flags.items()
                }
            })

        final_action = action
        reward_penalty = 0.0
        blocked = False
        block_reason = ""
        trade_gap = None

        # Initialize reward components for logging
        reward_components = {
            'base_return': 0.0,
            'safety_penalty': 0.0,
            'idle_penalty': 0.0,
            'unrealized_profit': 0.0,
            'trade_profit': 0.0,
            'action_bonus': 0.0,
            'opportunity_reward': 0.0,
            'opportunity_penalty': 0.0,
            'exit_reward': 0.0,
            'hyper_penalty': 0.0
        }

        # === SYMBOLIC SAFETY NET ===
        if self.enable_safety_net:
            is_downtrend = flags['is_downtrend']
            is_too_volatile = flags['is_too_volatile']

            if action == 2 and is_too_volatile and is_downtrend:
                final_action = 1
                reward_penalty += self.SAFETY_NET_PENALTY
                reward_components['safety_penalty'] = self.SAFETY_NET_PENALTY
                blocked = True
                block_reason = "EXTREME_VOLATILE+DOWNTREND"
                self.safety_net_triggers['buy_blocked_volatile'] += 1
                self.safety_net_triggers['total_blocks'] += 1

            if action == 0 and is_too_volatile and not is_downtrend:
                final_action = 1
                reward_penalty += self.SAFETY_NET_PENALTY
                reward_components['safety_penalty'] = self.SAFETY_NET_PENALTY
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
            self.entry_price = current_price
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
            self.entry_price = 0.0

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
            base_reward = np.log(self.net_worth / self.prev_net_worth)
            reward_components['base_return'] = base_reward
        else:
            base_reward = 0.0

        reward = base_reward + reward_penalty

        # === IDLE PENALTY (V11, lebih lembut) ===
        if final_action == 1 and not blocked:
            idle_steps = self.steps_since_last_trade
            idle_penalty = self.base_idle_penalty * (
                self.idle_growth_factor ** min(idle_steps, 200)
            )
            idle_penalty = min(idle_penalty, self.max_idle_penalty)

            if idle_steps > self.consecutive_idle_threshold:
                catastrophic_penalty = 0.3 * (
                    idle_steps - self.consecutive_idle_threshold
                ) / 100
                idle_penalty += min(catastrophic_penalty, 0.3)

            if self.balance_usdt > 0:
                reward -= idle_penalty
                reward_components['idle_penalty'] = -idle_penalty
            elif self.balance_btc > 0:
                reward -= idle_penalty * 0.5
                reward_components['idle_penalty'] = -idle_penalty * 0.5

        # === UNREALIZED PROFIT REWARD ===
        if self.balance_btc > 0 and final_action == 1 and self.entry_price > 0:
            unrealized_profit_pct = (
                current_price - self.entry_price) / self.entry_price
            if unrealized_profit_pct > 0:
                unr_reward = unrealized_profit_pct * self.unrealized_profit_multiplier
                reward += unr_reward
                reward_components['unrealized_profit'] = unr_reward

        # === TRADE PROFIT/LOSS REWARD ===
        if executed_action == "SELL":
            profit_pct = (self.net_worth - self.prev_net_worth) / \
                self.prev_net_worth
            if profit_pct > 0:
                trade_rew = profit_pct * 20.0
                reward += trade_rew
                reward_components['trade_profit'] = trade_rew
                if profit_pct > 0.01:
                    bonus = 0.05
                    reward += bonus
                    reward_components['action_bonus'] += bonus
            else:
                trade_pen = profit_pct * 5.0
                reward += trade_pen
                reward_components['trade_profit'] = trade_pen

        # === ACTION BONUS ===
        if executed_action in ["BUY", "SELL"]:
            reward += self.action_bonus
            reward_components['action_bonus'] += self.action_bonus

        # === OPPORTUNITY & EXIT REWARDS ===
        opp_buy = flags['opp_buy']
        opp_sell = flags['opp_sell']
        strong_exit = flags['strong_exit']

        if opp_buy and not blocked:
            if executed_action == "BUY":
                opp_rew = self.opportunity_reward + self.forced_trade_bonus
                reward += opp_rew
                reward_components['opportunity_reward'] = opp_rew
                self.safety_net_triggers['opportunity_used'] += 1
            elif final_action == 1 and self.balance_usdt > 0:
                opp_pen = -self.missed_opportunity_penalty
                reward += opp_pen
                reward_components['opportunity_penalty'] = opp_pen

        if opp_sell and self.balance_btc > 0 and not blocked:
            if executed_action == "SELL":
                exit_rew = self.exit_reward + self.forced_trade_bonus
                reward += exit_rew
                reward_components['exit_reward'] = exit_rew
                self.safety_net_triggers['exit_used'] += 1
            elif final_action == 1:
                exit_pen = -self.missed_opportunity_penalty * 0.5
                reward += exit_pen
                reward_components['opportunity_penalty'] += exit_pen

        if strong_exit and self.balance_btc > 0 and not blocked:
            if executed_action == "SELL":
                strong_rew = self.exit_reward + self.forced_trade_bonus
                reward += strong_rew
                reward_components['exit_reward'] += strong_rew
            elif final_action == 1:
                strong_pen = -self.missed_opportunity_penalty
                reward += strong_pen
                reward_components['opportunity_penalty'] += strong_pen

        # === ANTI-HYPER-TRADING ===
        if executed_action in ["BUY", "SELL"] and trade_gap is not None:
            if self.trade_count > self.exploration_trades_exemption:
                if trade_gap < self.min_trade_gap:
                    gap_violation = self.min_trade_gap - trade_gap
                    hyper_penalty = self.hyper_penalty_multiplier * gap_violation
                    reward -= hyper_penalty
                    reward_components['hyper_penalty'] = -hyper_penalty
                    self.safety_net_triggers['hyper_trading_penalized'] += 1

        # DIAGNOSTIC: Log action and rewards
        if self.diagnostic_mode:
            self.action_log.append({
                'step': self.current_step - 1,
                'action_requested': int(action),
                'action_final': int(final_action),
                'action_executed': executed_action,
                'blocked': blocked,
                'block_reason': block_reason
            })

            self.reward_components_log.append({
                'step': self.current_step - 1,
                'total_reward': float(reward),
                'components': {k: float(v) for k, v in reward_components.items()}
            })

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

    # ------------------------------------------------------------------ #
    # DIAGNOSTIC SAVE & RENDER
    # ------------------------------------------------------------------ #

    def _save_episode_diagnostics(self):
        """Save diagnostic logs for this episode"""
        episode_dir = os.path.join(
            self.diagnostic_log_path, f'episode_{self.episode_count}')
        os.makedirs(episode_dir, exist_ok=True)

        # Save reward components
        with open(os.path.join(episode_dir, 'reward_components.json'), 'w') as f:
            json.dump(self.reward_components_log, f, indent=2)

        # Save actions
        with open(os.path.join(episode_dir, 'actions.json'), 'w') as f:
            json.dump(self.action_log, f, indent=2)

        # Save states
        with open(os.path.join(episode_dir, 'states.json'), 'w') as f:
            json.dump(self.state_log, f, indent=2)

        # Save summary
        summary = {
            'episode': self.episode_count,
            'total_steps': len(self.action_log),
            'total_trades': self.trade_count,
            'final_net_worth': float(self.net_worth),
            'return_pct': float(
                (self.net_worth - self.initial_balance) /
                self.initial_balance * 100
            ),
            'safety_triggers': self.safety_net_triggers
        }
        with open(os.path.join(episode_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(
            f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, "
            f"Profit: {profit:.2f}, Trades: {self.trade_count}, "
            f"Idle Steps: {self.steps_since_last_trade}"
        )
