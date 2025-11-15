import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class CryptoTradingEnv(gym.Env):
    """
    Custom Environment Crypto Trading V2
    Fitur: Reward Log Return, Penalti Inaktivitas, dan Observasi ATR (Volatilitas).
    """

    def __init__(self, df, initial_balance=10000, fee=0.001):
        super(CryptoTradingEnv, self).__init__()

        self.df = df
        self.initial_balance = initial_balance
        self.fee = fee

        # Action Space: 3 pilihan (0: SELL, 1: HOLD, 2: BUY)
        self.action_space = spaces.Discrete(3)

        # Observation/State Space (9 fitur):
        # [Close, Prediction, RSI, MACD, SMA_7, ATR, USDT, BTC, Net Worth]
        # Shape diubah jadi 9 karena ada tambahan ATR
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset kondisi dompet ke modal awal
        self.balance_usdt = self.initial_balance
        self.balance_btc = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance

        # Mulai dari baris pertama data
        self.current_step = 0

        return self._next_observation(), {}

    def _next_observation(self):
        # Ambil data pasar pada timestep saat ini
        current_data = self.df.iloc[self.current_step]

        # Bentuk array state yang akan dilihat oleh agen
        # Urutan ini WAJIB SAMA dengan yang ada di ml_utils.py nanti
        obs = np.array([
            current_data['close'],
            current_data['prediction'],
            current_data['RSI'],
            current_data['MACD'],
            current_data['SMA_7'],
            current_data['ATR'],        # <-- FITUR BARU (Volatilitas)
            self.balance_usdt,
            self.balance_btc,
            self.net_worth
        ], dtype=np.float32)

        return obs

    def step(self, action):
        current_price = self.df.iloc[self.current_step]['close']

        # --- EKSEKUSI AKSI ---
        if action == 2 and self.balance_usdt > 0:  # BUY ALL
            # Hitung berapa BTC yang didapat setelah fee
            btc_bought = self.balance_usdt / current_price
            fee_cost = btc_bought * self.fee
            self.balance_btc += (btc_bought - fee_cost)
            self.balance_usdt = 0

        elif action == 0 and self.balance_btc > 0:  # SELL ALL
            # Hitung berapa USDT yang didapat setelah fee
            usdt_received = self.balance_btc * current_price
            fee_cost = usdt_received * self.fee
            self.balance_usdt += (usdt_received - fee_cost)
            self.balance_btc = 0

        # Action 1 (HOLD) tidak melakukan apa-apa pada saldo

        # --- UPDATE STATE ---
        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)

        # Hitung Net Worth baru
        next_price = self.df.iloc[self.current_step]['close']
        self.net_worth = self.balance_usdt + (self.balance_btc * next_price)

        # --- REWARD FUNCTION ---
        # 1. Base Reward: Logarithmic Return (Standar industri untuk kestabilan training)
        reward = np.log(self.net_worth / self.prev_net_worth)

        # 2. Penalty for Inactivity (Agar tidak cuma HODL saat punya cash nganggur)
        # Kita kurangi sedikit penaltinya jadi 0.00005 agar tidak terlalu 'panikan'
        if self.balance_usdt > 0 and action == 1:
            reward -= 0.00005

        # Simpan state untuk langkah selanjutnya
        self.prev_net_worth = self.net_worth
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        obs = self._next_observation()
        info = {'net_worth': self.net_worth}

        return obs, reward, done, False, info

    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(
            f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Profit: {profit:.2f}')
