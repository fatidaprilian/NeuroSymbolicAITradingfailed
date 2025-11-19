import pandas as pd
import numpy as np
import joblib
import os
import argparse
from tqdm import tqdm
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from trading_env import CryptoTradingEnv   # V17 Production


# ======================
# TQDM Progress Callback
# ======================

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(
            total=self.total_timesteps,
            desc="Training V17 Production",
            unit="steps"
        )

    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % 10000 == 0:
            self.pbar.set_postfix({'steps': self.num_timesteps})
        return True

    def _on_training_end(self):
        self.pbar.close()


# ======================
# Reward Logger Callback
# ======================

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _on_step(self):
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            rewards = self.locals["rewards"]
            with open(self.log_path, "a") as f:
                f.write(f"{self.num_timesteps},{rewards[0]}\n")
        return True


# ======================
# Argument Parser
# ======================

parser = argparse.ArgumentParser(
    description="Train DQN Agent V17 Production (Cost-of-Action + Hold Reward)"
)
parser.add_argument('--symbol', type=str, default='btc')
parser.add_argument(
    '--scenario',
    type=str,
    default='adaptive',
    choices=['adaptive', 'default', 'baseline']
)
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()

MODEL_DIR = "ml_models"
LOG_DIR = "logs"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

data_filename = f"{symbol_lower}_1h_data.csv"
model_save_name = f"dqn_agent_{symbol_lower}_{scenario}.zip"
test_data_save_name = f"test_data_{symbol_lower}_{scenario}.csv"


# ======================
# LOAD PRICE DATA
# ======================

print(f"\n{'='*60}")
print(
    f"🚀 TRAINING V17 PRODUCTION: [{symbol_lower.upper()}] [{scenario.upper()}]")
print(f"{'='*60}")

print(f"📂 Loading price data: {data_filename}")
df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
df = df.ffill()


# ======================
# Feature Engineering V2
# ======================

print("🛠 Feature Engineering...")
df['SMA_7'] = df['close'].rolling(7).mean()
df['SMA_30'] = df['close'].rolling(30).mean()
df['EMA_12'] = df['close'].ewm(span=12).mean()
df['EMA_26'] = df['close'].ewm(span=26).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9).mean()

delta = df['close'].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].fillna(50)

df['BB_middle'] = df['close'].rolling(20).mean()
df['BB_std'] = df['close'].rolling(20).std()
df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']

df['tr1'] = df['high'] - df['low']
df['tr2'] = (df['high'] - df['close'].shift()).abs()
df['tr3'] = (df['low'] - df['close'].shift()).abs()
df['ATR'] = pd.concat(
    [df['tr1'], df['tr2'], df['tr3']],
    axis=1
).max(axis=1).rolling(14).mean()

df.dropna(inplace=True)
print(f"📊 Rows after FE: {len(df)}")


# ======================
# Load Hybrid Models
# ======================

print(f"🧠 Loading LR + LSTM models for symbol: {symbol_lower}...")
model_lr = joblib.load(f"ml_models/model_lr_baseline_{symbol_lower}.pkl")
scaler_lr = joblib.load(f"ml_models/scaler_lr_{symbol_lower}.pkl")
model_lstm = load_model(f"ml_models/best_lstm_model_{symbol_lower}.keras")
scaler_lstm_features = joblib.load(
    f"ml_models/scaler_lstm_features_{symbol_lower}.pkl"
)
scaler_lstm_target = joblib.load(
    f"ml_models/scaler_lstm_target_{symbol_lower}.pkl"
)


# ======================
# Hybrid Prediction
# ======================

print("🔮 Generating hybrid predictions...")
features = [
    'close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
    'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR'
]

X_lr = scaler_lr.transform(df[features])
pred_lr = model_lr.predict(X_lr)

SEQ_LEN = 60
pred_lstm = np.full(len(df), np.nan)

X_lstm_scaled = scaler_lstm_features.transform(df[features])
X_seq = np.array([X_lstm_scaled[i:i + SEQ_LEN]
                  for i in range(len(df) - SEQ_LEN)])

pred_seq = model_lstm.predict(X_seq, verbose=0, batch_size=2048)
pred_lstm[SEQ_LEN:] = scaler_lstm_target.inverse_transform(pred_seq).flatten()

df['prediction'] = 0.8 * pred_lr + 0.2 * pred_lstm
df.dropna(inplace=True)
print(f"🔢 Data ready: {len(df)} rows")


# ======================
# Training/Test Split
# ======================

train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]

df_test.to_csv(f"{MODEL_DIR}/{test_data_save_name}")
print(f"💾 Saved test data: {test_data_save_name}")


# ======================
# Env Setup (V17)
# ======================

env_symbol = symbol_lower if scenario != 'default' else 'default'
enable_net = scenario != 'baseline'

print(f"Environment: symbol={env_symbol}, enable_safety_net={enable_net}")

env = DummyVecEnv([
    lambda: CryptoTradingEnv(df_train, symbol=env_symbol,
                             enable_safety_net=enable_net)
])

obs_shape = env.observation_space.shape[0]
if obs_shape != 9:
    raise ValueError(f"❌ ENV ERROR: Expected 9 features but got {obs_shape}.")

print("✅ Environment validated: V17 raw features (9-dim)")


# ======================
# DQN Setup
# ======================

print("🤖 Initializing DQN (V17 Production)...")

model_dqn = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-5,
    buffer_size=500000,
    learning_starts=10000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256, 128]),
    verbose=0,
)


# ======================
# TRAINING
# ======================

TOTAL_STEPS = 1_000_000
print(f"\n🚀 START TRAINING V17 — {TOTAL_STEPS:,} steps")

tqdm_cb = TqdmCallback(TOTAL_STEPS)
logger_cb = RewardLoggerCallback(
    f"{LOG_DIR}/rewards_{symbol_lower}_{scenario}.csv"
)

model_dqn.learn(total_timesteps=TOTAL_STEPS, callback=[tqdm_cb, logger_cb])

# Save model
model_path = f"{MODEL_DIR}/{model_save_name}"
model_dqn.save(model_path)

print(f"\n💾 Model saved: {model_path}")
print("🎯 Training complete.")
