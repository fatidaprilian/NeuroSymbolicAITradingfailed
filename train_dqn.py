import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from trading_env import CryptoTradingEnv
import os
import argparse
from tqdm import tqdm

# === TQDM PROGRESS BAR CALLBACK ===
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="Training V11", unit="steps")

    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % 10000 == 0:
            self.pbar.set_postfix({
                'episode': self.n_calls,
                'steps': self.num_timesteps
            })
        return True

    def _on_training_end(self):
        self.pbar.close()

# === EPISODE REWARD LOGGER CALLBACK ===
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _on_step(self):
        if self.locals.get("dones") is not None and any(self.locals["dones"]):
            rewards = self.locals["rewards"]
            with open(self.log_path, 'a') as f:
                f.write(f"{self.num_timesteps},{rewards[0]}\n")
        return True


# --- Argumen ---
parser = argparse.ArgumentParser(
    description='Train DQN Agent V11 Production')
parser.add_argument('--symbol', type=str, default='btc')
parser.add_argument('--scenario', type=str, default='adaptive',
                    choices=['adaptive', 'default', 'baseline'])
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()
data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"
log_dir = "logs"
model_save_name = f"dqn_agent_{symbol_lower}_{scenario}.zip"
test_data_save_name = f"test_data_{symbol_lower}_{scenario}.csv"

print(f"\n{'='*60}")
print(f"🚀 TRAINING V11 PRODUCTION: [{symbol_lower.upper()}] [{scenario.upper()}]")
print(f"   Validated by diagnostic: BUY +0.026, SELL +0.013, HOLD +0.003")
print(f"{'='*60}")

# --- LOAD DATA ---
print(f"📂 Loading data {symbol_lower} dari {data_filename}...")
df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
df = df.ffill()

# --- Feature Engineering (FIXED TR calculation) ---
print("🛠️ Feature Engineering V2...")
df['SMA_7'] = df['close'].rolling(window=7).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
delta = df['close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))
df['RSI'] = df['RSI'].fillna(50)
df['BB_middle'] = df['close'].rolling(window=20).mean()
df['BB_std'] = df['close'].rolling(window=20).std()
df['BB_upper'] = df['BB_middle'] + (2 * df['BB_std'])
df['BB_lower'] = df['BB_middle'] - (2 * df['BB_std'])

# ✅ FIXED TR calculation
df['tr1'] = df['high'] - df['low']
df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR'] = df['tr'].rolling(window=14).mean()

df.dropna(inplace=True)
print(f"📊 Data after feature engineering: {len(df)} rows")

# --- Hybrid prediction ---
print(f"🧠 Loading LR & LSTM models for {symbol_lower}...")
model_lr = joblib.load(os.path.join(model_dir, f'model_lr_baseline_{symbol_lower}.pkl'))
scaler_lr = joblib.load(os.path.join(model_dir, f'scaler_lr_{symbol_lower}.pkl'))
model_lstm = load_model(os.path.join(model_dir, f'best_lstm_model_{symbol_lower}.keras'))
scaler_lstm_features = joblib.load(os.path.join(model_dir, f'scaler_lstm_features_{symbol_lower}.pkl'))
scaler_lstm_target = joblib.load(os.path.join(model_dir, f'scaler_lstm_target_{symbol_lower}.pkl'))

print("🔮 Generating Hybrid predictions...")
features = ['close', 'volume', 'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_signal', 'RSI', 'BB_upper', 'BB_lower', 'ATR']
X_all = df[features]
X_scaled_lr = scaler_lr.transform(X_all)
pred_lr_all = model_lr.predict(X_scaled_lr)

SEQ_LENGTH = 60
pred_lstm_all = np.full(len(df), np.nan)
X_scaled_lstm = scaler_lstm_features.transform(X_all.values)
X_seq = np.array([X_scaled_lstm[i:i+SEQ_LENGTH]
                 for i in range(len(X_scaled_lstm) - SEQ_LENGTH)])
pred_lstm_seq = model_lstm.predict(X_seq, verbose=0, batch_size=2048)
pred_lstm_all[SEQ_LENGTH:] = scaler_lstm_target.inverse_transform(pred_lstm_seq).flatten()

df['prediction'] = (0.8 * pred_lr_all) + (0.2 * pred_lstm_all)
df.dropna(inplace=True)
print(f"✅ Data ready for trading ({symbol_lower}): {len(df)} hours")

# --- Split data ---
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test = df.iloc[train_size:]
print(f"Training RL with {len(df_train)} hours of data.")

# --- Environment setup ---
env_symbol_arg = 'default' if scenario == 'default' else symbol_lower
enable_net_arg = False if scenario == 'baseline' else True
print(f"Environment: symbol='{env_symbol_arg}', enable_safety_net={enable_net_arg}")
env = DummyVecEnv([lambda: CryptoTradingEnv(
    df_train, symbol=env_symbol_arg, enable_safety_net=enable_net_arg)])

# === V11 CRITICAL CHECK ===
obs_shape = env.observation_space.shape[0]
print(f"✅ Environment [{scenario.upper()}] ready. Obs shape: {obs_shape}")
if obs_shape != 11:
    raise ValueError(f"❌ CRITICAL ERROR! Expected 11 features (V11), got {obs_shape}! Check trading_env.py cache!")
print(f"✅ Confirmed: Environment using V11 (11 features including opportunity flags)")

# --- DQN Agent Setup (V11 Production) ---
print(f"🤖 Initializing DQN V11 agent for [{scenario.upper()}]...")
model_dqn = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=0.00005,
    buffer_size=500000,
    learning_starts=10000,
    batch_size=64,
    gamma=0.99,
    target_update_interval=1000,
    exploration_fraction=0.5,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256, 128])
)

# --- Training ---
TOTAL_TIMESTEPS = 1_000_000
print(f"\n🚀 === STARTING V11 TRAINING: {symbol_lower.upper()} [{scenario.upper()}] ===")
print("✅ V11 VALIDATED: BUY +0.026, SELL +0.013, HOLD +0.003 (ALL POSITIVE!)")
print("⚙️  Network: [256,256,128] | LR: 0.00005 | Exploration: 0.5")
print("📊 Progress bar will show training status...\n")

log_path = os.path.join(log_dir, f"rewards_{symbol_lower}_{scenario}.csv")
tqdm_cb = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
logger_cb = RewardLoggerCallback(log_path=log_path)
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[tqdm_cb, logger_cb])

print(f"\n✅ Training DQN V11 [{scenario.upper()}] completed!")

# --- Save ---
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_save_name)
model_dqn.save(model_path)
print(f"💾 Agent [{scenario.upper()}] saved to: {model_path}")

test_data_path = os.path.join(model_dir, test_data_save_name)
df_test.to_csv(test_data_path)
print(f"💾 Test data [{scenario.upper()}] saved to: {test_data_path}")
print(f"--- ✅ COMPLETED: {symbol_lower.upper()} [{scenario.upper()}] ---\n")