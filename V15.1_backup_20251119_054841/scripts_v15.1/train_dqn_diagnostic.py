import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from trading_env_diagnostic import CryptoTradingEnvDiagnostic  # V11.1 ✅
import os
import argparse
from tqdm import tqdm


class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps,
                         desc="Training Diagnostic V11.1", unit="steps")

    def _on_step(self):
        self.pbar.update(1)
        if self.num_timesteps % 1000 == 0:
            self.pbar.set_postfix({
                'episode': self.n_calls,
                'steps': self.num_timesteps
            })
        return True

    def _on_training_end(self):
        self.pbar.close()


# --- Argumen ---
parser = argparse.ArgumentParser(
    description='Train V11.1 Diagnostic Agent')
parser.add_argument('--symbol', type=str, default='btc')
parser.add_argument('--scenario', type=str, default='adaptive')
parser.add_argument('--steps', type=int, default=50000,
                    help='Diagnostic steps (default: 50k)')
args = parser.parse_args()

symbol_lower = args.symbol.lower()
scenario = args.scenario.lower()
TOTAL_TIMESTEPS = args.steps

data_filename = f"{symbol_lower}_1h_data.csv"
model_dir = "ml_models"

print(f"\n{'='*60}")
print(f"🔬 DIAGNOSTIC V11.1: [{symbol_lower.upper()}] [{scenario.upper()}]")
print(f"   CHANGES FROM V11:")
print(f"   - BUY conditions: AND → OR (more opportunities)")
print(f"   - SELL conditions: Added take profit (5%, 10%, 15%)")
print(f"   - Expected: 25-40% opportunity usage (was 9%)")
print(f"{'='*60}")

# --- LOAD DATA ---
print(f"📂 Loading data {symbol_lower} dari {data_filename}...")
df = pd.read_csv(data_filename, index_col='timestamp', parse_dates=True)
df = df.ffill()

# --- Feature Engineering ---
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

df['tr1'] = df['high'] - df['low']
df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
df['ATR'] = df['tr'].rolling(window=14).mean()

df.dropna(inplace=True)
print(f"📊 Data after feature engineering: {len(df)} rows")

# --- Hybrid prediction ---
print(f"🧠 Loading LR & LSTM models for {symbol_lower}...")
model_lr = joblib.load(os.path.join(
    model_dir, f'model_lr_baseline_{symbol_lower}.pkl'))
scaler_lr = joblib.load(os.path.join(
    model_dir, f'scaler_lr_{symbol_lower}.pkl'))
model_lstm = load_model(os.path.join(
    model_dir, f'best_lstm_model_{symbol_lower}.keras'))
scaler_lstm_features = joblib.load(os.path.join(
    model_dir, f'scaler_lstm_features_{symbol_lower}.pkl'))
scaler_lstm_target = joblib.load(os.path.join(
    model_dir, f'scaler_lstm_target_{symbol_lower}.pkl'))

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
pred_lstm_all[SEQ_LENGTH:] = scaler_lstm_target.inverse_transform(
    pred_lstm_seq).flatten()

df['prediction'] = (0.8 * pred_lr_all) + (0.2 * pred_lstm_all)
df.dropna(inplace=True)
print(f"✅ Data ready for diagnostic ({symbol_lower}): {len(df)} hours")

# --- Split data (use train portion only for diagnostic) ---
train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
print(f"Diagnostic with {len(df_train)} hours of training data.")

# --- Environment setup (V11.1 DIAGNOSTIC) ---
env_symbol_arg = 'default' if scenario == 'default' else symbol_lower
enable_net_arg = False if scenario == 'baseline' else True

print(
    f"Environment: symbol='{env_symbol_arg}', enable_safety_net={enable_net_arg}")

# Create wrapped environment for DQN
env = DummyVecEnv([lambda: CryptoTradingEnvDiagnostic(
    df_train,
    symbol=env_symbol_arg,
    enable_safety_net=enable_net_arg,
    diagnostic_mode=True,
    diagnostic_log_path='diagnostics'
)])

obs_shape = env.observation_space.shape[0]
print(f"✅ Diagnostic Environment [V13] ready. Obs shape: {obs_shape}")
if obs_shape != 9:  # ✅ V13 uses 9 features!
    raise ValueError(
        f"❌ CRITICAL ERROR! Expected 9 features (V13 raw), got {obs_shape}!")
print(f"✅ Confirmed: V13 with raw features (no opportunity flags)")

# --- DQN Agent Setup (SAME AS V11 - NOT AGGRESSIVE) ---
print(f"🤖 Initializing DQN diagnostic agent...")
model_dqn = DQN(
    "MlpPolicy",
    env,
    verbose=0,
    learning_rate=0.00005,         # Same as V11 (not aggressive)
    buffer_size=500000,
    learning_starts=10000,         # Same as V11
    batch_size=64,                 # Same as V11
    gamma=0.99,
    target_update_interval=1000,   # Same as V11
    exploration_fraction=0.5,      # Same as V11 (50% exploration)
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,    # Same as V11
    policy_kwargs=dict(net_arch=[256, 256, 128])
)

# --- Training ---
print(
    f"\n🚀 === STARTING V11.1 DIAGNOSTIC: {symbol_lower.upper()} [{scenario.upper()}] ===")
print(f"📊 Diagnostic steps: {TOTAL_TIMESTEPS:,}")
print("🔬 Testing: Relaxed opportunities + take profit triggers")
print("🎯 Expected: Opportunity usage 25-40% (vs 9% in V11)\n")

tqdm_cb = TqdmCallback(total_timesteps=TOTAL_TIMESTEPS)
model_dqn.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[tqdm_cb])

print(f"\n✅ Diagnostic training completed!")
print(f"📁 Diagnostic logs saved to: diagnostics/")
print(f"\n🔍 Next: Run analysis with:")
print(f"   python analyze_diagnostics.py")
print(f"\n{'='*60}")
