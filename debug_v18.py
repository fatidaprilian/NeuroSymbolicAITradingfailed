"""
Debug script: Diagnose V18 trading environment and DQN model behavior.
Writes results to debug_output.txt for inspection.
"""
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN

# Redirect output to file
log = open("debug_output.txt", "w")

def p(msg):
    print(msg)
    log.write(msg + "\n")
    log.flush()

p("=" * 60)
p("V18 ENVIRONMENT & MODEL DIAGNOSTIC")
p("=" * 60)

# --- 1. Check observation from actual test data ---
try:
    from src.features import load_and_preprocess_data, train_val_test_split
    from trading_env import CryptoTradingEnv

    df = load_and_preprocess_data('btc')
    _, _, df_test = train_val_test_split(df)

    env = CryptoTradingEnv(df_test, symbol='btc', enable_safety_net=True, log_trades=True)
    obs, _ = env.reset()

    p(f"\nTest data rows: {len(df_test)}")
    p(f"initial_close: {env.initial_close}")
    p(f"\nFirst observation (normalized):")
    labels = ['close_norm','pred_norm','RSI_norm','MACD_norm','SMA7_norm','ATR_norm','usdt_norm','btc_norm','nw_norm']
    for lbl, v in zip(labels, obs):
        p(f"  {lbl}: {v:.6f}")

    p(f"\nAny NaN in obs: {np.isnan(obs).any()}")
    p(f"Any Inf in obs: {np.isinf(obs).any()}")

except Exception as e:
    p(f"ENV ERROR: {e}")
    import traceback
    traceback.print_exc(file=log)

# --- 2. Load model and check Q-values ---
p("\n" + "=" * 60)
p("DQN MODEL Q-VALUE CHECK")
p("=" * 60)

try:
    model = DQN.load('ml_models/dqn_agent_btc_adaptive.zip')
    names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

    test_cases = {
        'Neutral (zeros)': np.zeros(9, dtype=np.float32),
        'Bullish signal':  np.array([0.05, 0.06, -0.3, 0.5, 0.04, 0.01, 0.0, 0.0, 0.0], dtype=np.float32),
        'Bearish signal':  np.array([-0.1, -0.12, 0.7, -0.8, -0.09, 0.03, 0.0, 0.0, 0.0], dtype=np.float32),
        'Actual first obs': obs,
    }

    for name, test_obs in test_cases.items():
        a, _ = model.predict(test_obs, deterministic=True)
        t = torch.FloatTensor(test_obs).unsqueeze(0)
        with torch.no_grad():
            q = model.policy.q_net(t)
        q_sell, q_hold, q_buy = q[0][0].item(), q[0][1].item(), q[0][2].item()
        p(f"\n  {name}:")
        p(f"    Action: {names[int(a)]}")
        p(f"    Q-SELL={q_sell:.4f}  Q-HOLD={q_hold:.4f}  Q-BUY={q_buy:.4f}")
        dominant = "HOLD-LOCKED" if q_hold > q_sell and q_hold > q_buy else "TRADING"
        p(f"    Policy: {dominant}")

except Exception as e:
    p(f"MODEL ERROR: {e}")
    import traceback
    traceback.print_exc(file=log)

# --- 3. Run 20 actual steps and inspect actions ---
p("\n" + "=" * 60)
p("FIRST 20 ENVIRONMENT STEPS TRACE")
p("=" * 60)

try:
    env2 = CryptoTradingEnv(df_test, symbol='btc', enable_safety_net=True, log_trades=True)
    obs2, _ = env2.reset()

    for i in range(20):
        a, _ = model.predict(obs2, deterministic=True)
        obs2, reward, done, _, info = env2.step(int(a))
        p(f"  Step {i+1:2d}: action={names[int(a)]:4s} | executed={info['executed']:4s} | blocked={info['blocked']} | nw={info['net_worth']:.2f} | reward={reward:.5f}")
        if done:
            p("  DONE early!")
            break

    p(f"\n  Trades executed in first 20 steps: {env2.trade_count}")
    p(f"  Safety blocks in first 20 steps: {env2.safety_net_triggers['total_blocks']}")

except Exception as e:
    p(f"STEP TRACE ERROR: {e}")
    import traceback
    traceback.print_exc(file=log)

log.close()
print("Diagnostic complete. Results in debug_output.txt")
