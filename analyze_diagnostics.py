import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# === CONFIGURATION ===
DIAGNOSTIC_DIR = "diagnostics"

print("\n" + "="*60)
print("🔬 DIAGNOSTIC ANALYSIS TOOL")
print("="*60)

# === 1. LOAD ALL EPISODE DATA ===
episodes = []
for folder in os.listdir(DIAGNOSTIC_DIR):
    if folder.startswith("episode_"):
        episode_path = os.path.join(DIAGNOSTIC_DIR, folder)
        try:
            with open(os.path.join(episode_path, 'summary.json'), 'r') as f:
                summary = json.load(f)
            with open(os.path.join(episode_path, 'reward_components.json'), 'r') as f:
                rewards = json.load(f)
            with open(os.path.join(episode_path, 'actions.json'), 'r') as f:
                actions = json.load(f)
            with open(os.path.join(episode_path, 'states.json'), 'r') as f:
                states = json.load(f)
            
            episodes.append({
                'summary': summary,
                'rewards': rewards,
                'actions': actions,
                'states': states,
                'path': episode_path
            })
        except Exception as e:
            print(f"⚠️  Warning: Could not load {folder}: {e}")

if len(episodes) == 0:
    print("\n❌ No episodes found in 'diagnostics/' folder!")
    print("   Please run: python train_dqn_diagnostic.py first\n")
    exit()

print(f"\n📊 Loaded {len(episodes)} episode(s) for analysis\n")

# === 2. AGGREGATE DATA ===
all_actions = []
all_rewards = []
all_reward_components = []

for ep in episodes:
    for action_record in ep['actions']:
        all_actions.append(action_record)
    for reward_record in ep['rewards']:
        all_rewards.append(reward_record)
        all_reward_components.append(reward_record['components'])

df_actions = pd.DataFrame(all_actions)
df_rewards = pd.DataFrame(all_rewards)
df_components = pd.DataFrame(all_reward_components)

# === 3. REWARD DISTRIBUTION BY ACTION ===
print("="*60)
print("📊 REWARD DISTRIBUTION ANALYSIS")
print("="*60)

action_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
df_actions['action_name'] = df_actions['action_final'].map(action_map)

# Merge with rewards
df_analysis = df_actions.merge(df_rewards, left_on='step', right_on='step', how='left')

# Group by action
action_stats = df_analysis.groupby('action_name')['total_reward'].agg([
    'count', 'mean', 'std', 'median'
]).round(6)

# Add positive reward percentage
pos_pct = df_analysis.groupby('action_name')['total_reward'].apply(
    lambda x: (x > 0).sum() / len(x) * 100
).round(1)
action_stats['pos_%'] = pos_pct

print("\nAction   | Count    | Mean       | Std        | Median     | Pos %")
print("-" * 80)
for action_name in ['SELL', 'HOLD', 'BUY']:
    if action_name in action_stats.index:
        row = action_stats.loc[action_name]
        print(f"{action_name:8} | {int(row['count']):8} | {row['mean']:10.6f} | "
              f"{row['std']:10.6f} | {row['median']:10.6f} | {row['pos_%']:5.1f}%")

# CRITICAL FINDING: HOLD vs Trading
if 'HOLD' in action_stats.index:
    hold_mean = action_stats.loc['HOLD', 'mean']
    buy_mean = action_stats.loc['BUY', 'mean'] if 'BUY' in action_stats.index else -999
    sell_mean = action_stats.loc['SELL', 'mean'] if 'SELL' in action_stats.index else -999
    
    if hold_mean > buy_mean and hold_mean > sell_mean:
        print("\n🔍 CRITICAL FINDING:")
        print("❌ HOLD has HIGHER mean reward than BUY/SELL!")
        print("   → This explains why agent prefers to HOLD")
        print(f"   → HOLD mean: {hold_mean:.6f}")
        print(f"   → BUY mean:  {buy_mean:.6f}")
        print(f"   → SELL mean: {sell_mean:.6f}")
        print("\n💡 RECOMMENDATION:")
        print("   1. Increase idle_penalty 5-10x")
        print("   2. Increase opportunity_reward 5x")
        print("   3. Increase action_bonus 3x")

# === 4. COMPONENT BREAKDOWN ===
print("\n" + "="*60)
print("🔬 REWARD COMPONENTS BREAKDOWN")
print("="*60)

component_means = df_components.mean().sort_values(ascending=False)
component_totals = df_components.sum().sort_values(ascending=False)

print("\nComponent               | Mean        | Total Sum")
print("-" * 60)
for comp, mean_val in component_means.items():
    total_val = component_totals[comp]
    print(f"{comp:23} | {mean_val:11.6f} | {total_val:11.2f}")

# Check imbalance
base_return_mag = abs(component_means.get('base_return', 0))
idle_penalty_mag = abs(component_means.get('idle_penalty', 0))

if base_return_mag > 0 and idle_penalty_mag / base_return_mag < 0.1:
    print("\n🔍 FINDING:")
    print("❌ Idle penalty is TOO WEAK compared to base_return!")
    print(f"   base_return magnitude: {base_return_mag:.6f}")
    print(f"   idle_penalty magnitude: {idle_penalty_mag:.6f}")
    print(f"   Ratio: {idle_penalty_mag / base_return_mag:.2%}")
    print("\n💡 RECOMMENDATION:")
    print("   Increase base_idle_penalty by 10-20x")

# === 5. ACTION FREQUENCY ===
print("\n" + "="*60)
print("📈 ACTION FREQUENCY ANALYSIS")
print("="*60)

action_counts = df_actions['action_name'].value_counts()
total_actions = len(df_actions)

print(f"\nTotal actions: {total_actions}")
for action_name in ['SELL', 'HOLD', 'BUY']:
    if action_name in action_counts.index:
        count = action_counts[action_name]
        pct = count / total_actions * 100
        print(f"  {action_name} ({action_name[0].lower()}): {count} ({pct:.1f}%)")

if 'HOLD' in action_counts.index:
    hold_pct = action_counts['HOLD'] / total_actions * 100
    if hold_pct > 90:
        print("\n🔍 CRITICAL FINDING:")
        print(f"❌ HOLD dominates: {hold_pct:.1f}% of all actions!")
        print("   → Agent has learned to HOLD as dominant strategy")
        print("\n💡 RECOMMENDATION:")
        print("   1. Increase exploration_final_eps from 0.05 to 0.15")
        print("   2. Try PPO instead of DQN (better exploration)")
        print("   3. Consider behavioral cloning warm-start")

# === 6. OPPORTUNITY ANALYSIS ===
print("\n" + "="*60)
print("🎯 OPPORTUNITY ANALYSIS")
print("="*60)

# Aggregate states and actions
buy_opportunities = 0
buy_taken = 0
buy_hold = 0
buy_sell = 0

sell_opportunities = 0
sell_taken = 0
sell_hold = 0
sell_buy = 0

for ep in episodes:
    for i, state in enumerate(ep['states']):
        flags = state['flags']
        action = ep['actions'][i] if i < len(ep['actions']) else None
        
        if not action:
            continue
        
        action_exec = action['action_executed']
        
        # Buy opportunities
        if flags.get('opp_buy', False):
            buy_opportunities += 1
            if action_exec == 'BUY':
                buy_taken += 1
            elif action_exec == 'HOLD':
                buy_hold += 1
            elif action_exec == 'SELL':
                buy_sell += 1
        
        # Sell opportunities
        if flags.get('opp_sell', False):
            sell_opportunities += 1
            if action_exec == 'SELL':
                sell_taken += 1
            elif action_exec == 'HOLD':
                sell_hold += 1
            elif action_exec == 'BUY':
                sell_buy += 1

if buy_opportunities > 0:
    print(f"\nBuy Opportunities: {buy_opportunities}")
    print(f"  Agent took BUY:  {buy_taken} ({buy_taken/buy_opportunities*100:.1f}%)")
    print(f"  Agent took HOLD: {buy_hold} ({buy_hold/buy_opportunities*100:.1f}%)")
    print(f"  Agent took SELL: {buy_sell} ({buy_sell/buy_opportunities*100:.1f}%)")
    
    if buy_taken / buy_opportunities < 0.3:
        print("  ❌ Agent ignores most buy opportunities!")

if sell_opportunities > 0:
    print(f"\nSell Opportunities: {sell_opportunities}")
    print(f"  Agent took SELL: {sell_taken} ({sell_taken/sell_opportunities*100:.1f}%)")
    print(f"  Agent took HOLD: {sell_hold} ({sell_hold/sell_opportunities*100:.1f}%)")
    print(f"  Agent took BUY:  {sell_buy} ({sell_buy/sell_opportunities*100:.1f}%)")
    
    if sell_taken / sell_opportunities < 0.3:
        print("  ❌ Agent ignores most sell opportunities!")

if buy_opportunities > 0 and buy_taken / buy_opportunities < 0.2:
    print("\n💡 RECOMMENDATION:")
    print("   1. Increase opportunity_reward from 0.02 to 0.10")
    print("   2. Add 'is_opportunity' as binary feature in state")
    print("   3. Check prediction model quality")

# === 7. VISUALIZATIONS ===
print("\n" + "="*60)
print("📊 GENERATING VISUALIZATIONS")
print("="*60)

# Plot 1: Reward distribution by action
plt.figure(figsize=(10, 6))
for action_name in ['SELL', 'HOLD', 'BUY']:
    if action_name in df_analysis['action_name'].values:
        data = df_analysis[df_analysis['action_name'] == action_name]['total_reward']
        plt.hist(data, bins=50, alpha=0.5, label=action_name)

plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Reward Distribution by Action')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'reward_distribution.png'), dpi=150)
print(f"  ✅ Saved: {DIAGNOSTIC_DIR}/reward_distribution.png")

# Plot 2: Component contributions
plt.figure(figsize=(12, 6))
component_means_sorted = component_means.sort_values()
colors = ['red' if x < 0 else 'green' for x in component_means_sorted.values]
plt.barh(range(len(component_means_sorted)), component_means_sorted.values, color=colors)
plt.yticks(range(len(component_means_sorted)), component_means_sorted.index)
plt.xlabel('Mean Contribution to Reward')
plt.title('Reward Component Contributions')
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'reward_components.png'), dpi=150)
print(f"  ✅ Saved: {DIAGNOSTIC_DIR}/reward_components.png")

# Plot 3: HOLD frequency over time
plt.figure(figsize=(12, 6))
window = 100
df_actions['is_hold'] = (df_actions['action_final'] == 1).astype(int)
hold_pct_rolling = df_actions['is_hold'].rolling(window=window).mean() * 100
plt.plot(hold_pct_rolling.index, hold_pct_rolling.values)
plt.xlabel('Step')
plt.ylabel('% HOLD Actions (100-step rolling avg)')
plt.title('HOLD Action Frequency Over Time')
plt.grid(True, alpha=0.3)
plt.axhline(50, color='orange', linestyle='--', label='50% threshold')
plt.axhline(90, color='red', linestyle='--', label='90% critical')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(DIAGNOSTIC_DIR, 'action_frequency_over_time.png'), dpi=150)
print(f"  ✅ Saved: {DIAGNOSTIC_DIR}/action_frequency_over_time.png")

plt.close('all')

# === 8. SUMMARY & RECOMMENDATIONS ===
print("\n" + "="*60)
print("💡 SUMMARY & RECOMMENDATIONS")
print("="*60)

issues_found = []
recommendations = []

# Check 1: HOLD reward dominance
if 'HOLD' in action_stats.index:
    hold_mean = action_stats.loc['HOLD', 'mean']
    buy_mean = action_stats.loc['BUY', 'mean'] if 'BUY' in action_stats.index else -999
    sell_mean = action_stats.loc['SELL', 'mean'] if 'SELL' in action_stats.index else -999
    
    if hold_mean > max(buy_mean, sell_mean):
        issues_found.append("CRITICAL: HOLD reward > Trading rewards")
        recommendations.append({
            'severity': 'CRITICAL',
            'issue': 'HOLD is more rewarding than trading',
            'fix': 'Increase idle_penalty 5-10x AND opportunity_reward 5x'
        })

# Check 2: HOLD dominance
if 'HOLD' in action_counts.index:
    hold_pct = action_counts['HOLD'] / total_actions * 100
    if hold_pct > 90:
        issues_found.append(f"HIGH: HOLD action {hold_pct:.1f}% frequency")
        recommendations.append({
            'severity': 'HIGH',
            'issue': 'Agent stuck in HOLD-only policy',
            'fix': 'Increase exploration_final_eps to 0.15 OR try PPO'
        })

# Check 3: Opportunity usage
if buy_opportunities > 0:
    buy_usage = buy_taken / buy_opportunities
    if buy_usage < 0.2:
        issues_found.append(f"MEDIUM: Only {buy_usage*100:.1f}% buy opportunities used")
        recommendations.append({
            'severity': 'MEDIUM',
            'issue': 'Ignoring buy opportunities',
            'fix': 'Increase opportunity_reward 5-10x'
        })

# Check 4: Component imbalance
if base_return_mag > 0 and idle_penalty_mag / base_return_mag < 0.1:
    issues_found.append("MEDIUM: Idle penalty too weak vs base_return")
    recommendations.append({
        'severity': 'MEDIUM',
        'issue': 'Reward component imbalance',
        'fix': 'Increase base_idle_penalty 10-20x'
    })

if len(issues_found) == 0:
    print("\n✅ No critical issues found! Reward structure seems balanced.")
    print("   If agent still passive, consider:")
    print("   - Increasing training timesteps (try 2M)")
    print("   - Trying different RL algorithm (PPO/SAC)")
    print("   - Checking if market data is suitable (trending vs ranging)")
else:
    print(f"\n❌ Found {len(issues_found)} issue(s):\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['severity']}] {rec['issue']}")
        print(f"   FIX: {rec['fix']}\n")

print("\n" + "="*60)
print("✅ DIAGNOSTIC ANALYSIS COMPLETE")
print("="*60)
print(f"\nNext steps:")
print(f"  1. Review the 3 PNG charts in '{DIAGNOSTIC_DIR}/' folder")
print(f"  2. Implement recommended fixes in trading_env.py")
print(f"  3. Re-run: python train_dqn_diagnostic.py")
print(f"  4. Re-analyze to verify improvements\n")