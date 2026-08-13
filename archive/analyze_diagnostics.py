"""
ANALYZE DIAGNOSTICS V15.4

Analyze diagnostic logs and compare behavior (trade count, blocks, rewards).
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def load_episode_diagnostics(episode_dir):
    """Load all diagnostic files for an episode"""
    try:
        with open(os.path.join(episode_dir, 'summary.json'), 'r') as f:
            summary = json.load(f)

        with open(os.path.join(episode_dir, 'reward_components.json'), 'r') as f:
            rewards = json.load(f)

        with open(os.path.join(episode_dir, 'actions.json'), 'r') as f:
            actions = json.load(f)

        return summary, rewards, actions
    except Exception as e:
        print(f"⚠️ Error loading {episode_dir}: {e}")
        return None, None, None


def analyze_reward_components(rewards):
    """Analyze reward component statistics"""
    components = {
        'base_return': [],
        'safety_penalty': [],
        'idle_penalty': [],
        'unrealized_profit': [],
        'trade_profit': [],
        'action_bonus': [],
        'hyper_penalty': [],
        'rapid_penalty': [],
        'hard_limit_penalty': [],
        'patience_reward': []   # NEW V15.3
    }

    total_rewards = []

    for step_data in rewards:
        total_rewards.append(step_data['total_reward'])
        for key in components.keys():
            if key in step_data['components']:
                components[key].append(step_data['components'][key])

    stats = {
        'total_reward': {
            'mean': np.mean(total_rewards),
            'sum': np.sum(total_rewards),
            'std': np.std(total_rewards)
        }
    }

    for key, values in components.items():
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'sum': np.sum(values),
                'count_nonzero': np.count_nonzero(values)
            }

    return stats


def plot_episode_analysis(summary, rewards, actions, output_dir):
    """Create visualization plots"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"V15.3 Episode {summary['episode']} Analysis", fontsize=16)

    # 1. Total reward over time
    total_rewards = [r['total_reward'] for r in rewards]
    axes[0, 0].plot(total_rewards, linewidth=0.5, alpha=0.7)
    axes[0, 0].set_title('Total Reward per Step')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Cumulative reward
    cumulative = np.cumsum(total_rewards)
    axes[0, 1].plot(cumulative, linewidth=1)
    axes[0, 1].set_title('Cumulative Reward')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Cumulative Reward')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Reward components breakdown
    component_sums = {}
    for key in ['base_return', 'idle_penalty', 'trade_profit', 'action_bonus',
                'hyper_penalty', 'safety_penalty', 'rapid_penalty',
                'hard_limit_penalty', 'patience_reward']:
        values = [r['components'].get(key, 0) for r in rewards]
        component_sums[key] = np.sum(values)

    axes[1, 0].bar(range(len(component_sums)), list(component_sums.values()))
    axes[1, 0].set_xticks(range(len(component_sums)))
    axes[1, 0].set_xticklabels(
        list(component_sums.keys()), rotation=45, ha='right')
    axes[1, 0].set_title('Reward Component Totals')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Action distribution
    action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
    blocked_count = 0

    for action in actions:
        action_counts[action['action_executed']] += 1
        if action['blocked']:
            blocked_count += 1

    axes[1, 1].bar(action_counts.keys(), action_counts.values())
    axes[1, 1].set_title(f'Action Distribution (Blocked: {blocked_count})')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)

    # 5. Trade gaps histogram
    trade_gaps = [a['trade_gap']
                  for a in actions if a['trade_gap'] is not None]
    if trade_gaps:
        axes[2, 0].hist(trade_gaps, bins=50, edgecolor='black', alpha=0.7)
        axes[2, 0].axvline(x=summary.get('max_trades_limit', 0) / 10,
                           color='r', linestyle='--', label='Target avg gap')
        axes[2, 0].set_title('Trade Gap Distribution')
        axes[2, 0].set_xlabel('Steps between trades')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # 6. Key metrics summary
    metrics_text = f"""
    Episode: {summary['episode']}
    Version: {summary.get('version', 'V15.3')}
    
    Total Trades: {summary['total_trades']} / {summary.get('max_trades_limit', 'N/A')}
    Final Return: {summary['return_pct']:.2f}%
    
    Safety Triggers:
    - Hyper Trading: {summary['safety_triggers'].get('hyper_trading_penalized', 0)}
    - Rapid Trading: {summary['safety_triggers'].get('rapid_trading_blocked', 0)}
    - Hard Limit: {summary['safety_triggers'].get('hard_limit_reached', 0)}
    - Total Blocks: {summary['safety_triggers'].get('total_blocks', 0)}
    """

    axes[2, 1].text(0.1, 0.5, metrics_text, fontsize=10,
                    verticalalignment='center', family='monospace')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(
        output_dir, f"episode_{summary['episode']}_analysis.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze V15.3 Diagnostics')
    parser.add_argument('--path', type=str, default='diagnostics_v15_3',
                        help='Path to diagnostics directory')
    args = parser.parse_args()

    diag_path = args.path

    if not os.path.exists(diag_path):
        print(f"❌ Diagnostics path not found: {diag_path}")
        return

    print(f"\n{'='*70}")
    print(f"🔍 ANALYZING V15.3 DIAGNOSTICS")
    print(f"{'='*70}\n")
    print(f"📁 Source: {diag_path}/")

    # Find all episodes
    episode_dirs = sorted(glob.glob(os.path.join(diag_path, 'episode_*')))

    if not episode_dirs:
        print(f"⚠️ No episodes found in {diag_path}")
        return

    print(f"📊 Found {len(episode_dirs)} episodes\n")

    # Create output directory
    output_dir = os.path.join(diag_path, 'analysis')
    os.makedirs(output_dir, exist_ok=True)

    # Analyze each episode
    all_summaries = []

    for ep_dir in episode_dirs:
        summary, rewards, actions = load_episode_diagnostics(ep_dir)

        if summary is None:
            continue

        print(f"Episode {summary['episode']}:")
        print(
            f"   Trades: {summary['total_trades']} / {summary.get('max_trades_limit', 'N/A')}")
        print(f"   Return: {summary['return_pct']:+.2f}%")
        print(
            f"   Blocks: {summary['safety_triggers'].get('total_blocks', 0)}")

        # Analyze rewards
        reward_stats = analyze_reward_components(rewards)
        print(f"   Avg Reward: {reward_stats['total_reward']['mean']:.6f}")

        # Generate plots
        plot_episode_analysis(summary, rewards, actions, output_dir)

        all_summaries.append(summary)
        print()

    # Aggregate statistics
    print(f"{'='*70}")
    print(f"📈 AGGREGATE STATISTICS")
    print(f"{'='*70}\n")

    total_trades = [s['total_trades'] for s in all_summaries]
    returns = [s['return_pct'] for s in all_summaries]

    print(f"Trade Count Statistics:")
    print(f"   Mean: {np.mean(total_trades):.1f}")
    print(f"   Median: {np.median(total_trades):.1f}")
    print(f"   Min: {np.min(total_trades)}")
    print(f"   Max: {np.max(total_trades)}")

    print(f"\nReturn Statistics:")
    print(f"   Mean: {np.mean(returns):+.2f}%")
    print(f"   Median: {np.median(returns):+.2f}%")
    print(f"   Min: {np.min(returns):+.2f}%")
    print(f"   Max: {np.max(returns):+.2f}%")

    # Compare with expected targets
    print(f"\n{'='*70}")
    print(f"🎯 V15.3 TARGET ASSESSMENT")
    print(f"{'='*70}\n")

    avg_trades = np.mean(total_trades)
    avg_return = np.mean(returns)

    print(f"Trade Count Control:")
    if 50 <= avg_trades <= 150:
        print(f"   ✅ GOOD: {avg_trades:.1f} trades (target: 50-150)")
    else:
        print(f"   ⚠️ OFF TARGET: {avg_trades:.1f} trades (target: 50-150)")

    print(f"\nReturn Performance:")
    if avg_return > 0:
        print(f"   ✅ POSITIVE: {avg_return:+.2f}% average return")
    else:
        print(f"   ❌ NEGATIVE: {avg_return:+.2f}% average return")

    print(f"\n📁 Analysis plots saved to: {output_dir}/")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
