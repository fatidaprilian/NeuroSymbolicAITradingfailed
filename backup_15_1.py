"""
BACKUP SCRIPT V15.1 → V15.1_backup
Backup semua hasil V15.1 sebelum training V15.2
"""

import os
import shutil
from datetime import datetime


def backup_v15_1():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = f"V15.1_backup_{timestamp}"

    print("="*60)
    print("🔄 BACKING UP V15.1 RESULTS")
    print("="*60)

    # Directories to backup
    backup_targets = {
        'ml_models': ['models', 'test_data'],
        'final_results': ['charts', 'reports'],
        'logs': ['training_logs'],
        'diagnostics': ['diagnostic_logs']
    }

    os.makedirs(backup_root, exist_ok=True)

    for source_dir, description in backup_targets.items():
        if os.path.exists(source_dir):
            dest_dir = os.path.join(backup_root, source_dir)
            print(f"\n📂 Backing up {source_dir}/ → {dest_dir}/")

            try:
                shutil.copytree(source_dir, dest_dir)

                # Count files
                file_count = sum([len(files)
                                 for _, _, files in os.walk(dest_dir)])
                print(f"   ✅ {file_count} files backed up")

            except Exception as e:
                print(f"   ⚠️ Warning: {e}")
        else:
            print(f"\n⚠️ {source_dir}/ not found, skipping...")

    # Backup key Python files
    print(f"\n📄 Backing up Python scripts...")
    scripts_to_backup = [
        'trading_env.py',
        'trading_env_diagnostic.py',
        'train_dqn.py',
        'train_dqn_diagnostic.py',
        'run_test.py'
    ]

    scripts_dir = os.path.join(backup_root, 'scripts_v15.1')
    os.makedirs(scripts_dir, exist_ok=True)

    for script in scripts_to_backup:
        if os.path.exists(script):
            shutil.copy2(script, scripts_dir)
            print(f"   ✅ {script}")

    # Create backup info file
    info_file = os.path.join(backup_root, 'BACKUP_INFO.txt')
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("V15.1 BACKUP INFORMATION\n")
        f.write("="*60 + "\n\n")
        f.write(
            f"Backup Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Version: V15.1 (Financial-Focused + Light Boost)\n\n")

        f.write("KEY CHARACTERISTICS V15.1:\n")
        f.write("- Pure financial reward (log-return + realized PnL)\n")
        f.write("- Idle penalty: 0.007 base, 1.02 growth, 0.14 max\n")
        f.write("- Hyper-trading penalty: 0.0012 (BTC/ETH), 0.0009 (XRP)\n")
        f.write("- Action bonus: 0.03\n")
        f.write("- Safety net penalty: -0.006\n\n")

        f.write("KNOWN ISSUES:\n")
        f.write("- Severe overtrading in BTC/ETH (934-4832 trades)\n")
        f.write("- ETH Adaptive: -99.17% return\n")
        f.write("- BTC models: -57% to -64% return\n")
        f.write("- Safety net ineffective against overtrading\n\n")

        f.write("REASON FOR V15.2:\n")
        f.write("- Strengthen hyper-trading penalties\n")
        f.write("- Add hard trade count limits\n")
        f.write("- Soften idle penalties\n")
        f.write("- Increase minimum trade gaps\n")

    print(f"\n✅ BACKUP COMPLETE!")
    print(f"📁 Location: {backup_root}/")
    print(f"💾 Info file: {info_file}")
    print("="*60)

    return backup_root


if __name__ == "__main__":
    backup_dir = backup_v15_1()
    print(f"\n🎯 Ready to proceed with V15.2 development")
    print(f"📌 V15.1 safely backed up to: {backup_dir}")
