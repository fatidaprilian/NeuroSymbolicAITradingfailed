"""
Pure Python Standard Library Dataset Generator for BTC, ETH, XRP.
Generates realistic 1-hour OHLCV price series without external dependencies.
"""

import csv
import math
import random
from datetime import datetime, timedelta

def generate_csv(symbol: str, initial_price: float, volatility: float, drift: float, seed: int):
    random.seed(seed)
    n_days = 730
    n_steps = n_days * 24
    dt = 1.0 / (365 * 24)

    start_date = datetime(2024, 1, 1, 0, 0, 0)
    current_price = initial_price

    filename = f"{symbol.lower()}_1h_data.csv"
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        for i in range(n_steps):
            timestamp = start_date + timedelta(hours=i)
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')

            # Random shock (Box-Muller transform for normal distribution)
            u1 = random.random()
            u2 = random.random()
            z = math.sqrt(-2.0 * math.log(max(u1, 1e-9))) * math.cos(2.0 * math.pi * u2)

            # Regime volatility clustering
            vol_step = volatility * (1.0 + 0.3 * math.sin(i / 100.0))
            log_ret = (drift - 0.5 * (vol_step**2)) * dt + vol_step * math.sqrt(dt) * z

            # Random jump
            if random.random() < 0.005:
                log_ret += (random.random() - 0.5) * 0.08

            open_price = current_price
            close_price = max(open_price * math.exp(log_ret), 0.01)
            high_price = max(open_price, close_price) * (1.0 + random.random() * 0.006)
            low_price = min(open_price, close_price) * (1.0 - random.random() * 0.006)
            volume = random.uniform(500, 5000) * (1.0 + abs(log_ret) * 10)

            writer.writerow([
                timestamp_str,
                f"{open_price:.4f}",
                f"{high_price:.4f}",
                f"{low_price:.4f}",
                f"{close_price:.4f}",
                f"{volume:.2f}"
            ])

            current_price = close_price

    print(f"✅ Generated dataset: {filename} ({n_steps} rows)")

if __name__ == "__main__":
    generate_csv('btc', 45000.0, 0.65, 0.15, 101)
    generate_csv('eth', 2500.0, 0.80, 0.10, 202)
    generate_csv('xrp', 0.55, 0.95, 0.05, 303)
