"""
Binance Public REST API Data Fetcher for BTC, ETH, XRP.
Fetches real 1-hour OHLCV historical price data directly from Binance public API without API keys.
"""

import os
import sys
import time
import csv
import json
import argparse
from datetime import datetime, timezone

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    import urllib.request
    HAS_REQUESTS = False


def fetch_binance_klines_batch(symbol: str, interval: str = "1h", startTime: int = None, limit: int = 1000):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    if startTime is not None:
        url += f"&startTime={startTime}"

    if HAS_REQUESTS:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Binance API HTTP {response.status_code}: {response.text}")
    else:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode('utf-8')
            return json.loads(data)


def fetch_historical_crypto_data(symbol: str = "BTCUSDT", years: float = 2.0):
    symbol_upper = symbol.upper()
    symbol_lower = symbol_upper.replace("USDT", "").lower()

    print(f"Fetching real {symbol_upper} historical data ({years} years, 1-hour candles) from Binance REST API...")

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - int(years * 365 * 24 * 3600 * 1000)

    current_start = start_ms
    all_klines = []

    while current_start < now_ms:
        try:
            klines = fetch_binance_klines_batch(symbol_upper, interval="1h", startTime=current_start, limit=1000)
            if not klines:
                break

            all_klines.extend(klines)

            # Move next start time to last candle close time + 1 ms
            last_close_time = klines[-1][6]
            if last_close_time <= current_start:
                break
            current_start = last_close_time + 1

            time.sleep(0.1)  # Rate limit safety
        except Exception as e:
            print(f"Warning/Error fetching batch starting at {current_start}: {e}")
            break

    if not all_klines:
        print(f"Error: Could not fetch data for {symbol_upper}.")
        return None

    # Deduplicate and sort
    seen = set()
    unique_klines = []
    for k in all_klines:
        t = k[0]
        if t not in seen:
            seen.add(t)
            unique_klines.append(k)

    unique_klines.sort(key=lambda x: x[0])

    # Save to data/ folder
    os.makedirs("data", exist_ok=True)
    csv_filename = os.path.join("data", f"{symbol_lower}_1h_data.csv")

    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        for k in unique_klines:
            dt = datetime.fromtimestamp(k[0] / 1000.0, tz=timezone.utc)
            timestamp_str = dt.strftime('%Y-%m-%d %H:%M:%S')

            writer.writerow([
                timestamp_str,
                f"{float(k[1]):.4f}",
                f"{float(k[2]):.4f}",
                f"{float(k[3]):.4f}",
                f"{float(k[4]):.4f}",
                f"{float(k[5]):.2f}"
            ])

    print(f"Saved dataset: {csv_filename} ({len(unique_klines)} candles from {datetime.fromtimestamp(unique_klines[0][0]/1000.0, tz=timezone.utc).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(unique_klines[-1][0]/1000.0, tz=timezone.utc).strftime('%Y-%m-%d')})")
    return csv_filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical crypto data from Binance public API.")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Symbol to fetch (e.g., BTCUSDT, ETHUSDT, XRPUSDT)')
    parser.add_argument('--years', type=float, default=2.0, help='Years of historical data')
    args = parser.parse_args()

    fetch_historical_crypto_data(args.symbol, args.years)