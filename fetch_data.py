import pandas as pd
from binance.client import Client
import sqlite3
import time
import argparse # Library baru untuk menerima argumen
import os

# --- Setup Argumen ---
parser = argparse.ArgumentParser(description='Fetch historical crypto data from Binance.')
parser.add_argument('--symbol', 
                    type=str, 
                    default='BTCUSDT', 
                    help='Symbol to fetch (e.g., ETHUSDT, XRPUSDT)')
args = parser.parse_args()

# --- Main Logic ---
symbol = args.symbol.upper() # Pastikan huruf besar
symbol_lower = symbol.lower().replace('usdt', '') # Untuk nama file (btc, eth, xrp)

# Inisialisasi client
client = Client()

print(f"🚀 Mulai mengambil data {symbol} (Timeframe 1h, 3 tahun terakhir)...")
print("Mohon bersabar...")

start_time = time.time()
try:
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "3 years ago UTC")
    end_time = time.time()
except Exception as e:
    print(f"❌ GAGAL mengambil data {symbol}. Error: {e}")
    print("Pastikan simbolnya valid di Binance (e.g., XRPUSDT bukan XRP).")
    exit()

print(f"✅ Data {symbol} berhasil ditarik dalam {end_time - start_time:.2f} detik!")

# Convert ke pandas DataFrame
df = pd.DataFrame(klines, columns=[
    'open_time', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
])

df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]
df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)
df.set_index('timestamp', inplace=True)
df.drop(columns=['open_time'], inplace=True)

print(f"\n📊 Statistik Data ({symbol}):")
print(f"Total baris: {len(df)}")
print(f"Dari tanggal: {df.index.min()}")
print(f"Sampai tanggal: {df.index.max()}")

# --- PENYIMPANAN (Nama file dinamis) ---
csv_filename = f"{symbol_lower}_1h_data.csv"
db_filename = 'trading_data.db' # Kita simpan di 1 DB tapi tabel berbeda

df.to_csv(csv_filename)
print(f"\n💾 Data disimpan ke CSV: {csv_filename}")

conn = sqlite3.connect(db_filename)
# Nama tabel dinamis
df.to_sql(f"{symbol_lower}_1h", conn, if_exists='replace', index=True) 
conn.close()
print(f"💾 Data disimpan ke Database: {db_filename} (Tabel: '{symbol_lower}_1h')")

print(f"\n🎉 Misi Pengumpulan Data {symbol} Selesai!")