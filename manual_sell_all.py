from binance.client import Client
from binance.enums import *
import os
from dotenv import load_dotenv
from pathlib import Path

# --- LOAD ENV ---
# Kita gunakan cara robust yang sama dengan backend agar terbaca
BASE_DIR = Path(__file__).resolve().parent
# Asumsi .env ada di dalam folder backend
ENV_PATH = BASE_DIR / "backend" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("BINANCE_TESTNET_KEY")
api_secret = os.getenv("BINANCE_TESTNET_SECRET")

if not api_key or not api_secret:
    print("❌ Error: API Key/Secret tidak ditemukan. Pastikan file backend/.env ada.")
    exit()

# --- INISIALISASI CLIENT ---
try:
    client = Client(api_key, api_secret, testnet=True)
    print("✅ Terhubung ke Binance Testnet.")
except Exception as e:
    print(f"❌ Gagal koneksi: {e}")
    exit()

# --- PROSES JUAL SEMUA BTC ---
try:
    # 1. Cek Saldo BTC
    btc_balance = float(client.get_asset_balance(asset='BTC')['free'])
    print(f"💰 Saldo BTC saat ini: {btc_balance:.6f} BTC")

    if btc_balance < 0.0001:
        print("⚠️ Saldo BTC terlalu sedikit untuk dijual.")
        exit()

    # 2. Hitung Quantity (Jual 99.9% agar aman dari masalah pembulatan desimal)
    # BTCUSDT di Binance biasanya butuh presisi 5 angka di belakang koma untuk quantity
    quantity = "{:0.05f}".format(btc_balance * 0.999)
    print(f"🔄 Akan menjual: {quantity} BTC ke USDT...")

    # 3. Eksekusi Order Market SELL
    order = client.create_order(
        symbol='BTCUSDT',
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=quantity
    )

    print("\n🎉 SUKSES! BTC berhasil dijual.")
    print(f"🆔 Order ID: {order['orderId']}")

    # 4. Cek Saldo USDT Akhir
    usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
    print(f"💵 Saldo USDT sekarang: ${usdt_balance:.2f}")

except Exception as e:
    print(f"\n🔥 GAGAL MENJUAL: {e}")
