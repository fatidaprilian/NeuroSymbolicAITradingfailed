from binance.client import Client
import os
# Kita coba import error dengan cara yang lebih aman
try:
    from binance.exceptions import BinanceAPIException as ClientError
except ImportError:
    # Fallback jika nama exception berbeda
    from binance.client import BinanceAPIException as ClientError

from dotenv import load_dotenv
import time

# Load .env dari folder backend
load_dotenv(os.path.join("backend", ".env"))

key = os.getenv("BINANCE_TESTNET_KEY")
secret = os.getenv("BINANCE_TESTNET_SECRET")

print("🔍 Memeriksa Kunci...")
if not key or not secret:
    print("❌ GAGAL: API Key atau Secret tidak terbaca dari .env!")
    exit()

print(f"🔑 Key terbaca: {key[:10]}... (Panjang: {len(key)})")
# Secret jangan diprint terlalu banyak demi keamanan, meski testnet
print(f"🔒 Secret terbaca: {secret[:5]}... (Panjang: {len(secret)})")

print("\n📡 Mencoba menghubungi Binance Testnet...")
try:
    # Coba sinkronisasi waktu
    temp_client = Client()
    server_time = temp_client.get_server_time()
    local_time = int(time.time() * 1000)
    diff = server_time['serverTime'] - local_time
    print(f"⏱️ Selisih waktu lokal vs server: {diff}ms")

    # Tes koneksi akun
    client = Client(key, secret, testnet=True)
    
    # Jika selisih waktu > 1 detik, kita coba pakai data server time
    if abs(diff) > 1000:
         print("⚠️ Waktu tidak sinkron, mencoba auto-sync internal...")
         # python-binance biasanya punya cara sendiri menangani ini jika testnet=True,
         # tapi kita biarkan default dulu.

    account = client.get_account()
    print("\n✅ KONEKSI SUKSES!")
    print("Saldo akun testnet kamu:")
    for asset in account['balances']:
        if float(asset['free']) > 0:
            print(f"- {asset['asset']}: {asset['free']}")

except ClientError as e:
    print(f"\n❌ ERROR DARI BINANCE (Code {e.code}):")
    print(e.message)
    if e.code == -1021:
         print("💡 TIP: Error -1021 = Masalah Timestamp. Cek jam komputermu!")
    elif e.code == -2015 or e.code == -2014:
         print("💡 TIP: Error -2014/-2015 = API Key salah atau formatnya tidak valid.")
except Exception as e:
    # Tangkap error lain yang mungkin bukan dari Binance langsung
    print(f"\n❌ ERROR LAIN:\n{type(e).__name__}: {e}")