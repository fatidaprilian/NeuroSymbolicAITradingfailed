import os
from binance.client import Client
from dotenv import load_dotenv
from pathlib import Path

# --- CARA LEBIH KUAT UNTUK LOAD .ENV ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

api_key = os.getenv("BINANCE_TESTNET_KEY")
api_secret = os.getenv("BINANCE_TESTNET_SECRET")

# Inisialisasi client
try:
    if not api_key or not api_secret:
        print("❌ GAGAL inisialisasi Client Binance: Key kosong.")
        client = None
    else:
        client = Client(api_key, api_secret, testnet=True)
        client.get_server_time() # Tes koneksi ringan
        print("✅ Client Binance Testnet siap.")
except Exception as e:
    print(f"❌ GAGAL inisialisasi Client Binance. Error: {e}")
    client = None

# --- FUNGSI REVISI (menerima argumen) ---
def get_testnet_balance(symbol_lower='btc'):
    """Mengambil saldo USDT dan koin spesifik (btc, eth, xrp)."""
    if not client:
        return None, "Koneksi Binance Client belum siap."
        
    try:
        account_info = client.get_account()
        balances = account_info.get('balances', [])
        
        usdt_balance = 0.0
        coin_balance = 0.0
        coin_upper = symbol_lower.upper() # BTC, ETH, XRP
        
        for asset in balances:
            if asset['asset'] == 'USDT':
                usdt_balance = float(asset['free'])
            elif asset['asset'] == coin_upper:
                coin_balance = float(asset['free'])
                
        return {
            "usdt": usdt_balance,
            symbol_lower: coin_balance # Mengembalikan 'btc': 1.0, 'eth': 10, dsb
        }, None
        
    except Exception as e:
        print(f"🔥 Error saat panggil Binance API (Balance): {e}")
        return None, str(e)

# --- FUNGSI REVISI (menerima argumen) ---
def get_trade_history(symbol_upper="BTCUSDT", limit=20):
    """Mengambil riwayat order terakhir yang SUKSES (FILLED) untuk koin spesifik."""
    if not client:
        return None, "Binance Client belum siap."
    
    try:
        orders = client.get_all_orders(symbol=symbol_upper, limit=limit*2)
        formatted_orders = []
        
        for order in reversed(orders):
            if order['status'] == 'FILLED':
                executed_qty = float(order['executedQty'])
                cummulative_quote_qty = float(order['cummulativeQuoteQty'])
                avg_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else 0

                formatted_orders.append({
                    "id": order['orderId'],
                    "time": int(order['time']),
                    "side": order['side'],
                    "price": avg_price,
                    "qty": executed_qty,
                    "total_usdt": cummulative_quote_qty
                })
                if len(formatted_orders) >= limit: break
                
        return formatted_orders, None
        
    except Exception as e:
        print(f"🔥 Error ambil history {symbol_upper}: {e}")
        return None, str(e)