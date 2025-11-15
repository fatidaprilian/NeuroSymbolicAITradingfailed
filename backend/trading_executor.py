from binance.client import Client
from binance.enums import *
from .binance_utils import client as binance_client

# --- KONFIGURASI SAFETY NET ---
# Kita set 5% sebagai "pasar badai". Jika volatilitas per jam > 5%, jangan trading.
VOLATILITY_THRESHOLD_PERCENT = 5.0


def execute_trade(action, market_state):
    """
    Eksekutor trading dengan 3 Lapis Safety Net:
    1. Trend (SMA)
    2. Momentum (RSI)
    3. Volatilitas (NATR)
    """
    if not binance_client:
        print("❌ TRADING ERROR: Client Binance belum siap.")
        return False

    # Unpack data pasar
    symbol_upper = market_state['symbol_upper']
    symbol_lower = market_state['symbol_lower']
    current_price = market_state['price']
    sma_30 = market_state['sma_30']
    rsi = market_state['rsi']
    natr = market_state['natr']  # Normalized ATR (%)

    print(
        f"🛡️ SAFETY CHECK ({symbol_upper}): Price=${current_price:.2f} | SMA30=${sma_30:.2f} | RSI={rsi:.2f} | NATR={natr:.2f}%")

    try:
        # --- 1. CEK SALDO SPESIFIK ---
        account = binance_client.get_account()
        usdt_balance = float(
            next(b['free'] for b in account['balances'] if b['asset'] == 'USDT'))
        coin_balance = float(next(
            b['free'] for b in account['balances'] if b['asset'] == symbol_lower.upper()))

        # --- 2. SAFETY NET RULES ---

        # RULE A: Trend Filter
        is_downtrend = current_price < sma_30

        # RULE B: Overbought Filter
        is_overbought = rsi > 70

        # RULE C: Volatility Filter (BARU!)
        # Jika NATR > 5%, pasar terlalu "berisik" (noisy/badai)
        is_too_volatile = natr > VOLATILITY_THRESHOLD_PERCENT

        # --- 3. BLOKIR SEMUA TINDAKAN JIKA TERLALU VOLATIL ---
        if is_too_volatile:
            print(
                f"✋ SAFETY NET: Aksi DIBLOKIR! Volatilitas terlalu tinggi (NATR {natr:.2f}% > {VOLATILITY_THRESHOLD_PERCENT}%)")
            return False

        # --- 4. EKSEKUSI JIKA LOLOS FILTER VOLATILITAS ---
        if action == 2:  # === AI MINTA BUY ===
            if is_downtrend:
                print(
                    "✋ SAFETY NET: Sinyal BUY diblokir! Pasar sedang DOWNTREND (Harga < SMA30).")
                return False
            if is_overbought:
                print(
                    "✋ SAFETY NET: Sinyal BUY diblokir! Pasar sedang OVERBOUGHT (RSI > 70).")
                return False

            if usdt_balance < 10:
                print(
                    f"⚠️ BUY Signal diabaikan: Saldo USDT tidak cukup (${usdt_balance:.2f}).")
                return False

            usdt_to_spend = usdt_balance * 0.99
            # Ambil presisi quantity dari exchange info (lebih robust)
            # Untuk sekarang kita hardcode 5 desimal untuk BTC/ETH/XRP
            quantity = float("{:0.05f}".format(usdt_to_spend / current_price))
            if quantity <= 0:
                return False

            print(
                f"🟢 EXECUTION PASSED: Mengirim order BUY {quantity} {symbol_upper}...")
            # UNCOMMENT UNTUK LIVE TESTNET
            # order = binance_client.create_order(symbol=symbol_upper, side=SIDE_BUY, type=ORDER_TYPE_MARKET, quantity=quantity)
            # print("✅ BUY ORDER SUCCESS! Order ID:", order['orderId'])
            return True

        elif action == 0:  # === AI MINTA SELL ===
            min_qty_map = {'btc': 0.0001,
                           'eth': 0.001, 'xrp': 1}  # Minimal jual
            min_qty = min_qty_map.get(symbol_lower, 0.001)

            if coin_balance < min_qty:
                print(
                    f"⚠️ SELL Signal diabaikan: Saldo {symbol_upper} tidak cukup ({coin_balance}).")
                return False

            quantity = float("{:0.05f}".format(coin_balance * 0.999))
            if quantity <= 0:
                return False

            print(
                f"🔴 EXECUTION PASSED: Mengirim order SELL {quantity} {symbol_upper}...")
            # UNCOMMENT UNTUK LIVE TESTNET
            # order = binance_client.create_order(symbol=symbol_upper, side=SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity)
            # print("✅ SELL ORDER SUCCESS! Order ID:", order['orderId'])
            return True

        else:  # === AI MINTA HOLD ===
            print("⏸️ ACTION: HOLD (AI memutuskan untuk menunggu)")
            return True

    except Exception as e:
        print(f"🔥 EXECUTION FAILED ({symbol_upper}): {e}")
        return False
