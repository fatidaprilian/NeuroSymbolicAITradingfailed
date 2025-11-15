import os
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, APIRouter
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from functools import partial

# Import fungsi
from .ml_utils import load_all_models, get_hybrid_prediction, get_historical_data, run_automated_trading_job
from .binance_utils import get_testnet_balance, get_trade_history

# --- SETUP ENV ---
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)
ALLOWED_ORIGINS = json.loads(os.getenv("ALLOWED_ORIGINS", '["*"]'))

# --- SETUP SCHEDULER ---
scheduler = BackgroundScheduler()

# --- LIFECYCLE ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n🚀 --- SYSTEM STARTUP ---")
    print("🔄 1. Loading All AI Models (BTC, ETH, XRP)...")
    load_all_models()  # Fungsi ini sudah multi-koin

    print("⏰ 2. Starting Schedulers...")

    # Menjalankan job terpisah untuk tiap koin, setiap 1 jam
    if not scheduler.get_job('job_btc'):
        scheduler.add_job(partial(run_automated_trading_job,
                          symbol_lower='btc'), 'interval', minutes=60, id='job_btc')
    if not scheduler.get_job('job_eth'):
        scheduler.add_job(partial(run_automated_trading_job,
                          symbol_lower='eth'), 'interval', minutes=60, id='job_eth')
    if not scheduler.get_job('job_xrp'):
        scheduler.add_job(partial(run_automated_trading_job,
                          symbol_lower='xrp'), 'interval', minutes=60, id='job_xrp')

    if not scheduler.running:
        scheduler.start()

    print("✅ System Ready. All 3 bots running.")
    yield
    print("\n🛑 --- SYSTEM SHUTDOWN ---")
    scheduler.shutdown()
    print("👋 Bye bye!")

# --- APP INIT ---
app = FastAPI(title="Crypto Bot API (Sinta 2 Edition)",
              version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ROUTES V1 ---
api_v1 = APIRouter(prefix="/api/v1", tags=["v1"])


@api_v1.get("/health")
def health_check():
    jobs = [str(j) for j in scheduler.get_jobs()]
    return {"status": "healthy", "scheduler": "running", "jobs_count": len(jobs)}


@api_v1.get("/predict/live")
def predict_live(symbol: str = 'btc'):
    result, error = get_hybrid_prediction(symbol.lower())
    if error:
        raise HTTPException(status_code=503, detail=error)
    return result


@api_v1.get("/market/history")
def market_history(symbol: str = 'btc', hours: int = 24):
    return get_historical_data(symbol.lower(), hours=hours)

# --- PERBAIKAN DI SINI ---


@api_v1.get("/portfolio/testnet")
def get_portfolio(symbol: str = 'btc'):
    # Teruskan argumen symbol_lower ke fungsi utilitas
    balances, error = get_testnet_balance(symbol.lower())
    if error:
        raise HTTPException(status_code=503, detail=f"Binance Error: {error}")
    return balances

# --- PERBAIKAN DI SINI ---


@api_v1.get("/trade/history")
def get_trade_history_endpoint(symbol: str = 'btc', limit: int = 10):
    symbol_upper = f"{symbol.lower().upper()}USDT"
    # Teruskan argumen symbol_upper ke fungsi utilitas
    history, error = get_trade_history(symbol_upper, limit=limit)
    if error:
        raise HTTPException(status_code=503, detail=error)
    return history


app.include_router(api_v1)


@app.get("/", tags=["root"])
def root():
    return {"system": "Sinta 2 Bot API", "docs": "/docs"}
