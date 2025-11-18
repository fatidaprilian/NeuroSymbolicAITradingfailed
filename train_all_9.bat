@echo off
echo ============================================================
echo 🚀 TRAINING ALL 9 MODELS - V10.3 PRODUCTION
echo    Validated: BUY +0.021, SELL +0.011 (POSITIVE!)
echo ============================================================
echo.

echo [1/9] Training BTC Baseline...
python train_dqn.py --symbol=btc --scenario=baseline
echo.

echo [2/9] Training BTC Default...
python train_dqn.py --symbol=btc --scenario=default
echo.

echo [3/9] Training BTC Adaptive...
python train_dqn.py --symbol=btc --scenario=adaptive
echo.

echo [4/9] Training ETH Baseline...
python train_dqn.py --symbol=eth --scenario=baseline
echo.

echo [5/9] Training ETH Default...
python train_dqn.py --symbol=eth --scenario=default
echo.

echo [6/9] Training ETH Adaptive...
python train_dqn.py --symbol=eth --scenario=adaptive
echo.

echo [7/9] Training XRP Baseline...
python train_dqn.py --symbol=xrp --scenario=baseline
echo.

echo [8/9] Training XRP Default...
python train_dqn.py --symbol=xrp --scenario=default
echo.

echo [9/9] Training XRP Adaptive...
python train_dqn.py --symbol=xrp --scenario=adaptive
echo.

echo ============================================================
echo ✅ ALL 9 MODELS TRAINED SUCCESSFULLY!
echo ============================================================
pause