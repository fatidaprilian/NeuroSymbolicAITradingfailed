@echo off
echo ========================================
echo TESTING 9 MODEL (3 ASET x 3 SCENARIO)
echo ========================================

REM === TESTING BTC ===
echo.
echo [1/9] Testing BTC - Baseline...
python run_test.py --symbol=btc --scenario=baseline
echo.
echo [2/9] Testing BTC - Default...
python run_test.py --symbol=btc --scenario=default
echo.
echo [3/9] Testing BTC - Adaptive...
python run_test.py --symbol=btc --scenario=adaptive

REM === TESTING ETH ===
echo.
echo [4/9] Testing ETH - Baseline...
python run_test.py --symbol=eth --scenario=baseline
echo.
echo [5/9] Testing ETH - Default...
python run_test.py --symbol=eth --scenario=default
echo.
echo [6/9] Testing ETH - Adaptive...
python run_test.py --symbol=eth --scenario=adaptive

REM === TESTING XRP ===
echo.
echo [7/9] Testing XRP - Baseline...
python run_test.py --symbol=xrp --scenario=baseline
echo.
echo [8/9] Testing XRP - Default...
python run_test.py --symbol=xrp --scenario=default
echo.
echo [9/9] Testing XRP - Adaptive...
python run_test.py --symbol=xrp --scenario=adaptive

echo.
echo ========================================
echo SELESAI! Hasil ada di folder final_results/
echo ========================================
pause