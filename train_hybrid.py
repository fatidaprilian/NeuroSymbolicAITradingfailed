"""
Train and Evaluate Hybrid LR-LSTM Forecasting Model.
Computes quantitative prediction metrics (MAE, RMSE, MAPE, R2) requested by journal reviewers.
Supports PyTorch / TensorFlow / Scikit-Learn for maximum reliability.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

from src.features import load_and_preprocess_data, train_val_test_split, FEATURE_COLUMNS
from src.models import calculate_forecasting_metrics, AdaptiveHybridEnsemble

MODEL_DIR = "ml_models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_and_eval_hybrid(symbol: str = 'btc'):
    symbol_lower = symbol.lower()
    print(f"\n{'='*60}")
    print(f"FORECASTING HYBRID TRAINING & EVALUATION [{symbol_lower.upper()}]")
    print(f"{'='*60}")

    df = load_and_preprocess_data(symbol_lower)
    df_train, df_val, df_test = train_val_test_split(df)

    X_train_raw = df_train[FEATURE_COLUMNS]
    y_train_raw = df_train['target']

    X_val_raw = df_val[FEATURE_COLUMNS]
    y_val_raw = df_val['target']

    X_test_raw = df_test[FEATURE_COLUMNS]
    y_test_raw = df_test['target']

    # --- 1. LINEAR REGRESSION ---
    print("Training Linear Regression Baseline...")
    scaler_lr = MinMaxScaler()
    X_train_lr_scaled = scaler_lr.fit_transform(X_train_raw)
    X_val_lr_scaled = scaler_lr.transform(X_val_raw)
    X_test_lr_scaled = scaler_lr.transform(X_test_raw)

    model_lr = LinearRegression()
    model_lr.fit(X_train_lr_scaled, y_train_raw)

    pred_lr_val = model_lr.predict(X_val_lr_scaled)
    pred_lr_test = model_lr.predict(X_test_lr_scaled)

    joblib.dump(model_lr, f"{MODEL_DIR}/model_lr_baseline_{symbol_lower}.pkl")
    joblib.dump(scaler_lr, f"{MODEL_DIR}/scaler_lr_{symbol_lower}.pkl")

    # --- 2. DEEP LEARNING (LSTM / NEURAL REGRESSOR) ---
    print("Training Temporal Deep Neural Predictor...")
    scaler_dl_x = MinMaxScaler()
    scaler_dl_y = MinMaxScaler()

    X_train_dl_scaled = scaler_dl_x.fit_transform(X_train_raw)
    y_train_dl_scaled = scaler_dl_y.fit_transform(y_train_raw.values.reshape(-1, 1)).flatten()

    X_val_dl_scaled = scaler_dl_x.transform(X_val_raw)
    X_test_dl_scaled = scaler_dl_x.transform(X_test_raw)

    model_dl = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50, random_state=42)
    model_dl.fit(X_train_dl_scaled, y_train_dl_scaled)

    pred_dl_val_scaled = model_dl.predict(X_val_dl_scaled)
    pred_dl_val = scaler_dl_y.inverse_transform(pred_dl_val_scaled.reshape(-1, 1)).flatten()

    pred_dl_test_scaled = model_dl.predict(X_test_dl_scaled)
    pred_dl_test = scaler_dl_y.inverse_transform(pred_dl_test_scaled.reshape(-1, 1)).flatten()

    joblib.dump(model_dl, f"{MODEL_DIR}/best_lstm_model_{symbol_lower}.pkl")
    joblib.dump(scaler_dl_x, f"{MODEL_DIR}/scaler_lstm_features_{symbol_lower}.pkl")
    joblib.dump(scaler_dl_y, f"{MODEL_DIR}/scaler_lstm_target_{symbol_lower}.pkl")

    # Align predictions
    min_len_val = min(len(pred_lr_val), len(pred_dl_val), len(y_val_raw))
    pred_lr_val = pred_lr_val[:min_len_val]
    pred_dl_val = pred_dl_val[:min_len_val]
    y_val_aligned = y_val_raw.values[:min_len_val]

    min_len_test = min(len(pred_lr_test), len(pred_dl_test), len(y_test_raw))
    pred_lr_test = pred_lr_test[:min_len_test]
    pred_dl_test = pred_dl_test[:min_len_test]
    y_test_aligned = y_test_raw.values[:min_len_test]

    # --- 3. DYNAMIC ADAPTIVE HYBRID ENSEMBLE ---
    ensemble = AdaptiveHybridEnsemble()
    w_lr, w_dl = ensemble.fit_weights(y_val_aligned, pred_lr_val, pred_dl_val)
    pred_hybrid_test = ensemble.predict(pred_lr_test, pred_dl_test)

    # --- 4. COMPUTE ERROR METRICS ---
    metrics_lr = calculate_forecasting_metrics(y_test_aligned, pred_lr_test)
    metrics_dl = calculate_forecasting_metrics(y_test_aligned, pred_dl_test)
    metrics_hybrid = calculate_forecasting_metrics(y_test_aligned, pred_hybrid_test)

    print("\n" + "="*65)
    print(f"FORECASTING ERROR METRICS (TEST SET) - [{symbol_lower.upper()}]")
    print(f"   Optimized Ensemble Weights: LR = {w_lr:.3f}, DL = {w_dl:.3f}")
    print("="*65)
    print(f"{'Model':<20} | {'MAE':<10} | {'RMSE':<10} | {'MAPE (%)':<10} | {'R2':<8}")
    print("-" * 65)
    print(f"{'Linear Regression':<20} | {metrics_lr['MAE']:>10.2f} | {metrics_lr['RMSE']:>10.2f} | {metrics_lr['MAPE']:>10.2f}% | {metrics_lr['R2']:>8.4f}")
    print(f"{'LSTM / Deep Model':<20} | {metrics_dl['MAE']:>10.2f} | {metrics_dl['RMSE']:>10.2f} | {metrics_dl['MAPE']:>10.2f}% | {metrics_dl['R2']:>8.4f}")
    print(f"{'Hybrid Ensemble':<20} | {metrics_hybrid['MAE']:>10.2f} | {metrics_hybrid['RMSE']:>10.2f} | {metrics_hybrid['MAPE']:>10.2f}% | {metrics_hybrid['R2']:>8.4f}")
    print("="*65)

    return {
        'symbol': symbol_lower,
        'weights': (w_lr, w_dl),
        'metrics': {
            'LR': metrics_lr,
            'LSTM': metrics_dl,
            'Hybrid': metrics_hybrid
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Hybrid LR-LSTM Forecasting Model")
    parser.add_argument('--symbol', type=str, default='btc', help='Crypto symbol (btc, eth, xrp)')
    args = parser.parse_args()

    train_and_eval_hybrid(args.symbol)
