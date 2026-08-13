"""
Prediction Models and Hybrid Ensembling Module for Price Forecasting.
Calculates quantitative error metrics (MAE, RMSE, MAPE, R2) and optimizes hybrid weights.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize


def calculate_forecasting_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes quantitative prediction accuracy metrics requested by journal reviewers:
    - MAE  (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - MAPE (Mean Absolute Percentage Error in %)
    - R2   (Coefficient of Determination)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    r2 = r2_score(y_true, y_pred)

    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'R2': float(r2)
    }


class AdaptiveHybridEnsemble:
    """
    Adaptive Dynamic Weighting Ensemble for Linear Regression and LSTM predictions.
    Replaces static manual weights with validation-loss optimized convex weights:
    P_hybrid = w * P_LR + (1 - w) * P_LSTM
    """

    def __init__(self, initial_w_lr: float = 0.8):
        self.w_lr = initial_w_lr
        self.w_lstm = 1.0 - initial_w_lr

    def fit_weights(self, y_val: np.ndarray, pred_lr_val: np.ndarray, pred_lstm_val: np.ndarray):
        """
        Optimizes weight w in [0, 1] to minimize validation RMSE.
        """
        y_val = np.array(y_val).flatten()
        pred_lr = np.array(pred_lr_val).flatten()
        pred_lstm = np.array(pred_lstm_val).flatten()

        def objective(w):
            w_val = w[0]
            pred = w_val * pred_lr + (1.0 - w_val) * pred_lstm
            return mean_squared_error(y_val, pred)

        res = minimize(objective, [0.5], bounds=[(0.0, 1.0)], method='L-BFGS-B')
        if res.success:
            self.w_lr = float(res.x[0])
            self.w_lstm = 1.0 - self.w_lr

        return self.w_lr, self.w_lstm

    def predict(self, pred_lr: np.ndarray, pred_lstm: np.ndarray) -> np.ndarray:
        """
        Combines predictions using optimized weights.
        """
        return (self.w_lr * np.array(pred_lr)) + (self.w_lstm * np.array(pred_lstm))
