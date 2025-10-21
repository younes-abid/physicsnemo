"""
Metrics Calculator for Weather Pipeline - Streamlit Demo

Handles computation of regression and prediction metrics.
Based on the successful prediction pipeline implementation.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Any, Tuple
import pandas as pd


class MetricsCalculator:
    """Calculates various performance metrics for weather predictions."""
    
    def __init__(self):
        """Initialize MetricsCalculator."""
        pass
    
    def compute_regression_metrics(self, 
                                 u10_pred: np.ndarray, 
                                 v10_pred: np.ndarray,
                                 u10_true: np.ndarray, 
                                 v10_true: np.ndarray,
                                 data_manager=None) -> Dict[str, Any]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values
            u10_true: Ground truth U10 values
            v10_true: Ground truth V10 values
            data_manager: DataManager for normalization (optional)
            
        Returns:
            Dictionary containing all computed metrics
        """
        
        # Flatten arrays for sklearn metrics
        u10_pred_flat = u10_pred.flatten()
        v10_pred_flat = v10_pred.flatten()
        u10_true_flat = u10_true.flatten()
        v10_true_flat = v10_true.flatten()
        
        # Compute U10 metrics
        u10_metrics = self._compute_component_metrics(u10_pred_flat, u10_true_flat, "U10")
        
        # Compute V10 metrics
        v10_metrics = self._compute_component_metrics(v10_pred_flat, v10_true_flat, "V10")
        
        # Compute overall metrics
        overall_mae = (u10_metrics["mae"] + v10_metrics["mae"]) / 2
        overall_rmse = np.sqrt((u10_metrics["mse"] + v10_metrics["mse"]) / 2)
        overall_r2 = (u10_metrics["r2"] + v10_metrics["r2"]) / 2
        
        # Combine all metrics
        metrics = {
            "u10": u10_metrics,
            "v10": v10_metrics,
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "overall_r2": overall_r2,
        }
        
        return metrics
    
    def _compute_component_metrics(self, pred_flat: np.ndarray, true_flat: np.ndarray, component: str) -> Dict[str, float]:
        """Compute metrics for a single component (U10 or V10)."""
        
        # Basic regression metrics
        mae = mean_absolute_error(true_flat, pred_flat)
        mse = mean_squared_error(true_flat, pred_flat)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_flat, pred_flat)
        
        # Statistical metrics
        pred_mean = np.mean(pred_flat)
        true_mean = np.mean(true_flat)
        pred_std = np.std(pred_flat)
        true_std = np.std(true_flat)
        
        # Correlation
        correlation = np.corrcoef(pred_flat, true_flat)[0, 1]
        
        # Bias and relative errors
        bias = pred_mean - true_mean
        relative_bias = bias / true_mean if true_mean != 0 else np.inf
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "correlation": correlation,
            "bias": bias,
            "relative_bias": relative_bias,
            "pred_mean": pred_mean,
            "true_mean": true_mean,
            "pred_std": pred_std,
            "true_std": true_std,
        }
    
    def format_metrics_for_display(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Format metrics into a nice DataFrame for display."""
        
        data = []
        
        # U10 metrics
        u10_metrics = metrics["u10"]
        data.append({
            "Component": "U10",
            "MAE": f"{u10_metrics['mae']:.4f}",
            "RMSE": f"{u10_metrics['rmse']:.4f}",
            "R²": f"{u10_metrics['r2']:.4f}",
            "Correlation": f"{u10_metrics['correlation']:.4f}",
            "Bias": f"{u10_metrics['bias']:.4f}",
            "Pred Mean": f"{u10_metrics['pred_mean']:.4f}",
            "True Mean": f"{u10_metrics['true_mean']:.4f}",
        })
        
        # V10 metrics
        v10_metrics = metrics["v10"]
        data.append({
            "Component": "V10",
            "MAE": f"{v10_metrics['mae']:.4f}",
            "RMSE": f"{v10_metrics['rmse']:.4f}",
            "R²": f"{v10_metrics['r2']:.4f}",
            "Correlation": f"{v10_metrics['correlation']:.4f}",
            "Bias": f"{v10_metrics['bias']:.4f}",
            "Pred Mean": f"{v10_metrics['pred_mean']:.4f}",
            "True Mean": f"{v10_metrics['true_mean']:.4f}",
        })
        
        # Overall metrics
        data.append({
            "Component": "Overall",
            "MAE": f"{metrics['overall_mae']:.4f}",
            "RMSE": f"{metrics['overall_rmse']:.4f}",
            "R²": f"{metrics['overall_r2']:.4f}",
            "Correlation": "-",
            "Bias": "-",
            "Pred Mean": "-",
            "True Mean": "-",
        })
        
        return pd.DataFrame(data)
    
    def get_metrics_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Get a concise summary of key metrics."""
        return {
            "Overall R²": f"{metrics['overall_r2']:.4f}",
            "Overall RMSE": f"{metrics['overall_rmse']:.4f}",
            "Overall MAE": f"{metrics['overall_mae']:.4f}",
            "U10 R²": f"{metrics['u10']['r2']:.4f}",
            "V10 R²": f"{metrics['v10']['r2']:.4f}",
        }