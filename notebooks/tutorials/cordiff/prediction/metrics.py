"""
Metrics Calculator for CorrDiff Prediction Pipeline

Handles computation of various performance metrics for model evaluation.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Any


class MetricsCalculator:
    """Calculates various metrics for evaluating model performance."""
    
    def __init__(self):
        """Initialize MetricsCalculator."""
        pass
    
    def compute_basic_metrics(self, predictions: np.ndarray, targets: np.ndarray, 
                            variable_name: str = "") -> Dict[str, float]:
        """
        Compute basic regression metrics.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            variable_name: Name of the variable (for labeling)
            
        Returns:
            Dictionary containing computed metrics
        """
        # Ensure inputs are numpy arrays
        pred = np.array(predictions).flatten()
        true = np.array(targets).flatten()
        
        # Remove NaN values if any
        mask = ~(np.isnan(pred) | np.isnan(true))
        pred = pred[mask]
        true = true[mask]
        
        if len(pred) == 0:
            return {f"mae_{variable_name}": np.nan, f"rmse_{variable_name}": np.nan, f"r2_{variable_name}": np.nan}
        
        # Basic metrics
        mae = np.mean(np.abs(pred - true))
        mse = np.mean((pred - true) ** 2)
        rmse = np.sqrt(mse)
        
        # R² computation
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        ss_res = np.sum((true - pred) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Mean and std comparisons
        pred_mean = np.mean(pred)
        true_mean = np.mean(true)
        pred_std = np.std(pred)
        true_std = np.std(true)
        
        metrics = {
            f"mae_{variable_name}": mae,
            f"mse_{variable_name}": mse,
            f"rmse_{variable_name}": rmse,
            f"r2_{variable_name}": r2,
            f"pred_mean_{variable_name}": pred_mean,
            f"true_mean_{variable_name}": true_mean,
            f"pred_std_{variable_name}": pred_std,
            f"true_std_{variable_name}": true_std,
            f"mean_bias_{variable_name}": pred_mean - true_mean,
            f"std_bias_{variable_name}": pred_std - true_std,
        }
        
        return metrics
    
    def compute_metrics(self, u10_pred: np.ndarray, v10_pred: np.ndarray,
                                 u10_true: np.ndarray, v10_true: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive metrics for regression predictions.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values
            u10_true: Ground truth U10 values
            v10_true: Ground truth V10 values
            
        Returns:
            Dictionary containing all computed metrics
        """
        print("=== Computing Regression Performance Metrics ===")
        
        # Compute metrics for each component
        u10_metrics = self.compute_basic_metrics(u10_pred, u10_true, "U10")
        v10_metrics = self.compute_basic_metrics(v10_pred, v10_true, "V10")
        
        # Combine metrics
        metrics = {**u10_metrics, **v10_metrics}
        
        # Compute overall metrics
        metrics["overall_mae"] = (metrics["mae_U10"] + metrics["mae_V10"]) / 2
        metrics["overall_rmse"] = (metrics["rmse_U10"] + metrics["rmse_V10"]) / 2
        metrics["overall_r2"] = (metrics["r2_U10"] + metrics["r2_V10"]) / 2
        
        # Print formatted results
        print(f"U10 (Wind Component):")
        print(f"  - MAE: {metrics['mae_U10']:.4f}")
        print(f"  - RMSE: {metrics['rmse_U10']:.4f}")
        print(f"  - R²: {metrics['r2_U10']:.4f}")
        print(f"  - Pred mean: {metrics['pred_mean_U10']:.4f}, True mean: {metrics['true_mean_U10']:.4f}")
        print(f"  - Pred std: {metrics['pred_std_U10']:.4f}, True std: {metrics['true_std_U10']:.4f}")
        
        print(f"\nV10 (Wind Component):")
        print(f"  - MAE: {metrics['mae_V10']:.4f}")
        print(f"  - RMSE: {metrics['rmse_V10']:.4f}")
        print(f"  - R²: {metrics['r2_V10']:.4f}")
        print(f"  - Pred mean: {metrics['pred_mean_V10']:.4f}, True mean: {metrics['true_mean_V10']:.4f}")
        print(f"  - Pred std: {metrics['pred_std_V10']:.4f}, True std: {metrics['true_std_V10']:.4f}")
        
        print(f"\nOverall Performance:")
        print(f"  - Average MAE: {metrics['overall_mae']:.4f}")
        print(f"  - Average RMSE: {metrics['overall_rmse']:.4f}")
        print(f"  - Average R²: {metrics['overall_r2']:.4f}")
        
        return metrics
    
    def compute_comparison_metrics(self, regression_pred: Tuple[np.ndarray, np.ndarray],
                                 diffusion_pred: Tuple[np.ndarray, np.ndarray],
                                 ground_truth: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """
        Compare regression vs diffusion predictions.
        
        Args:
            regression_pred: Tuple of (u10_reg, v10_reg)
            diffusion_pred: Tuple of (u10_diff, v10_diff)
            ground_truth: Tuple of (u10_true, v10_true)
            
        Returns:
            Dictionary containing comparison metrics
        """
        u10_reg, v10_reg = regression_pred
        u10_diff, v10_diff = diffusion_pred
        u10_true, v10_true = ground_truth
        
        print("=== Computing Comparison Metrics ===")
        
        # Compute metrics for regression
        reg_metrics = self.compute_metrics(u10_reg, v10_reg, u10_true, v10_true)
        
        # Compute metrics for diffusion
        diff_metrics = self.compute_metrics(u10_diff, v10_diff, u10_true, v10_true)
        
        # Create comparison dictionary
        comparison = {
            "regression": {
                "u10_r2": reg_metrics["r2_U10"],
                "v10_r2": reg_metrics["r2_V10"],
                "overall_r2": reg_metrics["overall_r2"],
                "u10_rmse": reg_metrics["rmse_U10"],
                "v10_rmse": reg_metrics["rmse_V10"],
                "overall_rmse": reg_metrics["overall_rmse"],
            },
            "diffusion": {
                "u10_r2": diff_metrics["r2_U10"],
                "v10_r2": diff_metrics["r2_V10"],
                "overall_r2": diff_metrics["overall_r2"],
                "u10_rmse": diff_metrics["rmse_U10"],
                "v10_rmse": diff_metrics["rmse_V10"],
                "overall_rmse": diff_metrics["overall_rmse"],
            }
        }
        
        # Compute improvements
        r2_improvement = diff_metrics["overall_r2"] - reg_metrics["overall_r2"]
        rmse_improvement = reg_metrics["overall_rmse"] - diff_metrics["overall_rmse"]  # Lower is better
        
        comparison["improvements"] = {
            "r2_improvement": r2_improvement,
            "rmse_improvement": rmse_improvement,
            "r2_improvement_percent": (r2_improvement / reg_metrics["overall_r2"]) * 100 if reg_metrics["overall_r2"] != 0 else 0,
            "rmse_improvement_percent": (rmse_improvement / reg_metrics["overall_rmse"]) * 100 if reg_metrics["overall_rmse"] != 0 else 0,
        }
        
        print(f"\n=== Model Comparison ===")
        print(f"Regression R²: {reg_metrics['overall_r2']:.4f}")
        print(f"Diffusion R²: {diff_metrics['overall_r2']:.4f}")
        print(f"R² Improvement: {r2_improvement:+.4f} ({comparison['improvements']['r2_improvement_percent']:+.1f}%)")
        
        print(f"\nRegression RMSE: {reg_metrics['overall_rmse']:.4f}")
        print(f"Diffusion RMSE: {diff_metrics['overall_rmse']:.4f}")
        print(f"RMSE Improvement: {rmse_improvement:+.4f} ({comparison['improvements']['rmse_improvement_percent']:+.1f}%)")
        
        return comparison
    
    def print_summary_statistics(self, u10_pred: np.ndarray, v10_pred: np.ndarray,
                               u10_true: np.ndarray, v10_true: np.ndarray,
                               title: str = "Prediction") -> None:
        """Print summary statistics for predictions."""
        print(f"\n=== {title} Summary Statistics ===")
        
        print(f"U10 Prediction vs Ground Truth:")
        print(f"  Pred mean: {u10_pred.mean():.4f}, True mean: {u10_true.mean():.4f}")
        print(f"  Pred std: {u10_pred.std():.4f}, True std: {u10_true.std():.4f}")
        print(f"  Pred range: [{u10_pred.min():.4f}, {u10_pred.max():.4f}]")
        print(f"  True range: [{u10_true.min():.4f}, {u10_true.max():.4f}]")
        
        print(f"\nV10 Prediction vs Ground Truth:")
        print(f"  Pred mean: {v10_pred.mean():.4f}, True mean: {v10_true.mean():.4f}")
        print(f"  Pred std: {v10_pred.std():.4f}, True std: {v10_true.std():.4f}")
        print(f"  Pred range: [{v10_pred.min():.4f}, {v10_pred.max():.4f}]")
        print(f"  True range: [{v10_true.min():.4f}, {v10_true.max():.4f}]")
    
    def compute_metrics_with_normalization(self, pred_denorm: np.ndarray, true_raw: np.ndarray,
                                         data_manager, variable_name: str) -> Dict[str, float]:
        """
        Compute metrics correctly by normalizing ground truth to match model output space.
        
        Args:
            pred_denorm: Denormalized predictions (for visualization)
            true_raw: Raw ground truth values
            data_manager: DataManager instance for normalization
            variable_name: Variable name (U10 or V10)
            
        Returns:
            Dictionary containing computed metrics
        """
        # Method 1: Compare in denormalized space (for interpretation)
        denorm_metrics = self.compute_basic_metrics(pred_denorm, true_raw, f"{variable_name}_denorm")
        
        # Method 2: Compare in normalized space (for model evaluation)
        # Normalize ground truth to match model output space
        true_norm = data_manager.normalize_output(true_raw, variable_name)
        pred_norm = data_manager.normalize_output(pred_denorm, variable_name)
        norm_metrics = self.compute_basic_metrics(pred_norm, true_norm, f"{variable_name}_norm")
        
        # Combine both metric sets
        combined_metrics = {**denorm_metrics, **norm_metrics}
        
        return combined_metrics
    
    def compute_corrected_regression_metrics(self, u10_pred_denorm: np.ndarray, v10_pred_denorm: np.ndarray,
                                           u10_true_raw: np.ndarray, v10_true_raw: np.ndarray,
                                           data_manager) -> Dict[str, float]:
        """
        Compute regression metrics correctly by comparing in both normalized and denormalized spaces.
        
        Args:
            u10_pred_denorm: Denormalized U10 predictions
            v10_pred_denorm: Denormalized V10 predictions
            u10_true_raw: Raw U10 ground truth
            v10_true_raw: Raw V10 ground truth
            data_manager: DataManager instance
            
        Returns:
            Dictionary containing all computed metrics
        """
        print("=== Computing CORRECTED Regression Performance Metrics ===")
        print("Note: Models work in normalized space, so proper comparison requires normalization")
        
        # Compute corrected metrics for each component
        u10_metrics = self.compute_metrics_with_normalization(u10_pred_denorm, u10_true_raw, data_manager, "U10")
        v10_metrics = self.compute_metrics_with_normalization(v10_pred_denorm, v10_true_raw, data_manager, "V10")
        
        # Combine metrics
        metrics = {**u10_metrics, **v10_metrics}
        
        # Use normalized space metrics for primary evaluation (these are what the model actually learns)
        primary_u10_r2 = metrics["r2_U10_norm"]
        primary_v10_r2 = metrics["r2_V10_norm"]
        primary_u10_rmse = metrics["rmse_U10_norm"]
        primary_v10_rmse = metrics["rmse_V10_norm"]
        
        # Compute overall metrics
        metrics["overall_mae"] = (metrics["mae_U10_denorm"] + metrics["mae_V10_denorm"]) / 2
        metrics["overall_rmse"] = (metrics["rmse_U10_denorm"] + metrics["rmse_V10_denorm"]) / 2
        metrics["overall_r2"] = (primary_u10_r2 + primary_v10_r2) / 2  # Use normalized space R²
        
        # Print formatted results
        print(f"U10 (Wind Component):")
        print(f"  - MAE (denorm): {metrics['mae_U10_denorm']:.4f}")
        print(f"  - RMSE (denorm): {metrics['rmse_U10_denorm']:.4f}")
        print(f"  - R² (normalized space): {primary_u10_r2:.4f}")
        print(f"  - R² (denorm space): {metrics['r2_U10_denorm']:.4f}")
        print(f"  - Pred mean: {metrics['pred_mean_U10_denorm']:.4f}, True mean: {metrics['true_mean_U10_denorm']:.4f}")
        print(f"  - Pred std: {metrics['pred_std_U10_denorm']:.4f}, True std: {metrics['true_std_U10_denorm']:.4f}")
        
        print(f"\nV10 (Wind Component):")
        print(f"  - MAE (denorm): {metrics['mae_V10_denorm']:.4f}")
        print(f"  - RMSE (denorm): {metrics['rmse_V10_denorm']:.4f}")
        print(f"  - R² (normalized space): {primary_v10_r2:.4f}")
        print(f"  - R² (denorm space): {metrics['r2_V10_denorm']:.4f}")
        print(f"  - Pred mean: {metrics['pred_mean_V10_denorm']:.4f}, True mean: {metrics['true_mean_V10_denorm']:.4f}")
        print(f"  - Pred std: {metrics['pred_std_V10_denorm']:.4f}, True std: {metrics['true_std_V10_denorm']:.4f}")
        
        print(f"\nOverall Performance:")
        print(f"  - Average MAE (denorm): {metrics['overall_mae']:.4f}")
        print(f"  - Average RMSE (denorm): {metrics['overall_rmse']:.4f}")
        print(f"  - Average R² (normalized space): {metrics['overall_r2']:.4f}")
        
        print(f"\n✓ Key insight: R² in normalized space reflects model's actual learning capability")
        
        return metrics