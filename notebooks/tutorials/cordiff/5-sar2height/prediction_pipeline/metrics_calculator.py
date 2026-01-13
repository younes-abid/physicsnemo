"""
SAR2Height Metrics Calculator

Comprehensive evaluation metrics for SAR-to-height prediction models.
Includes regression metrics, diffusion metrics, and comparison analysis.
"""

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SAR2HeightMetrics:
    """
    Calculates comprehensive metrics for SAR2Height prediction evaluation.
    
    Supports:
    - Standard regression metrics (RMSE, MAE, R², etc.)
    - Height-specific metrics (vertical accuracy, elevation errors)
    - Spatial metrics (local correlation, spatial patterns)
    - Statistical significance testing
    - Model comparison metrics
    """
    
    def __init__(self):
        """Initialize SAR2HeightMetrics."""
        self.metrics_cache = {}
        
    def calculate_regression_metrics(self, 
                                   predicted: np.ndarray, 
                                   ground_truth: np.ndarray,
                                   model_name: str = "regression") -> Dict[str, float]:
        """
        Calculate standard regression metrics for height prediction.
        
        Args:
            predicted: Predicted height values
            ground_truth: Ground truth height values
            model_name: Name of the model for caching
            
        Returns:
            Dictionary of regression metrics
        """
        print(f"=== Calculating Regression Metrics for {model_name} ===")
        
        # Ensure arrays are flattened and finite
        pred_flat = predicted.flatten()
        truth_flat = ground_truth.flatten()
        
        # Remove NaN and infinite values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(truth_flat)
        pred_valid = pred_flat[valid_mask]
        truth_valid = truth_flat[valid_mask]
        
        if len(pred_valid) == 0:
            print("⚠ Warning: No valid predictions found")
            return {}
        
        print(f"  - Valid pixels: {len(pred_valid)}/{len(pred_flat)} ({100*len(pred_valid)/len(pred_flat):.1f}%)")
        
        # Basic error metrics
        mae = mean_absolute_error(truth_valid, pred_valid)
        mse = mean_squared_error(truth_valid, pred_valid)
        rmse = np.sqrt(mse)
        
        # Coefficient of determination
        r2 = r2_score(truth_valid, pred_valid)
        
        # Bias and systematic errors
        bias = np.mean(pred_valid - truth_valid)
        relative_bias = bias / np.mean(truth_valid) if np.mean(truth_valid) != 0 else 0
        
        # Correlation coefficient
        correlation = np.corrcoef(pred_valid, truth_valid)[0, 1]
        
        # Residual analysis
        residuals = pred_valid - truth_valid
        residual_std = np.std(residuals)
        
        # Percentage-based metrics
        mape = np.mean(np.abs((truth_valid - pred_valid) / (truth_valid + 1e-8))) * 100
        
        # Height-specific metrics
        vertical_accuracy_1m = np.mean(np.abs(residuals) <= 1.0) * 100  # % within 1m
        vertical_accuracy_2m = np.mean(np.abs(residuals) <= 2.0) * 100  # % within 2m
        vertical_accuracy_5m = np.mean(np.abs(residuals) <= 5.0) * 100  # % within 5m
        
        # Normalized metrics
        normalized_rmse = rmse / (np.max(truth_valid) - np.min(truth_valid))
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'correlation': float(correlation),
            'bias': float(bias),
            'relative_bias_percent': float(relative_bias * 100),
            'residual_std': float(residual_std),
            'mape': float(mape),
            'normalized_rmse': float(normalized_rmse),
            'vertical_accuracy_1m': float(vertical_accuracy_1m),
            'vertical_accuracy_2m': float(vertical_accuracy_2m),
            'vertical_accuracy_5m': float(vertical_accuracy_5m),
            'valid_pixels': int(len(pred_valid)),
            'total_pixels': int(len(pred_flat)),
            'data_completeness': float(len(pred_valid) / len(pred_flat))
        }
        
        # Cache metrics
        self.metrics_cache[model_name] = metrics
        
        print(f"✓ {model_name} metrics calculated:")
        print(f"  - RMSE: {rmse:.3f} m")
        print(f"  - MAE: {mae:.3f} m")
        print(f"  - R²: {r2:.3f}")
        print(f"  - Bias: {bias:.3f} m")
        print(f"  - Vertical Accuracy (1m): {vertical_accuracy_1m:.1f}%")
        
        return metrics
    
    def calculate_height_distribution_metrics(self, 
                                            predicted: np.ndarray, 
                                            ground_truth: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics related to height distribution preservation."""
        
        pred_flat = predicted.flatten()
        truth_flat = ground_truth.flatten()
        
        # Remove invalid values
        valid_mask = np.isfinite(pred_flat) & np.isfinite(truth_flat)
        pred_valid = pred_flat[valid_mask]
        truth_valid = truth_flat[valid_mask]
        
        if len(pred_valid) == 0:
            return {}
        
        # Distribution statistics
        pred_stats = {
            'mean': np.mean(pred_valid),
            'std': np.std(pred_valid),
            'min': np.min(pred_valid),
            'max': np.max(pred_valid),
            'range': np.max(pred_valid) - np.min(pred_valid),
            'percentiles': np.percentile(pred_valid, [5, 25, 50, 75, 95])
        }
        
        truth_stats = {
            'mean': np.mean(truth_valid),
            'std': np.std(truth_valid),
            'min': np.min(truth_valid),
            'max': np.max(truth_valid),
            'range': np.max(truth_valid) - np.min(truth_valid),
            'percentiles': np.percentile(truth_valid, [5, 25, 50, 75, 95])
        }
        
        # Distribution comparison
        ks_statistic, ks_pvalue = stats.ks_2samp(pred_valid, truth_valid)
        
        # Earth mover's distance (Wasserstein distance)
        try:
            wasserstein_dist = stats.wasserstein_distance(pred_valid, truth_valid)
        except:
            wasserstein_dist = np.nan
        
        return {
            'predicted_distribution': pred_stats,
            'ground_truth_distribution': truth_stats,
            'ks_test': {
                'statistic': float(ks_statistic),
                'p_value': float(ks_pvalue)
            },
            'wasserstein_distance': float(wasserstein_dist) if not np.isnan(wasserstein_dist) else None,
            'mean_difference': float(pred_stats['mean'] - truth_stats['mean']),
            'std_difference': float(pred_stats['std'] - truth_stats['std']),
            'range_difference': float(pred_stats['range'] - truth_stats['range'])
        }
    
    def calculate_spatial_metrics(self, 
                                predicted: np.ndarray, 
                                ground_truth: np.ndarray,
                                window_size: int = 32) -> Dict[str, Any]:
        """Calculate spatial correlation and pattern preservation metrics."""
        
        if predicted.shape != ground_truth.shape:
            raise ValueError("Predicted and ground truth arrays must have the same shape")
        
        H, W = predicted.shape
        
        # Local correlation analysis
        correlations = []
        
        for i in range(0, H - window_size, window_size // 2):
            for j in range(0, W - window_size, window_size // 2):
                # Extract window
                pred_window = predicted[i:i+window_size, j:j+window_size].flatten()
                truth_window = ground_truth[i:i+window_size, j:j+window_size].flatten()
                
                # Remove invalid values
                valid_mask = np.isfinite(pred_window) & np.isfinite(truth_window)
                if np.sum(valid_mask) > window_size:  # Require minimum valid pixels
                    pred_valid = pred_window[valid_mask]
                    truth_valid = truth_window[valid_mask]
                    
                    if len(pred_valid) > 1 and np.std(pred_valid) > 0 and np.std(truth_valid) > 0:
                        corr = np.corrcoef(pred_valid, truth_valid)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
        
        # Gradient analysis
        pred_grad_y, pred_grad_x = np.gradient(predicted)
        truth_grad_y, truth_grad_x = np.gradient(ground_truth)
        
        # Flatten gradients and remove invalid values
        pred_grad_mag = np.sqrt(pred_grad_x**2 + pred_grad_y**2).flatten()
        truth_grad_mag = np.sqrt(truth_grad_x**2 + truth_grad_y**2).flatten()
        
        valid_grad_mask = np.isfinite(pred_grad_mag) & np.isfinite(truth_grad_mag)
        
        gradient_correlation = 0.0
        if np.sum(valid_grad_mask) > 0:
            pred_grad_valid = pred_grad_mag[valid_grad_mask]
            truth_grad_valid = truth_grad_mag[valid_grad_mask]
            
            if len(pred_grad_valid) > 1 and np.std(pred_grad_valid) > 0 and np.std(truth_grad_valid) > 0:
                gradient_correlation = np.corrcoef(pred_grad_valid, truth_grad_valid)[0, 1]
                if np.isnan(gradient_correlation):
                    gradient_correlation = 0.0
        
        return {
            'local_correlations': {
                'mean': float(np.mean(correlations)) if correlations else 0.0,
                'std': float(np.std(correlations)) if correlations else 0.0,
                'min': float(np.min(correlations)) if correlations else 0.0,
                'max': float(np.max(correlations)) if correlations else 0.0,
                'n_windows': len(correlations)
            },
            'gradient_correlation': float(gradient_correlation),
            'window_size': window_size
        }
    
    def compare_models(self, 
                      regression_pred: np.ndarray,
                      diffusion_pred: np.ndarray, 
                      ground_truth: np.ndarray) -> Dict[str, Any]:
        """Compare regression and diffusion model performance."""
        
        print("=== Comparing Regression vs Diffusion Models ===")
        
        # Calculate metrics for both models
        reg_metrics = self.calculate_regression_metrics(regression_pred, ground_truth, "regression")
        diff_metrics = self.calculate_regression_metrics(diffusion_pred, ground_truth, "diffusion")
        
        # Model comparison
        improvements = {}
        for metric_key in ['rmse', 'mae', 'r2', 'correlation', 'bias', 'vertical_accuracy_1m']:
            if metric_key in reg_metrics and metric_key in diff_metrics:
                reg_val = reg_metrics[metric_key]
                diff_val = diff_metrics[metric_key]
                
                # For metrics where lower is better (rmse, mae, bias)
                if metric_key in ['rmse', 'mae', 'bias']:
                    improvement = (reg_val - diff_val) / abs(reg_val) * 100 if reg_val != 0 else 0
                # For metrics where higher is better (r2, correlation, accuracy)
                else:
                    improvement = (diff_val - reg_val) / abs(reg_val) * 100 if reg_val != 0 else 0
                
                improvements[metric_key] = {
                    'regression_value': float(reg_val),
                    'diffusion_value': float(diff_val),
                    'improvement_percent': float(improvement),
                    'better_model': 'diffusion' if improvement > 0 else 'regression'
                }
        
        # Statistical significance testing
        reg_residuals = (regression_pred - ground_truth).flatten()
        diff_residuals = (diffusion_pred - ground_truth).flatten()
        
        # Remove invalid values
        valid_mask = (np.isfinite(reg_residuals) & np.isfinite(diff_residuals) & 
                     np.isfinite(ground_truth.flatten()))
        
        if np.sum(valid_mask) > 100:  # Require minimum samples for testing
            reg_residuals_valid = np.abs(reg_residuals[valid_mask])
            diff_residuals_valid = np.abs(diff_residuals[valid_mask])
            
            # Wilcoxon signed-rank test for paired comparison
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(reg_residuals_valid, diff_residuals_valid)
                significance_test = {
                    'test': 'wilcoxon_signed_rank',
                    'statistic': float(wilcoxon_stat),
                    'p_value': float(wilcoxon_p),
                    'significant': wilcoxon_p < 0.05
                }
            except:
                significance_test = {'test': 'failed', 'reason': 'insufficient_data'}
        else:
            significance_test = {'test': 'skipped', 'reason': 'insufficient_samples'}
        
        return {
            'regression_metrics': reg_metrics,
            'diffusion_metrics': diff_metrics,
            'improvements': improvements,
            'significance_test': significance_test,
            'summary': {
                'better_rmse': 'diffusion' if improvements.get('rmse', {}).get('improvement_percent', 0) > 0 else 'regression',
                'better_mae': 'diffusion' if improvements.get('mae', {}).get('improvement_percent', 0) > 0 else 'regression',
                'better_r2': 'diffusion' if improvements.get('r2', {}).get('improvement_percent', 0) > 0 else 'regression'
            }
        }
    
    def calculate_ensemble_metrics(self, 
                                 ensemble_mean: np.ndarray,
                                 ensemble_std: np.ndarray,
                                 ground_truth: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics specific to ensemble predictions."""
        
        # Standard regression metrics for ensemble mean
        ensemble_metrics = self.calculate_regression_metrics(ensemble_mean, ground_truth, "ensemble")
        
        # Uncertainty calibration
        error_map = np.abs(ensemble_mean - ground_truth)
        uncertainty_map = ensemble_std
        
        # Remove invalid values
        valid_mask = (np.isfinite(error_map.flatten()) & 
                     np.isfinite(uncertainty_map.flatten()))
        
        calibration_metrics = {}
        if np.sum(valid_mask) > 100:
            error_valid = error_map.flatten()[valid_mask]
            uncertainty_valid = uncertainty_map.flatten()[valid_mask]
            
            # Uncertainty-error correlation
            uncertainty_correlation = np.corrcoef(uncertainty_valid, error_valid)[0, 1]
            if np.isnan(uncertainty_correlation):
                uncertainty_correlation = 0.0
            
            # Reliability diagram (simplified)
            n_bins = 10
            uncertainty_bins = np.percentile(uncertainty_valid, np.linspace(0, 100, n_bins + 1))
            reliability_data = []
            
            for i in range(n_bins):
                bin_mask = ((uncertainty_valid >= uncertainty_bins[i]) & 
                           (uncertainty_valid < uncertainty_bins[i + 1]))
                
                if np.sum(bin_mask) > 10:  # Minimum samples per bin
                    bin_uncertainty = np.mean(uncertainty_valid[bin_mask])
                    bin_error = np.mean(error_valid[bin_mask])
                    reliability_data.append({
                        'bin_index': i,
                        'mean_uncertainty': float(bin_uncertainty),
                        'mean_error': float(bin_error),
                        'n_samples': int(np.sum(bin_mask))
                    })
            
            calibration_metrics = {
                'uncertainty_error_correlation': float(uncertainty_correlation),
                'reliability_diagram': reliability_data,
                'mean_uncertainty': float(np.mean(uncertainty_valid)),
                'mean_error': float(np.mean(error_valid))
            }
        
        return {
            'ensemble_regression_metrics': ensemble_metrics,
            'calibration_metrics': calibration_metrics,
            'uncertainty_statistics': {
                'mean_uncertainty': float(np.mean(uncertainty_map[np.isfinite(uncertainty_map)])),
                'std_uncertainty': float(np.std(uncertainty_map[np.isfinite(uncertainty_map)])),
                'max_uncertainty': float(np.max(uncertainty_map[np.isfinite(uncertainty_map)])),
                'min_uncertainty': float(np.min(uncertainty_map[np.isfinite(uncertainty_map)]))
            }
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all calculated metrics."""
        return {
            'cached_metrics': self.metrics_cache,
            'available_models': list(self.metrics_cache.keys()),
            'calculation_timestamp': np.datetime64('now').astype(str)
        }