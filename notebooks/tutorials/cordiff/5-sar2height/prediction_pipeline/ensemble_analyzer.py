"""
SAR2Height Ensemble Analyzer

Handles statistical analysis of ensemble predictions from diffusion models.
Provides uncertainty quantification, statistical summaries, and ensemble metrics.
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


class EnsembleAnalyzer:
    """
    Analyzes ensemble predictions to provide uncertainty quantification and statistics.
    
    Supports:
    - Mean and variance estimation
    - Confidence intervals
    - Percentile analysis
    - Uncertainty maps
    - Ensemble spread metrics
    """
    
    def __init__(self):
        """Initialize EnsembleAnalyzer."""
        self.ensemble_predictions = None
        self.ensemble_stats = None
        self.n_ensemble = 0
        
    def analyze_ensemble(self, 
                        ensemble_predictions: List[np.ndarray],
                        confidence_levels: List[float] = [0.68, 0.95, 0.99]) -> Dict[str, Any]:
        """
        Analyze ensemble predictions and compute comprehensive statistics.
        
        Args:
            ensemble_predictions: List of prediction arrays from different seeds
            confidence_levels: Confidence levels for interval estimation
            
        Returns:
            Dictionary containing ensemble statistics and uncertainty measures
        """
        print("=== Analyzing SAR2Height Ensemble Predictions ===")
        
        if not ensemble_predictions:
            raise ValueError("No ensemble predictions provided")
        
        self.ensemble_predictions = ensemble_predictions
        self.n_ensemble = len(ensemble_predictions)
        
        # Convert to numpy array for easier computation
        ensemble_array = np.stack(ensemble_predictions, axis=0)  # Shape: (n_ensemble, H, W)
        
        print(f"  - Ensemble size: {self.n_ensemble}")
        print(f"  - Prediction shape: {ensemble_array.shape}")
        
        # Basic statistics
        ensemble_mean = np.mean(ensemble_array, axis=0)
        ensemble_std = np.std(ensemble_array, axis=0)
        ensemble_var = np.var(ensemble_array, axis=0)
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        ensemble_percentiles = {}
        for p in percentiles:
            ensemble_percentiles[f'p{p}'] = np.percentile(ensemble_array, p, axis=0)
        
        # Confidence intervals
        confidence_intervals = {}
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            lower_p = (alpha / 2) * 100
            upper_p = (1 - alpha / 2) * 100
            
            confidence_intervals[f'ci_{int(conf_level*100)}'] = {
                'lower': np.percentile(ensemble_array, lower_p, axis=0),
                'upper': np.percentile(ensemble_array, upper_p, axis=0),
                'width': np.percentile(ensemble_array, upper_p, axis=0) - 
                        np.percentile(ensemble_array, lower_p, axis=0)
            }
        
        # Uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(ensemble_array)
        
        # Global statistics
        global_stats = self._compute_global_statistics(ensemble_array)
        
        # Ensemble spread analysis
        spread_analysis = self._analyze_ensemble_spread(ensemble_array)
        
        self.ensemble_stats = {
            'basic_statistics': {
                'mean': ensemble_mean,
                'std': ensemble_std,
                'var': ensemble_var,
                'min': np.min(ensemble_array, axis=0),
                'max': np.max(ensemble_array, axis=0),
                'range': np.max(ensemble_array, axis=0) - np.min(ensemble_array, axis=0)
            },
            'percentiles': ensemble_percentiles,
            'confidence_intervals': confidence_intervals,
            'uncertainty_metrics': uncertainty_metrics,
            'global_statistics': global_stats,
            'spread_analysis': spread_analysis,
            'ensemble_info': {
                'n_members': self.n_ensemble,
                'shape': ensemble_array.shape[1:],
                'confidence_levels': confidence_levels
            }
        }
        
        print(f"✓ Ensemble analysis completed successfully!")
        print(f"  - Mean prediction range: [{ensemble_mean.min():.3f}, {ensemble_mean.max():.3f}]")
        print(f"  - Average uncertainty (std): {ensemble_std.mean():.3f}")
        print(f"  - Max uncertainty (std): {ensemble_std.max():.3f}")
        
        return self.ensemble_stats
    
    def _compute_uncertainty_metrics(self, ensemble_array: np.ndarray) -> Dict[str, Any]:
        """Compute various uncertainty metrics."""
        
        # Coefficient of variation (relative uncertainty)
        mean_pred = np.mean(ensemble_array, axis=0)
        std_pred = np.std(ensemble_array, axis=0)
        cv = np.abs(std_pred / (mean_pred + 1e-8))  # Add small epsilon to avoid division by zero
        
        # Signal-to-noise ratio
        snr = np.abs(mean_pred / (std_pred + 1e-8))
        
        # Prediction interval width (95% by default)
        pi_width = np.percentile(ensemble_array, 97.5, axis=0) - np.percentile(ensemble_array, 2.5, axis=0)
        
        # Relative prediction interval width
        relative_pi_width = pi_width / (np.abs(mean_pred) + 1e-8)
        
        # Inter-quartile range
        iqr = np.percentile(ensemble_array, 75, axis=0) - np.percentile(ensemble_array, 25, axis=0)
        
        return {
            'coefficient_of_variation': cv,
            'signal_to_noise_ratio': snr,
            'prediction_interval_width_95': pi_width,
            'relative_prediction_interval_width': relative_pi_width,
            'interquartile_range': iqr,
            'uncertainty_map': std_pred,  # Main uncertainty measure
        }
    
    def _compute_global_statistics(self, ensemble_array: np.ndarray) -> Dict[str, float]:
        """Compute global ensemble statistics."""
        
        # Flatten for global statistics
        flat_ensemble = ensemble_array.reshape(ensemble_array.shape[0], -1)
        
        # Global mean and std for each ensemble member
        member_means = np.mean(flat_ensemble, axis=1)
        member_stds = np.std(flat_ensemble, axis=1)
        
        # Overall statistics
        overall_mean = np.mean(ensemble_array)
        overall_std = np.std(ensemble_array)
        
        # Ensemble spread
        ensemble_spread = np.std(member_means)
        
        # Ensemble skill metrics
        between_member_variance = np.var(member_means)
        within_member_variance = np.mean(member_stds**2)
        
        return {
            'overall_mean': float(overall_mean),
            'overall_std': float(overall_std),
            'ensemble_spread': float(ensemble_spread),
            'between_member_variance': float(between_member_variance),
            'within_member_variance': float(within_member_variance),
            'member_means_mean': float(np.mean(member_means)),
            'member_means_std': float(np.std(member_means)),
            'member_stds_mean': float(np.mean(member_stds)),
            'member_stds_std': float(np.std(member_stds))
        }
    
    def _analyze_ensemble_spread(self, ensemble_array: np.ndarray) -> Dict[str, Any]:
        """Analyze the spread characteristics of the ensemble."""
        
        # Pairwise correlation between ensemble members
        n_members = ensemble_array.shape[0]
        flat_ensemble = ensemble_array.reshape(n_members, -1)
        
        correlations = []
        for i in range(n_members):
            for j in range(i+1, n_members):
                corr = np.corrcoef(flat_ensemble[i], flat_ensemble[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        # Rank-based analysis
        ranks = np.argsort(ensemble_array, axis=0)
        rank_variance = np.var(ranks, axis=0)
        
        # Ensemble range at each pixel
        pixel_ranges = np.max(ensemble_array, axis=0) - np.min(ensemble_array, axis=0)
        
        return {
            'pairwise_correlations': {
                'mean': float(np.mean(correlations)) if correlations else 0.0,
                'std': float(np.std(correlations)) if correlations else 0.0,
                'min': float(np.min(correlations)) if correlations else 0.0,
                'max': float(np.max(correlations)) if correlations else 0.0,
                'all_correlations': correlations
            },
            'rank_statistics': {
                'mean_rank_variance': float(np.mean(rank_variance)),
                'max_rank_variance': float(np.max(rank_variance)),
                'rank_variance_map': rank_variance
            },
            'range_statistics': {
                'mean_pixel_range': float(np.mean(pixel_ranges)),
                'max_pixel_range': float(np.max(pixel_ranges)),
                'pixel_range_map': pixel_ranges
            }
        }
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get a summary of uncertainty metrics for reporting."""
        if self.ensemble_stats is None:
            raise ValueError("No ensemble analysis available. Call analyze_ensemble() first.")
        
        uncertainty = self.ensemble_stats['uncertainty_metrics']
        global_stats = self.ensemble_stats['global_statistics']
        
        return {
            'average_uncertainty': float(np.mean(uncertainty['uncertainty_map'])),
            'max_uncertainty': float(np.max(uncertainty['uncertainty_map'])),
            'min_uncertainty': float(np.min(uncertainty['uncertainty_map'])),
            'uncertainty_range': float(np.max(uncertainty['uncertainty_map']) - np.min(uncertainty['uncertainty_map'])),
            'mean_coefficient_of_variation': float(np.mean(uncertainty['coefficient_of_variation'])),
            'mean_signal_to_noise_ratio': float(np.mean(uncertainty['signal_to_noise_ratio'])),
            'ensemble_spread': global_stats['ensemble_spread'],
            'n_ensemble_members': self.n_ensemble
        }
    
    def get_confidence_map(self, confidence_level: float = 0.95) -> Dict[str, np.ndarray]:
        """
        Get confidence interval maps for a specific confidence level.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with lower, upper bounds and interval width maps
        """
        if self.ensemble_stats is None:
            raise ValueError("No ensemble analysis available. Call analyze_ensemble() first.")
        
        ci_key = f'ci_{int(confidence_level*100)}'
        if ci_key not in self.ensemble_stats['confidence_intervals']:
            raise ValueError(f"Confidence level {confidence_level} not computed")
        
        return self.ensemble_stats['confidence_intervals'][ci_key]
    
    def get_percentile_map(self, percentile: float) -> np.ndarray:
        """
        Get the map for a specific percentile.
        
        Args:
            percentile: Percentile value (0-100)
            
        Returns:
            Percentile map as numpy array
        """
        if self.ensemble_stats is None:
            raise ValueError("No ensemble analysis available. Call analyze_ensemble() first.")
        
        p_key = f'p{int(percentile)}'
        if p_key not in self.ensemble_stats['percentiles']:
            raise ValueError(f"Percentile {percentile} not computed")
        
        return self.ensemble_stats['percentiles'][p_key]
    
    def compare_with_ground_truth(self, ground_truth: np.ndarray) -> Dict[str, Any]:
        """
        Compare ensemble predictions with ground truth.
        
        Args:
            ground_truth: Ground truth array
            
        Returns:
            Dictionary with comparison metrics
        """
        if self.ensemble_stats is None:
            raise ValueError("No ensemble analysis available. Call analyze_ensemble() first.")
        
        ensemble_mean = self.ensemble_stats['basic_statistics']['mean']
        ensemble_std = self.ensemble_stats['basic_statistics']['std']
        
        # Basic error metrics
        error = ensemble_mean - ground_truth
        mae = np.mean(np.abs(error))
        rmse = np.sqrt(np.mean(error**2))
        mse = np.mean(error**2)
        
        # Bias analysis
        bias = np.mean(error)
        
        # Coverage analysis (what percentage of truth falls within confidence intervals)
        coverage_analysis = {}
        for ci_key, ci_data in self.ensemble_stats['confidence_intervals'].items():
            within_ci = ((ground_truth >= ci_data['lower']) & 
                        (ground_truth <= ci_data['upper']))
            coverage_rate = np.mean(within_ci)
            coverage_analysis[ci_key] = {
                'coverage_rate': float(coverage_rate),
                'expected_rate': float(ci_key.split('_')[1]) / 100.0
            }
        
        # Uncertainty calibration
        uncertainty_map = ensemble_std
        abs_error_map = np.abs(error)
        uncertainty_correlation = np.corrcoef(uncertainty_map.flat, abs_error_map.flat)[0, 1]
        
        return {
            'error_metrics': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mse': float(mse),
                'bias': float(bias),
                'error_std': float(np.std(error))
            },
            'coverage_analysis': coverage_analysis,
            'uncertainty_calibration': {
                'uncertainty_error_correlation': float(uncertainty_correlation) if not np.isnan(uncertainty_correlation) else 0.0
            },
            'error_map': error,
            'absolute_error_map': abs_error_map
        }