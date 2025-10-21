"""
Visualizer for Weather Pipeline - Streamlit Demo

Handles visualization of predictions, ground truth, and residuals.
Based on the successful prediction pipeline implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional
import streamlit as st


class Visualizer:
    """Creates visualizations for weather predictions."""
    
    def __init__(self):
        """Initialize Visualizer."""
        # Set matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_predictions_vs_truth(self, 
                                u10_pred: np.ndarray, 
                                v10_pred: np.ndarray,
                                u10_true: np.ndarray, 
                                v10_true: np.ndarray,
                                title: str = "Regression Results") -> plt.Figure:
        """
        Create a comprehensive visualization showing predictions vs ground truth.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values  
            u10_true: Ground truth U10 values
            v10_true: Ground truth V10 values
            title: Title for the plot
            
        Returns:
            matplotlib Figure object
        """
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # U10 plots
        self._plot_component_comparison(axes[0, :], u10_pred, u10_true, "U10 Wind Component")
        
        # V10 plots  
        self._plot_component_comparison(axes[1, :], v10_pred, v10_true, "V10 Wind Component")
        
        plt.tight_layout()
        return fig
    
    def _plot_component_comparison(self, axes, pred: np.ndarray, true: np.ndarray, title: str):
        """Plot prediction, truth, and residual for a single component."""
        
        # Common colormap settings
        vmin = min(pred.min(), true.min())
        vmax = max(pred.max(), true.max())
        
        # Plot 1: Ground Truth
        im1 = axes[0].imshow(true, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{title} - Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0], shrink=0.8)
        
        # Plot 2: Prediction  
        im2 = axes[1].imshow(pred, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{title} - Prediction')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], shrink=0.8)
        
        # Plot 3: Residual (Prediction - Truth)
        residual = pred - true
        vmin_res, vmax_res = residual.min(), residual.max()
        abs_max = max(abs(vmin_res), abs(vmax_res))
        
        im3 = axes[2].imshow(residual, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max)
        axes[2].set_title(f'{title} - Residual (Pred - Truth)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], shrink=0.8)
        
        # Add statistics as text
        mae = np.mean(np.abs(residual))
        rmse = np.sqrt(np.mean(residual**2))
        r2 = 1 - np.sum(residual**2) / np.sum((true - np.mean(true))**2)
        
        stats_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
        axes[2].text(0.02, 0.98, stats_text, transform=axes[2].transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_scatter_comparison(self, 
                              u10_pred: np.ndarray, 
                              v10_pred: np.ndarray,
                              u10_true: np.ndarray, 
                              v10_true: np.ndarray,
                              title: str = "Scatter Plot Comparison") -> plt.Figure:
        """
        Create scatter plots comparing predictions vs ground truth.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values
            u10_true: Ground truth U10 values  
            v10_true: Ground truth V10 values
            title: Title for the plot
            
        Returns:
            matplotlib Figure object
        """
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # U10 scatter plot
        self._plot_scatter_component(axes[0], u10_pred.flatten(), u10_true.flatten(), "U10")
        
        # V10 scatter plot  
        self._plot_scatter_component(axes[1], v10_pred.flatten(), v10_true.flatten(), "V10")
        
        plt.tight_layout()
        return fig
    
    def _plot_scatter_component(self, ax, pred_flat: np.ndarray, true_flat: np.ndarray, component: str):
        """Create scatter plot for a single component."""
        
        # Create scatter plot with alpha for density visualization
        ax.scatter(true_flat, pred_flat, alpha=0.6, s=1)
        
        # Add perfect prediction line
        min_val = min(true_flat.min(), pred_flat.min())
        max_val = max(true_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Set labels and title
        ax.set_xlabel(f'{component} Ground Truth')
        ax.set_ylabel(f'{component} Prediction')
        ax.set_title(f'{component} Predictions vs Ground Truth')
        ax.legend()
        
        # Add R² score
        r2 = 1 - np.sum((true_flat - pred_flat)**2) / np.sum((true_flat - np.mean(true_flat))**2)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    def plot_histograms(self, 
                       u10_pred: np.ndarray, 
                       v10_pred: np.ndarray,
                       u10_true: np.ndarray, 
                       v10_true: np.ndarray,
                       title: str = "Distribution Comparison") -> plt.Figure:
        """
        Create histograms comparing distributions of predictions and ground truth.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values
            u10_true: Ground truth U10 values
            v10_true: Ground truth V10 values
            title: Title for the plot
            
        Returns:
            matplotlib Figure object  
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # U10 histogram
        axes[0, 0].hist(u10_true.flatten(), bins=50, alpha=0.7, label='Ground Truth', density=True)
        axes[0, 0].hist(u10_pred.flatten(), bins=50, alpha=0.7, label='Prediction', density=True)
        axes[0, 0].set_title('U10 Distribution')
        axes[0, 0].set_xlabel('U10 Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # V10 histogram
        axes[0, 1].hist(v10_true.flatten(), bins=50, alpha=0.7, label='Ground Truth', density=True)
        axes[0, 1].hist(v10_pred.flatten(), bins=50, alpha=0.7, label='Prediction', density=True)
        axes[0, 1].set_title('V10 Distribution')
        axes[0, 1].set_xlabel('V10 Value')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        
        # U10 residual histogram
        u10_residual = u10_pred.flatten() - u10_true.flatten()
        axes[1, 0].hist(u10_residual, bins=50, alpha=0.7, color='red')
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('U10 Residual Distribution')
        axes[1, 0].set_xlabel('Residual (Pred - Truth)')
        axes[1, 0].set_ylabel('Count')
        
        # V10 residual histogram
        v10_residual = v10_pred.flatten() - v10_true.flatten()
        axes[1, 1].hist(v10_residual, bins=50, alpha=0.7, color='red')
        axes[1, 1].axvline(0, color='black', linestyle='--', alpha=0.8)
        axes[1, 1].set_title('V10 Residual Distribution')
        axes[1, 1].set_xlabel('Residual (Pred - Truth)')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        return fig
    
    def create_summary_plot(self, 
                           u10_pred: np.ndarray, 
                           v10_pred: np.ndarray,
                           u10_true: np.ndarray, 
                           v10_true: np.ndarray,
                           metrics: dict,
                           title: str = "Regression Summary") -> plt.Figure:
        """
        Create a comprehensive summary plot with all visualizations.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values
            u10_true: Ground truth U10 values
            v10_true: Ground truth V10 values
            metrics: Dictionary containing computed metrics
            title: Title for the plot
            
        Returns:
            matplotlib Figure object
        """
        
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: U10 comparison
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1]) 
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        self._plot_component_comparison([ax1, ax2, ax3], u10_pred, u10_true, "U10")
        self._plot_scatter_component(ax4, u10_pred.flatten(), u10_true.flatten(), "U10")
        
        # Row 2: V10 comparison
        ax5 = fig.add_subplot(gs[1, 0])
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2]) 
        ax8 = fig.add_subplot(gs[1, 3])
        
        self._plot_component_comparison([ax5, ax6, ax7], v10_pred, v10_true, "V10")
        self._plot_scatter_component(ax8, v10_pred.flatten(), v10_true.flatten(), "V10")
        
        # Row 3: Summary metrics table
        ax9 = fig.add_subplot(gs[2, :])
        ax9.axis('off')
        
        # Create metrics text
        metrics_text = f"""
        PERFORMANCE METRICS SUMMARY
        
        Overall Performance:
        • R² Score: {metrics['overall_r2']:.4f}
        • RMSE: {metrics['overall_rmse']:.4f} 
        • MAE: {metrics['overall_mae']:.4f}
        
        U10 Component:
        • R²: {metrics['u10']['r2']:.4f}
        • RMSE: {metrics['u10']['rmse']:.4f}
        • Correlation: {metrics['u10']['correlation']:.4f}
        
        V10 Component:
        • R²: {metrics['v10']['r2']:.4f} 
        • RMSE: {metrics['v10']['rmse']:.4f}
        • Correlation: {metrics['v10']['correlation']:.4f}
        """
        
        ax9.text(0.1, 0.5, metrics_text, transform=ax9.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        return fig