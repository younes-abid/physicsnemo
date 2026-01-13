"""
SAR2Height Visualizer

Advanced visualization and plotting for SAR-to-height prediction results.
Supports comprehensive visualization of inputs, outputs, comparisons, and ensemble analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class SAR2HeightVisualizer:
    """
    Creates comprehensive visualizations for SAR2Height prediction pipeline.
    
    Supports:
    - Input data visualization (all SAR channels)
    - Ground truth vs predictions comparison
    - Residual analysis plots
    - Ensemble uncertainty visualization
    - Statistical analysis plots
    - Model comparison visualizations
    """
    
    def __init__(self, figsize_base: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize SAR2HeightVisualizer.
        
        Args:
            figsize_base: Base figure size for single plots
            dpi: DPI for high-quality plots
        """
        self.figsize_base = figsize_base
        self.dpi = dpi
        self.color_maps = {
            'height': 'terrain',
            'sar': 'viridis',
            'error': 'RdBu_r',
            'uncertainty': 'plasma'
        }
        
    def plot_input_channels(self, 
                          input_data_dict: Dict[str, np.ndarray],
                          title: str = "SAR Input Channels",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot all SAR input channels in a grid layout.
        
        Args:
            input_data_dict: Dictionary with variable names as keys and 2D arrays as values
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_channels = len(input_data_dict)
        channels = list(input_data_dict.keys())
        
        # Calculate grid layout
        cols = min(3, n_channels)
        rows = (n_channels + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=self.dpi)
        if n_channels == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, channel in enumerate(channels):
            data = input_data_dict[channel]
            
            # Determine colormap based on channel type
            if 'db' in channel.lower():
                cmap = 'viridis'
                vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
            else:
                cmap = 'plasma'
                vmin, vmax = np.percentile(data[np.isfinite(data)], [1, 99])
            
            im = axes[i].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
            axes[i].set_title(f'{channel}', fontweight='bold')
            axes[i].axis('off')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Input channels plot saved: {save_path}")
        
        return fig
    
    def plot_prediction_comparison(self,
                                 ground_truth: np.ndarray,
                                 regression_pred: np.ndarray,
                                 diffusion_pred: np.ndarray,
                                 title: str = "Model Predictions Comparison",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of ground truth, regression, and diffusion predictions.
        
        Args:
            ground_truth: Ground truth height data
            regression_pred: Regression model predictions
            diffusion_pred: Diffusion model predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Calculate global value range for consistent colormap
        all_data = np.concatenate([
            ground_truth.flatten(),
            regression_pred.flatten(), 
            diffusion_pred.flatten()
        ])
        valid_data = all_data[np.isfinite(all_data)]
        vmin, vmax = np.percentile(valid_data, [1, 99])
        
        # Calculate residuals
        reg_residual = regression_pred - ground_truth
        diff_residual = diffusion_pred - ground_truth
        
        # Residual range
        residual_max = max(np.abs(np.percentile(reg_residual[np.isfinite(reg_residual)], [1, 99])).max(),
                          np.abs(np.percentile(diff_residual[np.isfinite(diff_residual)], [1, 99])).max())
        
        fig = plt.figure(figsize=(20, 12), dpi=self.dpi)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Top row: Predictions
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax1.set_title('Ground Truth DSM', fontweight='bold', fontsize=12)
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(regression_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax2.set_title('Regression Prediction', fontweight='bold', fontsize=12)
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(diffusion_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax3.set_title('Diffusion Prediction', fontweight='bold', fontsize=12)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Bottom row: Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')  # Empty space for symmetry
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(reg_residual, cmap=self.color_maps['error'], 
                        vmin=-residual_max, vmax=residual_max)
        ax5.set_title('Regression Residual\n(Pred - Truth)', fontweight='bold', fontsize=12)
        ax5.axis('off')
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046)
        cbar5.set_label('Height Error (m)', fontsize=10)
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(diff_residual, cmap=self.color_maps['error'],
                        vmin=-residual_max, vmax=residual_max)
        ax6.set_title('Diffusion Residual\n(Pred - Truth)', fontweight='bold', fontsize=12)
        ax6.axis('off')
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046)
        cbar6.set_label('Height Error (m)', fontsize=10)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Prediction comparison plot saved: {save_path}")
        
        return fig
    
    def plot_ensemble_analysis(self,
                             ensemble_mean: np.ndarray,
                             ensemble_std: np.ndarray,
                             ground_truth: np.ndarray,
                             ensemble_predictions: List[np.ndarray],
                             title: str = "Ensemble Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive ensemble analysis including mean, uncertainty, and individual members.
        
        Args:
            ensemble_mean: Mean of ensemble predictions
            ensemble_std: Standard deviation of ensemble predictions  
            ground_truth: Ground truth height data
            ensemble_predictions: List of individual ensemble member predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_members = len(ensemble_predictions)
        
        # Calculate ranges
        height_range = np.percentile(ground_truth[np.isfinite(ground_truth)], [1, 99])
        uncertainty_range = [0, np.percentile(ensemble_std[np.isfinite(ensemble_std)], 99)]
        
        fig = plt.figure(figsize=(24, 16), dpi=self.dpi)
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Row 1: Main results
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth, cmap=self.color_maps['height'], 
                        vmin=height_range[0], vmax=height_range[1])
        ax1.set_title('Ground Truth', fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(ensemble_mean, cmap=self.color_maps['height'],
                        vmin=height_range[0], vmax=height_range[1])
        ax2.set_title('Ensemble Mean', fontweight='bold')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(ensemble_std, cmap=self.color_maps['uncertainty'],
                        vmin=uncertainty_range[0], vmax=uncertainty_range[1])
        ax3.set_title('Ensemble Uncertainty (Std)', fontweight='bold')
        ax3.axis('off')
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
        cbar3.set_label('Uncertainty (m)')
        
        # Ensemble residual
        ensemble_residual = ensemble_mean - ground_truth
        residual_range = np.abs(np.percentile(ensemble_residual[np.isfinite(ensemble_residual)], [1, 99])).max()
        
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(ensemble_residual, cmap=self.color_maps['error'],
                        vmin=-residual_range, vmax=residual_range)
        ax4.set_title('Ensemble Residual', fontweight='bold')
        ax4.axis('off')
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046)
        cbar4.set_label('Error (m)')
        
        # Row 2: Sample ensemble members (first 4)
        for i in range(min(4, n_members)):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(ensemble_predictions[i], cmap=self.color_maps['height'],
                          vmin=height_range[0], vmax=height_range[1])
            ax.set_title(f'Member {i+1}', fontweight='bold', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 3: Statistical analysis
        ax_hist = fig.add_subplot(gs[2, :2])
        
        # Histogram of uncertainties
        uncertainty_flat = ensemble_std[np.isfinite(ensemble_std)].flatten()
        ax_hist.hist(uncertainty_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax_hist.set_xlabel('Uncertainty (m)', fontweight='bold')
        ax_hist.set_ylabel('Frequency', fontweight='bold')
        ax_hist.set_title('Distribution of Ensemble Uncertainty', fontweight='bold')
        ax_hist.grid(True, alpha=0.3)
        
        # Scatter plot: uncertainty vs error
        ax_scatter = fig.add_subplot(gs[2, 2:])
        error_flat = np.abs(ensemble_residual[np.isfinite(ensemble_residual)]).flatten()
        uncertainty_flat_matched = ensemble_std[np.isfinite(ensemble_residual)].flatten()
        
        if len(error_flat) > 0 and len(uncertainty_flat_matched) > 0:
            # Subsample for visualization if too many points
            if len(error_flat) > 10000:
                indices = np.random.choice(len(error_flat), 10000, replace=False)
                error_sample = error_flat[indices]
                uncertainty_sample = uncertainty_flat_matched[indices]
            else:
                error_sample = error_flat
                uncertainty_sample = uncertainty_flat_matched
            
            ax_scatter.scatter(uncertainty_sample, error_sample, alpha=0.5, s=1, color='coral')
            ax_scatter.set_xlabel('Predicted Uncertainty (m)', fontweight='bold')
            ax_scatter.set_ylabel('Actual Error (m)', fontweight='bold') 
            ax_scatter.set_title('Uncertainty Calibration', fontweight='bold')
            ax_scatter.grid(True, alpha=0.3)
            
            # Add perfect calibration line
            max_val = max(ax_scatter.get_xlim()[1], ax_scatter.get_ylim()[1])
            ax_scatter.plot([0, max_val], [0, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Calibration')
            ax_scatter.legend()
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Ensemble analysis plot saved: {save_path}")
        
        return fig
    
    def plot_metrics_comparison(self,
                              metrics_dict: Dict[str, Dict[str, float]],
                              title: str = "Model Performance Metrics",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comparison of metrics across different models.
        
        Args:
            metrics_dict: Dictionary with model names as keys and metrics as values
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        if not metrics_dict:
            print("⚠ Warning: No metrics provided for comparison")
            return plt.figure()
        
        # Select key metrics for visualization
        key_metrics = ['rmse', 'mae', 'r2', 'correlation', 'vertical_accuracy_1m']
        available_metrics = set()
        for model_metrics in metrics_dict.values():
            available_metrics.update(model_metrics.keys())
        
        plot_metrics = [m for m in key_metrics if m in available_metrics]
        
        if not plot_metrics:
            print("⚠ Warning: No common metrics found for plotting")
            return plt.figure()
        
        n_metrics = len(plot_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 4, 6), dpi=self.dpi)
        if n_metrics == 1:
            axes = [axes]
        
        models = list(metrics_dict.keys())
        colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(plot_metrics):
            values = [metrics_dict[model].get(metric, 0) for model in models]
            
            bars = axes[i].bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            axes[i].set_title(f'{metric.upper().replace("_", " ")}', fontweight='bold')
            axes[i].set_ylabel('Value', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Metrics comparison plot saved: {save_path}")
        
        return fig
    
    def plot_height_distributions(self,
                                ground_truth: np.ndarray,
                                predictions_dict: Dict[str, np.ndarray],
                                title: str = "Height Distribution Comparison",
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot height distribution comparison between ground truth and predictions.
        
        Args:
            ground_truth: Ground truth height data
            predictions_dict: Dictionary with model names and predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=self.dpi)
        
        # Flatten and filter data
        truth_flat = ground_truth[np.isfinite(ground_truth)].flatten()
        
        # Plot histograms
        ax1.hist(truth_flat, bins=50, alpha=0.7, label='Ground Truth', color='black', density=True)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(predictions_dict)))
        for i, (model_name, pred) in enumerate(predictions_dict.items()):
            pred_flat = pred[np.isfinite(pred)].flatten()
            ax1.hist(pred_flat, bins=50, alpha=0.6, label=model_name, 
                    color=colors[i], density=True)
        
        ax1.set_xlabel('Height (m)', fontweight='bold')
        ax1.set_ylabel('Density', fontweight='bold')
        ax1.set_title('Height Distributions', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Q-Q plots
        for i, (model_name, pred) in enumerate(predictions_dict.items()):
            pred_flat = pred[np.isfinite(pred)].flatten()
            
            # Calculate quantiles
            if len(pred_flat) > 100:  # Minimum data requirement
                quantiles = np.linspace(0, 1, 100)
                truth_quantiles = np.quantile(truth_flat, quantiles)
                pred_quantiles = np.quantile(pred_flat, quantiles)
                
                ax2.scatter(truth_quantiles, pred_quantiles, alpha=0.6, 
                           color=colors[i], label=model_name, s=20)
        
        # Add perfect agreement line
        all_values = np.concatenate([truth_flat] + [pred[np.isfinite(pred)].flatten() 
                                                  for pred in predictions_dict.values()])
        min_val, max_val = np.percentile(all_values, [1, 99])
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.8)
        
        ax2.set_xlabel('Ground Truth Quantiles (m)', fontweight='bold')
        ax2.set_ylabel('Predicted Quantiles (m)', fontweight='bold')
        ax2.set_title('Q-Q Plot', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Height distributions plot saved: {save_path}")
        
        return fig
    
    def create_summary_dashboard(self,
                               input_data_dict: Dict[str, np.ndarray],
                               ground_truth: np.ndarray,
                               regression_pred: np.ndarray,
                               diffusion_pred: np.ndarray,
                               ensemble_mean: np.ndarray,
                               ensemble_std: np.ndarray,
                               metrics_summary: Dict[str, Any],
                               title: str = "SAR2Height Prediction Dashboard",
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create a comprehensive dashboard summarizing all results.
        
        Args:
            input_data_dict: Dictionary of input SAR channels
            ground_truth: Ground truth height data
            regression_pred: Regression predictions
            diffusion_pred: Diffusion predictions
            ensemble_mean: Ensemble mean predictions
            ensemble_std: Ensemble uncertainty
            metrics_summary: Summary of all metrics
            title: Main title for the dashboard
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(28, 20), dpi=self.dpi)
        gs = GridSpec(4, 6, figure=fig, hspace=0.4, wspace=0.3)
        
        # Calculate value ranges
        all_height_data = np.concatenate([
            ground_truth.flatten(), regression_pred.flatten(), 
            diffusion_pred.flatten(), ensemble_mean.flatten()
        ])
        height_range = np.percentile(all_height_data[np.isfinite(all_height_data)], [1, 99])
        
        uncertainty_range = [0, np.percentile(ensemble_std[np.isfinite(ensemble_std)], 99)]
        
        # Row 1: Input channels (first 3)
        input_channels = list(input_data_dict.keys())[:3]
        for i, channel in enumerate(input_channels):
            ax = fig.add_subplot(gs[0, i])
            data = input_data_dict[channel]
            vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
            
            im = ax.imshow(data, cmap='viridis', vmin=vmin, vmax=vmax)
            ax.set_title(f'Input: {channel}', fontweight='bold', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 1 continued: Metrics table
        ax_table = fig.add_subplot(gs[0, 3:])
        ax_table.axis('off')
        
        # Create metrics table
        table_data = []
        if 'regression' in metrics_summary and 'diffusion' in metrics_summary:
            reg_metrics = metrics_summary['regression']
            diff_metrics = metrics_summary['diffusion']
            
            for metric in ['rmse', 'mae', 'r2', 'vertical_accuracy_1m']:
                if metric in reg_metrics and metric in diff_metrics:
                    table_data.append([
                        metric.upper().replace('_', ' '),
                        f"{reg_metrics[metric]:.3f}",
                        f"{diff_metrics[metric]:.3f}"
                    ])
        
        if table_data:
            table = ax_table.table(cellText=table_data,
                                 colLabels=['Metric', 'Regression', 'Diffusion'],
                                 cellLoc='center',
                                 loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax_table.set_title('Model Performance Comparison', fontweight='bold', fontsize=12)
        
        # Row 2: Main predictions
        predictions = [
            ('Ground Truth', ground_truth, 'terrain'),
            ('Regression', regression_pred, 'terrain'),
            ('Diffusion', diffusion_pred, 'terrain'),
            ('Ensemble Mean', ensemble_mean, 'terrain'),
            ('Ensemble Uncertainty', ensemble_std, 'plasma'),
        ]
        
        for i, (name, data, cmap) in enumerate(predictions):
            ax = fig.add_subplot(gs[1, i])
            
            if 'Uncertainty' in name:
                vmin, vmax = uncertainty_range
            else:
                vmin, vmax = height_range
                
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(name, fontweight='bold', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 3: Residual analysis
        residuals = [
            ('Regression Residual', regression_pred - ground_truth),
            ('Diffusion Residual', diffusion_pred - ground_truth),
            ('Ensemble Residual', ensemble_mean - ground_truth)
        ]
        
        for i, (name, residual) in enumerate(residuals):
            ax = fig.add_subplot(gs[2, i])
            
            residual_max = np.abs(np.percentile(residual[np.isfinite(residual)], [1, 99])).max()
            im = ax.imshow(residual, cmap='RdBu_r', vmin=-residual_max, vmax=residual_max)
            ax.set_title(name, fontweight='bold', fontsize=10)
            ax.axis('off')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046)
            cbar.set_label('Error (m)', fontsize=8)
        
        # Row 3 continued: Statistical plots
        ax_dist = fig.add_subplot(gs[2, 3:5])
        
        # Distribution comparison
        truth_flat = ground_truth[np.isfinite(ground_truth)].flatten()
        reg_flat = regression_pred[np.isfinite(regression_pred)].flatten()
        diff_flat = diffusion_pred[np.isfinite(diffusion_pred)].flatten()
        ens_flat = ensemble_mean[np.isfinite(ensemble_mean)].flatten()
        
        ax_dist.hist(truth_flat, bins=30, alpha=0.6, label='Truth', density=True, color='black')
        ax_dist.hist(reg_flat, bins=30, alpha=0.5, label='Regression', density=True, color='blue')
        ax_dist.hist(diff_flat, bins=30, alpha=0.5, label='Diffusion', density=True, color='red')
        ax_dist.hist(ens_flat, bins=30, alpha=0.5, label='Ensemble', density=True, color='green')
        
        ax_dist.set_xlabel('Height (m)', fontweight='bold')
        ax_dist.set_ylabel('Density', fontweight='bold')
        ax_dist.set_title('Height Distributions', fontweight='bold', fontsize=10)
        ax_dist.legend(fontsize=8)
        ax_dist.grid(True, alpha=0.3)
        
        # Row 4: Additional analysis placeholder
        ax_summary = fig.add_subplot(gs[3, :])
        ax_summary.axis('off')
        
        # Summary text
        summary_text = f"""
        SAR2Height Prediction Summary:
        • Input Channels: {len(input_data_dict)} SAR features
        • Prediction Resolution: {ground_truth.shape[0]} × {ground_truth.shape[1]} pixels
        • Height Range: {height_range[0]:.1f} to {height_range[1]:.1f} meters
        • Best RMSE: {'Diffusion' if metrics_summary.get('diffusion', {}).get('rmse', float('inf')) < metrics_summary.get('regression', {}).get('rmse', float('inf')) else 'Regression'}
        """
        
        ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes,
                       fontsize=12, ha='center', va='center', 
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        fig.suptitle(title, fontsize=20, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Summary dashboard saved: {save_path}")
        
        return fig