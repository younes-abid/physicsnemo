"""
SAR2Height Visualizer

Advanced visualization and plotting for SAR-to-height prediction results.
Supports comprehensive visualization of inputs, outputs, comparisons, and ensemble analysis.
Updated to support aspect ratio correction for proper geographic display.
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
    - Aspect ratio correction for proper geographic display
    """
    
    def __init__(self, 
                 figsize_base: Tuple[int, int] = (12, 8), 
                 dpi: int = 100,
                 aspect_ratio_factor: float = 1.0,
                 apply_aspect_correction: bool = False,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize SAR2HeightVisualizer.
        
        Args:
            figsize_base: Base figure size for single plots
            dpi: DPI for high-quality plots
            aspect_ratio_factor: Factor to correct aspect ratio (height_resolution/width_resolution)
            apply_aspect_correction: Whether to apply aspect ratio correction
            config: Additional visualization configuration
        """
        self.figsize_base = figsize_base
        self.dpi = dpi
        self.aspect_ratio_factor = aspect_ratio_factor
        self.apply_aspect_correction = apply_aspect_correction
        self.config = config or {}
        
        # Updated colormaps: Gray for intensity (SAR), Jet for heights (DSM)
        self.color_maps = {
            'height': 'jet',  # Changed to jet for height/DSM data
            'sar_intensity': 'gray',  # Gray for intensity channels
            'sar_other': 'viridis',  # Other SAR features
            'error': 'RdBu_r',
            'uncertainty': 'plasma'
        }
        
        # Aspect ratio configuration
        self._setup_aspect_ratio()
        
        print(f"🎨 Visualizer initialized with aspect ratio correction: {self.apply_aspect_correction}")
        if self.apply_aspect_correction:
            print(f"   📐 Aspect ratio factor: {self.aspect_ratio_factor:.1f}x")
    
    def _setup_aspect_ratio(self):
        """Setup aspect ratio correction parameters."""
        if self.apply_aspect_correction and self.aspect_ratio_factor != 1.0:
            self.aspect_ratio_method = "physical"
            # Calculate display aspect ratio
            self.display_aspect = 1.0 / self.aspect_ratio_factor  # Inverse for matplotlib
        else:
            self.aspect_ratio_method = "equal"
            self.display_aspect = "equal"
    
    def _configure_image_aspect(self, ax, data_shape: Tuple[int, int]):
        """Configure aspect ratio for an image plot."""
        if self.apply_aspect_correction and self.aspect_ratio_factor != 1.0:
            # Apply physical aspect ratio correction
            ax.set_aspect(self.display_aspect)
        else:
            # Use equal aspect (square pixels)
            ax.set_aspect('equal')
    
    def plot_input_channels(self, 
                          input_data_dict: Dict[str, np.ndarray],
                          title: str = "SAR Input Channels",
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot all SAR input channels in a grid layout with proper colormaps and aspect ratio.
        
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
        
        # Adjust figure size for aspect ratio if needed
        base_width, base_height = 4, 3
        if self.apply_aspect_correction:
            # Adjust subplot size to account for aspect ratio
            adjusted_height = base_height * self.aspect_ratio_factor / 4  # Normalize to reasonable size
            fig_size = (cols * base_width, rows * adjusted_height)
        else:
            fig_size = (cols * base_width, rows * base_height)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=self.dpi)
        if n_channels == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if hasattr(axes, '__iter__') else [axes]
        else:
            axes = axes.flatten()
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, channel in enumerate(channels):
            data = input_data_dict[channel]
            
            # Use gray colormap for intensity channels, viridis for others
            if 'intensity' in channel.lower():
                cmap = self.color_maps['sar_intensity']
            else:
                cmap = self.color_maps['sar_other']
                
            vmin, vmax = np.percentile(data[np.isfinite(data)], [2, 98])
            
            im = axes[i].imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[i].set_title(f'{channel}', fontweight='bold')
            axes[i].axis('off')
            
            # Configure aspect ratio
            self._configure_image_aspect(axes[i], data.shape)
            
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
        Plot comparison of ground truth, regression, and diffusion predictions with aspect ratio correction.
        
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
        
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 12 * min(2.0, self.aspect_ratio_factor / 2)  # Limit extreme ratios
            fig_size = (20, fig_height)
        else:
            fig_size = (20, 12)
        
        fig = plt.figure(figsize=fig_size, dpi=self.dpi)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Top row: Predictions - using jet colormap for heights
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax1.set_title('Ground Truth DSM', fontweight='bold', fontsize=12)
        ax1.axis('off')
        self._configure_image_aspect(ax1, ground_truth.shape)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(regression_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax2.set_title('Regression Prediction', fontweight='bold', fontsize=12)
        ax2.axis('off')
        self._configure_image_aspect(ax2, regression_pred.shape)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(diffusion_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax3.set_title('Diffusion Prediction', fontweight='bold', fontsize=12)
        ax3.axis('off')
        self._configure_image_aspect(ax3, diffusion_pred.shape)
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        # Bottom row: Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')  # Empty space for symmetry
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(reg_residual, cmap=self.color_maps['error'], 
                        vmin=-residual_max, vmax=residual_max)
        ax5.set_title('Regression Residual\n(Pred - Truth)', fontweight='bold', fontsize=12)
        ax5.axis('off')
        self._configure_image_aspect(ax5, reg_residual.shape)
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046)
        cbar5.set_label('Height Error (m)', fontsize=10)
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(diff_residual, cmap=self.color_maps['error'],
                        vmin=-residual_max, vmax=residual_max)
        ax6.set_title('Diffusion Residual\n(Pred - Truth)', fontweight='bold', fontsize=12)
        ax6.axis('off')
        self._configure_image_aspect(ax6, diff_residual.shape)
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046)
        cbar6.set_label('Height Error (m)', fontsize=10)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Prediction comparison plot saved: {save_path}")
        
        return fig

    def plot_all_ensemble_members(self,
                                 ensemble_predictions: List[np.ndarray],
                                 ground_truth: np.ndarray,
                                 title: str = "All Ensemble Members",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot all ensemble members in a grid with statistical summary and aspect ratio correction.
        
        Args:
            ensemble_predictions: List of all ensemble member predictions
            ground_truth: Ground truth for comparison
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_members = len(ensemble_predictions)
        
        # Calculate grid layout for ensemble members + 4 statistical plots
        total_plots = n_members + 4  # +4 for min, max, mean, std
        cols = min(6, int(np.ceil(np.sqrt(total_plots))))
        rows = int(np.ceil(total_plots / cols))
        
        # Adjust figure size for aspect ratio
        base_subplot_size = 3
        if self.apply_aspect_correction:
            subplot_height = base_subplot_size * min(2.0, self.aspect_ratio_factor / 3)
            fig_size = (cols * base_subplot_size, rows * subplot_height)
        else:
            fig_size = (cols * base_subplot_size, rows * base_subplot_size)
        
        fig, axes = plt.subplots(rows, cols, figsize=fig_size, dpi=self.dpi)
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        # Calculate value range for consistent colormap
        all_data = np.concatenate([pred.flatten() for pred in ensemble_predictions] + [ground_truth.flatten()])
        valid_data = all_data[np.isfinite(all_data)]
        vmin, vmax = np.percentile(valid_data, [1, 99])
        
        # Plot ensemble members
        for i, pred in enumerate(ensemble_predictions):
            im = axes[i].imshow(pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Member {i+1}', fontweight='bold', fontsize=10)
            axes[i].axis('off')
            self._configure_image_aspect(axes[i], pred.shape)
            plt.colorbar(im, ax=axes[i], fraction=0.046)
        
        # Calculate ensemble statistics
        ensemble_array = np.stack(ensemble_predictions, axis=0)
        ensemble_min = np.min(ensemble_array, axis=0)
        ensemble_max = np.max(ensemble_array, axis=0)
        ensemble_mean = np.mean(ensemble_array, axis=0)
        ensemble_std = np.std(ensemble_array, axis=0)
        
        # Plot statistical summaries
        stat_plots = [
            ('Min', ensemble_min, self.color_maps['height'], vmin, vmax),
            ('Max', ensemble_max, self.color_maps['height'], vmin, vmax),
            ('Mean', ensemble_mean, self.color_maps['height'], vmin, vmax),
            ('Std', ensemble_std, self.color_maps['uncertainty'], 0, np.percentile(ensemble_std[np.isfinite(ensemble_std)], 99))
        ]
        
        for i, (name, data, cmap, vmin_stat, vmax_stat) in enumerate(stat_plots):
            idx = n_members + i
            if idx < len(axes):
                im = axes[idx].imshow(data, cmap=cmap, vmin=vmin_stat, vmax=vmax_stat)
                axes[idx].set_title(f'Ensemble {name}', fontweight='bold', fontsize=10)
                axes[idx].axis('off')
                self._configure_image_aspect(axes[idx], data.shape)
                cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046)
                if name == 'Std':
                    cbar.set_label('Uncertainty (m)', fontsize=8)
        
        # Hide unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ All ensemble members plot saved: {save_path}")
        
        return fig

    def plot_regression_predictions(self,
                                  ground_truth: np.ndarray,
                                  regression_pred: np.ndarray,
                                  title: str = "Regression Model Analysis",
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot regression prediction analysis grid with aspect ratio correction.
        
        Args:
            ground_truth: Ground truth height data
            regression_pred: Regression model predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 10 * min(1.5, self.aspect_ratio_factor / 3)
            fig_size = (12, fig_height)
        else:
            fig_size = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=self.dpi)
        
        # Calculate value ranges
        height_data = np.concatenate([ground_truth.flatten(), regression_pred.flatten()])
        valid_data = height_data[np.isfinite(height_data)]
        vmin, vmax = np.percentile(valid_data, [1, 99])
        
        residual = regression_pred - ground_truth
        residual_max = np.abs(np.percentile(residual[np.isfinite(residual)], [1, 99])).max()
        
        # Ground Truth
        im1 = axes[0, 0].imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Ground Truth DSM', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], ground_truth.shape)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Regression Prediction
        im2 = axes[0, 1].imshow(regression_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Regression Prediction', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], regression_pred.shape)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Regression Residual
        im3 = axes[1, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[1, 0].set_title('Regression Residual\n(Pred - Truth)', fontweight='bold')
        axes[1, 0].axis('off')
        self._configure_image_aspect(axes[1, 0], residual.shape)
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        cbar3.set_label('Height Error (m)')
        
        # Statistical plot
        axes[1, 1].hist(residual[np.isfinite(residual)].flatten(), bins=50, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Residual Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Regression analysis plot saved: {save_path}")
        
        return fig

    def plot_regression_residual_focus(self,
                                     ground_truth: np.ndarray,
                                     regression_pred: np.ndarray,
                                     title: str = "Regression Residual Analysis",
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Focused plot on regression residual analysis with aspect ratio correction.
        
        Args:
            ground_truth: Ground truth height data
            regression_pred: Regression model predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        residual = regression_pred - ground_truth
        residual_max = np.abs(np.percentile(residual[np.isfinite(residual)], [1, 99])).max()
        
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 10 * min(1.5, self.aspect_ratio_factor / 3)
            fig_size = (12, fig_height)
        else:
            fig_size = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=self.dpi)
        
        # Main residual plot
        im1 = axes[0, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[0, 0].set_title('Regression Residual Map', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], residual.shape)
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        cbar1.set_label('Height Error (m)')
        
        # Absolute residual
        abs_residual = np.abs(residual)
        im2 = axes[0, 1].imshow(abs_residual, cmap='Reds', vmin=0, vmax=residual_max)
        axes[0, 1].set_title('Absolute Regression Error', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], abs_residual.shape)
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        cbar2.set_label('|Error| (m)')
        
        # Residual histogram
        residual_flat = residual[np.isfinite(residual)].flatten()
        axes[1, 0].hist(residual_flat, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Q-Q plot against normal distribution
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residual_flat, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Regression residual focus plot saved: {save_path}")
        
        return fig

    def plot_diffusion_predictions(self,
                                 ground_truth: np.ndarray,
                                 diffusion_pred: np.ndarray,
                                 title: str = "Diffusion Model Analysis",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot diffusion prediction analysis grid with aspect ratio correction.
        
        Args:
            ground_truth: Ground truth height data
            diffusion_pred: Diffusion model predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 10 * min(1.5, self.aspect_ratio_factor / 3)
            fig_size = (12, fig_height)
        else:
            fig_size = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=self.dpi)
        
        # Calculate value ranges
        height_data = np.concatenate([ground_truth.flatten(), diffusion_pred.flatten()])
        valid_data = height_data[np.isfinite(height_data)]
        vmin, vmax = np.percentile(valid_data, [1, 99])
        
        residual = diffusion_pred - ground_truth
        residual_max = np.abs(np.percentile(residual[np.isfinite(residual)], [1, 99])).max()
        
        # Ground Truth
        im1 = axes[0, 0].imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Ground Truth DSM', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], ground_truth.shape)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        # Diffusion Prediction
        im2 = axes[0, 1].imshow(diffusion_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Diffusion Prediction', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], diffusion_pred.shape)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        # Diffusion Residual
        im3 = axes[1, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[1, 0].set_title('Diffusion Residual\n(Pred - Truth)', fontweight='bold')
        axes[1, 0].axis('off')
        self._configure_image_aspect(axes[1, 0], residual.shape)
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        cbar3.set_label('Height Error (m)')
        
        # Statistical plot
        axes[1, 1].hist(residual[np.isfinite(residual)].flatten(), bins=50, alpha=0.7, 
                       color='lightgreen', edgecolor='black')
        axes[1, 1].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Residual Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Diffusion analysis plot saved: {save_path}")
        
        return fig

    def plot_diffusion_residual_focus(self,
                                    ground_truth: np.ndarray,
                                    diffusion_pred: np.ndarray,
                                    title: str = "Diffusion Residual Analysis",
                                    save_path: Optional[str] = None) -> plt.Figure:
        """
        Focused plot on diffusion residual analysis with aspect ratio correction.
        
        Args:
            ground_truth: Ground truth height data
            diffusion_pred: Diffusion model predictions
            title: Main title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        residual = diffusion_pred - ground_truth
        residual_max = np.abs(np.percentile(residual[np.isfinite(residual)], [1, 99])).max()
        
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 10 * min(1.5, self.aspect_ratio_factor / 3)
            fig_size = (12, fig_height)
        else:
            fig_size = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=self.dpi)
        
        # Main residual plot
        im1 = axes[0, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[0, 0].set_title('Diffusion Residual Map', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], residual.shape)
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        cbar1.set_label('Height Error (m)')
        
        # Absolute residual
        abs_residual = np.abs(residual)
        im2 = axes[0, 1].imshow(abs_residual, cmap='Reds', vmin=0, vmax=residual_max)
        axes[0, 1].set_title('Absolute Diffusion Error', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], abs_residual.shape)
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        cbar2.set_label('|Error| (m)')
        
        # Residual histogram
        residual_flat = residual[np.isfinite(residual)].flatten()
        axes[1, 0].hist(residual_flat, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        # Q-Q plot against normal distribution
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residual_flat, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Diffusion residual focus plot saved: {save_path}")
        
        return fig
    
    def plot_ensemble_analysis(self,
                             ensemble_mean: np.ndarray,
                             ensemble_std: np.ndarray,
                             ground_truth: np.ndarray,
                             ensemble_predictions: List[np.ndarray],
                             title: str = "Ensemble Analysis",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot comprehensive ensemble analysis including mean, uncertainty, and individual members with aspect ratio correction.
        
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
        
        # Calculate ranges - using jet colormap for heights
        height_range = np.percentile(ground_truth[np.isfinite(ground_truth)], [1, 99])
        uncertainty_range = [0, np.percentile(ensemble_std[np.isfinite(ensemble_std)], 99)]
        
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 16 * min(1.5, self.aspect_ratio_factor / 4)
            fig_size = (24, fig_height)
        else:
            fig_size = (24, 16)
        
        fig = plt.figure(figsize=fig_size, dpi=self.dpi)
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)
        
        # Row 1: Main results
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth, cmap=self.color_maps['height'], 
                        vmin=height_range[0], vmax=height_range[1])
        ax1.set_title('Ground Truth', fontweight='bold')
        ax1.axis('off')
        self._configure_image_aspect(ax1, ground_truth.shape)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(ensemble_mean, cmap=self.color_maps['height'],
                        vmin=height_range[0], vmax=height_range[1])
        ax2.set_title('Ensemble Mean', fontweight='bold')
        ax2.axis('off')
        self._configure_image_aspect(ax2, ensemble_mean.shape)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(ensemble_std, cmap=self.color_maps['uncertainty'],
                        vmin=uncertainty_range[0], vmax=uncertainty_range[1])
        ax3.set_title('Ensemble Uncertainty (Std)', fontweight='bold')
        ax3.axis('off')
        self._configure_image_aspect(ax3, ensemble_std.shape)
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
        self._configure_image_aspect(ax4, ensemble_residual.shape)
        cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046)
        cbar4.set_label('Error (m)')
        
        # Row 2: Sample ensemble members (first 4) - using jet colormap
        for i in range(min(4, n_members)):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(ensemble_predictions[i], cmap=self.color_maps['height'],
                          vmin=height_range[0], vmax=height_range[1])
            ax.set_title(f'Member {i+1}', fontweight='bold', fontsize=10)
            ax.axis('off')
            self._configure_image_aspect(ax, ensemble_predictions[i].shape)
            plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Row 3: Statistical analysis (unchanged - these are plots, not images)
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