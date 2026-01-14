"""
SAR2Height Visualizer

Advanced visualization and plotting for SAR-to-height prediction results.
Supports comprehensive visualization of inputs, outputs, comparisons, and ensemble analysis.
Updated to support aspect ratio correction for proper geographic display.
Includes automatic subplot saving for clean individual figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import os
from pathlib import Path

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
    - Automatic individual subplot saving for LaTeX papers
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

    def _save_individual_subplots(self, fig, save_path: str, subplot_metadata: List[Dict[str, Any]]):
        """
        Save individual subplots as clean figures for LaTeX papers.
        
        Args:
            fig: The main figure containing subplots
            save_path: Path where the main figure was saved
            subplot_metadata: List of metadata for each subplot
        """
        if not save_path:
            return
            
        # Create subfolder with same name as the figure
        base_path = Path(save_path)
        subfolder_name = base_path.stem  # Get filename without extension
        subfolder_path = base_path.parent / subfolder_name
        subfolder_path.mkdir(exist_ok=True)
        
        print(f"  📁 Saving individual subplots to: {subfolder_path}")
        
        # Get all axes from the figure
        axes = fig.get_axes()
        
        for i, (ax, metadata) in enumerate(zip(axes, subplot_metadata)):
            # Skip empty or non-image axes
            if not metadata.get('has_data', True):
                continue
                
            # Determine subplot position for naming
            row, col = metadata.get('position', (i // 10, i % 10))  # Fallback positioning
            subplot_name = f"{row}_{col}"
            
            # Create clean subplot figure
            clean_fig, clean_ax = plt.subplots(1, 1, figsize=(6, 6), dpi=self.dpi)
            
            # Copy the image data and properties
            images = ax.get_images()
            if images and metadata.get('plot_type') == 'image':
                # Copy image
                im = images[0]
                array = im.get_array()
                extent = im.get_extent()
                cmap = im.get_cmap()
                vmin, vmax = im.get_clim()
                
                # Plot clean image without labels or colorbar
                clean_ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, 
                               extent=extent if extent != (0, 1, 0, 1) else None)
                clean_ax.axis('off')
                
                # Apply aspect ratio correction
                if hasattr(self, '_configure_image_aspect'):
                    self._configure_image_aspect(clean_ax, array.shape)
                
            elif metadata.get('plot_type') == 'histogram':
                # Copy histogram data
                patches = ax.patches
                if patches:
                    # Get histogram data from the original axis
                    n, bins, patches_orig = None, None, None
                    # Extract data by re-plotting (this is a workaround)
                    hist_data = metadata.get('hist_data', [])
                    if hist_data:
                        clean_ax.hist(hist_data, bins=metadata.get('bins', 50), 
                                    alpha=0.7, color=metadata.get('color', 'blue'),
                                    edgecolor='black')
                        clean_ax.grid(True, alpha=0.3)
                        if metadata.get('vline_x') is not None:
                            clean_ax.axvline(metadata['vline_x'], color='red', 
                                           linestyle='--', alpha=0.7)
                
            elif metadata.get('plot_type') == 'scatter':
                # Copy scatter plot data
                collections = ax.collections
                if collections:
                    for collection in collections:
                        offsets = collection.get_offsets()
                        colors = collection.get_facecolors()
                        sizes = collection.get_sizes()
                        clean_ax.scatter(offsets[:, 0], offsets[:, 1], 
                                       c=colors, s=sizes, alpha=0.5)
                        clean_ax.grid(True, alpha=0.3)
                        
                # Add perfect calibration line if it exists
                lines = ax.get_lines()
                if lines:
                    for line in lines:
                        xdata, ydata = line.get_data()
                        clean_ax.plot(xdata, ydata, color=line.get_color(),
                                    linestyle=line.get_linestyle(),
                                    linewidth=line.get_linewidth(),
                                    alpha=line.get_alpha())
            
            # Remove all labels, titles, and text
            clean_ax.set_title('')
            clean_ax.set_xlabel('')
            clean_ax.set_ylabel('')
            clean_ax.tick_params(labelbottom=False, labelleft=False,
                               bottom=False, left=False)
            
            # Save clean subplot
            subplot_path = subfolder_path / f"{subplot_name}.png"
            clean_fig.savefig(subplot_path, dpi=self.dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
            plt.close(clean_fig)
            
            # Save metadata as JSON
            metadata_path = subfolder_path / f"{subplot_name}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"  💾 Saved {len([m for m in subplot_metadata if m.get('has_data', True)])} individual subplots")

    def _create_subplot_metadata(self, ax, row: int, col: int, plot_type: str, **kwargs) -> Dict[str, Any]:
        """
        Create metadata for a subplot.
        
        Args:
            ax: The matplotlib axis
            row: Row position in grid
            col: Column position in grid
            plot_type: Type of plot ('image', 'histogram', 'scatter', etc.)
            **kwargs: Additional metadata
            
        Returns:
            Dictionary containing subplot metadata
        """
        metadata = {
            'position': (row, col),
            'plot_type': plot_type,
            'has_data': True,
            'title': ax.get_title(),
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
        }
        
        if plot_type == 'image':
            images = ax.get_images()
            if images:
                im = images[0]
                metadata.update({
                    'colormap': im.get_cmap().name,
                    'vmin': im.get_clim()[0],
                    'vmax': im.get_clim()[1],
                    'extent': im.get_extent(),
                    'data_shape': im.get_array().shape if hasattr(im.get_array(), 'shape') else None,
                })
        
        # Add any additional metadata
        metadata.update(kwargs)
        
        return metadata

    # ...existing _setup_aspect_ratio and _configure_image_aspect methods...
    
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
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        for i, channel in enumerate(channels):
            row, col = divmod(i, cols)
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
            
            # Create metadata
            metadata = self._create_subplot_metadata(
                axes[i], row, col, 'image',
                channel_name=channel,
                colormap=cmap,
                vmin=vmin,
                vmax=vmax,
                data_type='SAR_input',
                colorbar_label=f'{channel} values'
            )
            subplot_metadata.append(metadata)
        
        # Handle unused subplots
        for i in range(n_channels, len(axes)):
            axes[i].axis('off')
            row, col = divmod(i, cols)
            metadata = self._create_subplot_metadata(axes[i], row, col, 'empty', has_data=False)
            subplot_metadata.append(metadata)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Input channels plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Top row: Predictions - using jet colormap for heights
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax1.set_title('Ground Truth DSM', fontweight='bold', fontsize=12)
        ax1.axis('off')
        self._configure_image_aspect(ax1, ground_truth.shape)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        metadata1 = self._create_subplot_metadata(
            ax1, 0, 0, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='ground_truth_dsm',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(regression_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax2.set_title('Regression Prediction', fontweight='bold', fontsize=12)
        ax2.axis('off')
        self._configure_image_aspect(ax2, regression_pred.shape)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        metadata2 = self._create_subplot_metadata(
            ax2, 0, 1, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='regression_prediction',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(diffusion_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        ax3.set_title('Diffusion Prediction', fontweight='bold', fontsize=12)
        ax3.axis('off')
        self._configure_image_aspect(ax3, diffusion_pred.shape)
        plt.colorbar(im3, ax=ax3, fraction=0.046)
        
        metadata3 = self._create_subplot_metadata(
            ax3, 0, 2, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='diffusion_prediction',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata3)
        
        # Bottom row: Residuals
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')  # Empty space for symmetry
        metadata4 = self._create_subplot_metadata(ax4, 1, 0, 'empty', has_data=False)
        subplot_metadata.append(metadata4)
        
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(reg_residual, cmap=self.color_maps['error'], 
                        vmin=-residual_max, vmax=residual_max)
        ax5.set_title('Regression Residual\n(Pred - Truth)', fontweight='bold', fontsize=12)
        ax5.axis('off')
        self._configure_image_aspect(ax5, reg_residual.shape)
        cbar5 = plt.colorbar(im5, ax=ax5, fraction=0.046)
        cbar5.set_label('Height Error (m)', fontsize=10)
        
        metadata5 = self._create_subplot_metadata(
            ax5, 1, 1, 'image',
            colormap='RdBu_r', vmin=-residual_max, vmax=residual_max,
            data_type='regression_residual',
            colorbar_label='Height Error (m)'
        )
        subplot_metadata.append(metadata5)
        
        ax6 = fig.add_subplot(gs[1, 2])
        im6 = ax6.imshow(diff_residual, cmap=self.color_maps['error'],
                        vmin=-residual_max, vmax=residual_max)
        ax6.set_title('Diffusion Residual\n(Pred - Truth)', fontweight='bold', fontsize=12)
        ax6.axis('off')
        self._configure_image_aspect(ax6, diff_residual.shape)
        cbar6 = plt.colorbar(im6, ax=ax6, fraction=0.046)
        cbar6.set_label('Height Error (m)', fontsize=10)
        
        metadata6 = self._create_subplot_metadata(
            ax6, 1, 2, 'image',
            colormap='RdBu_r', vmin=-residual_max, vmax=residual_max,
            data_type='diffusion_residual',
            colorbar_label='Height Error (m)'
        )
        subplot_metadata.append(metadata6)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Prediction comparison plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Plot ensemble members
        for i, pred in enumerate(ensemble_predictions):
            row, col = divmod(i, cols)
            im = axes[i].imshow(pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
            axes[i].set_title(f'Member {i+1}', fontweight='bold', fontsize=10)
            axes[i].axis('off')
            self._configure_image_aspect(axes[i], pred.shape)
            plt.colorbar(im, ax=axes[i], fraction=0.046)
            
            metadata = self._create_subplot_metadata(
                axes[i], row, col, 'image',
                colormap='jet', vmin=vmin, vmax=vmax,
                data_type=f'ensemble_member_{i+1}',
                colorbar_label='Height (m)'
            )
            subplot_metadata.append(metadata)
        
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
                row, col = divmod(idx, cols)
                im = axes[idx].imshow(data, cmap=cmap, vmin=vmin_stat, vmax=vmax_stat)
                axes[idx].set_title(f'Ensemble {name}', fontweight='bold', fontsize=10)
                axes[idx].axis('off')
                self._configure_image_aspect(axes[idx], data.shape)
                cbar = plt.colorbar(im, ax=axes[idx], fraction=0.046)
                if name == 'Std':
                    cbar.set_label('Uncertainty (m)', fontsize=8)
                
                metadata = self._create_subplot_metadata(
                    axes[idx], row, col, 'image',
                    colormap=cmap if isinstance(cmap, str) else cmap.name,
                    vmin=vmin_stat, vmax=vmax_stat,
                    data_type=f'ensemble_{name.lower()}',
                    colorbar_label='Uncertainty (m)' if name == 'Std' else 'Height (m)'
                )
                subplot_metadata.append(metadata)
        
        # Hide unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')
            row, col = divmod(i, cols)
            metadata = self._create_subplot_metadata(axes[i], row, col, 'empty', has_data=False)
            subplot_metadata.append(metadata)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ All ensemble members plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        residual_flat = residual[np.isfinite(residual)].flatten()
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Ground Truth
        im1 = axes[0, 0].imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Ground Truth DSM', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], ground_truth.shape)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        metadata1 = self._create_subplot_metadata(
            axes[0, 0], 0, 0, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='ground_truth_dsm',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata1)
        
        # Regression Prediction
        im2 = axes[0, 1].imshow(regression_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Regression Prediction', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], regression_pred.shape)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        metadata2 = self._create_subplot_metadata(
            axes[0, 1], 0, 1, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='regression_prediction',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata2)
        
        # Regression Residual
        im3 = axes[1, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[1, 0].set_title('Regression Residual\n(Pred - Truth)', fontweight='bold')
        axes[1, 0].axis('off')
        self._configure_image_aspect(axes[1, 0], residual.shape)
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        cbar3.set_label('Height Error (m)')
        
        metadata3 = self._create_subplot_metadata(
            axes[1, 0], 1, 0, 'image',
            colormap='RdBu_r', vmin=-residual_max, vmax=residual_max,
            data_type='regression_residual',
            colorbar_label='Height Error (m)'
        )
        subplot_metadata.append(metadata3)
        
        # Statistical plot
        axes[1, 1].hist(residual_flat, bins=50, alpha=0.7, 
                       color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Residual Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        metadata4 = self._create_subplot_metadata(
            axes[1, 1], 1, 1, 'histogram',
            hist_data=residual_flat, bins=50, color='skyblue',
            vline_x=0, data_type='residual_distribution'
        )
        subplot_metadata.append(metadata4)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Regression analysis plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        residual_flat = residual[np.isfinite(residual)].flatten()
        
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 10 * min(1.5, self.aspect_ratio_factor / 3)
            fig_size = (12, fig_height)
        else:
            fig_size = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=self.dpi)
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Main residual plot
        im1 = axes[0, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[0, 0].set_title('Regression Residual Map', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], residual.shape)
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        cbar1.set_label('Height Error (m)')
        
        metadata1 = self._create_subplot_metadata(
            axes[0, 0], 0, 0, 'image',
            colormap='RdBu_r', vmin=-residual_max, vmax=residual_max,
            data_type='regression_residual_map',
            colorbar_label='Height Error (m)'
        )
        subplot_metadata.append(metadata1)
        
        # Absolute residual
        abs_residual = np.abs(residual)
        im2 = axes[0, 1].imshow(abs_residual, cmap='Reds', vmin=0, vmax=residual_max)
        axes[0, 1].set_title('Absolute Regression Error', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], abs_residual.shape)
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        cbar2.set_label('|Error| (m)')
        
        metadata2 = self._create_subplot_metadata(
            axes[0, 1], 0, 1, 'image',
            colormap='Reds', vmin=0, vmax=residual_max,
            data_type='absolute_regression_error',
            colorbar_label='|Error| (m)'
        )
        subplot_metadata.append(metadata2)
        
        # Residual histogram
        axes[1, 0].hist(residual_flat, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        metadata3 = self._create_subplot_metadata(
            axes[1, 0], 1, 0, 'histogram',
            hist_data=residual_flat, bins=50, color='lightcoral',
            vline_x=0, data_type='residual_distribution'
        )
        subplot_metadata.append(metadata3)
        
        # Q-Q plot against normal distribution
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residual_flat, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        
        metadata4 = self._create_subplot_metadata(
            axes[1, 1], 1, 1, 'scatter',
            data_type='qq_plot_normal',
            plot_description='Q-Q plot against normal distribution'
        )
        subplot_metadata.append(metadata4)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Regression residual focus plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        residual_flat = residual[np.isfinite(residual)].flatten()
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Ground Truth
        im1 = axes[0, 0].imshow(ground_truth, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 0].set_title('Ground Truth DSM', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], ground_truth.shape)
        plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        
        metadata1 = self._create_subplot_metadata(
            axes[0, 0], 0, 0, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='ground_truth_dsm',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata1)
        
        # Diffusion Prediction
        im2 = axes[0, 1].imshow(diffusion_pred, cmap=self.color_maps['height'], vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Diffusion Prediction', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], diffusion_pred.shape)
        plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        
        metadata2 = self._create_subplot_metadata(
            axes[0, 1], 0, 1, 'image',
            colormap='jet', vmin=vmin, vmax=vmax,
            data_type='diffusion_prediction',
            colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata2)
        
        # Diffusion Residual
        im3 = axes[1, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[1, 0].set_title('Diffusion Residual\n(Pred - Truth)', fontweight='bold')
        axes[1, 0].axis('off')
        self._configure_image_aspect(axes[1, 0], residual.shape)
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        cbar3.set_label('Height Error (m)')
        
        metadata3 = self._create_subplot_metadata(
            axes[1, 0], 1, 0, 'image',
            colormap='RdBu_r', vmin=-residual_max, vmax=residual_max,
            data_type='diffusion_residual',
            colorbar_label='Height Error (m)'
        )
        subplot_metadata.append(metadata3)
        
        # Statistical plot
        axes[1, 1].hist(residual_flat, bins=50, alpha=0.7, 
                       color='lightgreen', edgecolor='black')
        axes[1, 1].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontweight='bold')
        axes[1, 1].set_title('Residual Distribution', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        metadata4 = self._create_subplot_metadata(
            axes[1, 1], 1, 1, 'histogram',
            hist_data=residual_flat, bins=50, color='lightgreen',
            vline_x=0, data_type='residual_distribution'
        )
        subplot_metadata.append(metadata4)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Diffusion analysis plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        residual_flat = residual[np.isfinite(residual)].flatten()
        
        # Adjust figure size for aspect ratio
        if self.apply_aspect_correction:
            fig_height = 10 * min(1.5, self.aspect_ratio_factor / 3)
            fig_size = (12, fig_height)
        else:
            fig_size = (12, 10)
        
        fig, axes = plt.subplots(2, 2, figsize=fig_size, dpi=self.dpi)
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Main residual plot
        im1 = axes[0, 0].imshow(residual, cmap=self.color_maps['error'], vmin=-residual_max, vmax=residual_max)
        axes[0, 0].set_title('Diffusion Residual Map', fontweight='bold')
        axes[0, 0].axis('off')
        self._configure_image_aspect(axes[0, 0], residual.shape)
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)
        cbar1.set_label('Height Error (m)')
        
        metadata1 = self._create_subplot_metadata(
            axes[0, 0], 0, 0, 'image',
            colormap='RdBu_r', vmin=-residual_max, vmax=residual_max,
            data_type='diffusion_residual_map',
            colorbar_label='Height Error (m)'
        )
        subplot_metadata.append(metadata1)
        
        # Absolute residual
        abs_residual = np.abs(residual)
        im2 = axes[0, 1].imshow(abs_residual, cmap='Reds', vmin=0, vmax=residual_max)
        axes[0, 1].set_title('Absolute Diffusion Error', fontweight='bold')
        axes[0, 1].axis('off')
        self._configure_image_aspect(axes[0, 1], abs_residual.shape)
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)
        cbar2.set_label('|Error| (m)')
        
        metadata2 = self._create_subplot_metadata(
            axes[0, 1], 0, 1, 'image',
            colormap='Reds', vmin=0, vmax=residual_max,
            data_type='absolute_diffusion_error',
            colorbar_label='|Error| (m)'
        )
        subplot_metadata.append(metadata2)
        
        # Residual histogram
        axes[1, 0].hist(residual_flat, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 0].set_xlabel('Residual (m)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(0, color='black', linestyle='--', alpha=0.7)
        
        metadata3 = self._create_subplot_metadata(
            axes[1, 0], 1, 0, 'histogram',
            hist_data=residual_flat, bins=50, color='lightgreen',
            vline_x=0, data_type='residual_distribution'
        )
        subplot_metadata.append(metadata3)
        
        # Q-Q plot against normal distribution
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(residual_flat, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlabel('Theoretical Quantiles')
        axes[1, 1].set_ylabel('Sample Quantiles')
        
        metadata4 = self._create_subplot_metadata(
            axes[1, 1], 1, 1, 'scatter',
            data_type='qq_plot_normal',
            plot_description='Q-Q plot against normal distribution'
        )
        subplot_metadata.append(metadata4)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Diffusion residual focus plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
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
        
        # Prepare metadata for subplots
        subplot_metadata = []
        
        # Row 1: Main results
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(ground_truth, cmap=self.color_maps['height'], 
                        vmin=height_range[0], vmax=height_range[1])
        ax1.set_title('Ground Truth', fontweight='bold')
        ax1.axis('off')
        self._configure_image_aspect(ax1, ground_truth.shape)
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        metadata1 = self._create_subplot_metadata(
            ax1, 0, 0, 'image',
            colormap='jet', vmin=height_range[0], vmax=height_range[1],
            data_type='ground_truth', colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata1)
        
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(ensemble_mean, cmap=self.color_maps['height'],
                        vmin=height_range[0], vmax=height_range[1])
        ax2.set_title('Ensemble Mean', fontweight='bold')
        ax2.axis('off')
        self._configure_image_aspect(ax2, ensemble_mean.shape)
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        metadata2 = self._create_subplot_metadata(
            ax2, 0, 1, 'image',
            colormap='jet', vmin=height_range[0], vmax=height_range[1],
            data_type='ensemble_mean', colorbar_label='Height (m)'
        )
        subplot_metadata.append(metadata2)
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(ensemble_std, cmap=self.color_maps['uncertainty'],
                        vmin=uncertainty_range[0], vmax=uncertainty_range[1])
        ax3.set_title('Ensemble Uncertainty (Std)', fontweight='bold')
        ax3.axis('off')
        self._configure_image_aspect(ax3, ensemble_std.shape)
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046)
        cbar3.set_label('Uncertainty (m)')
        
        metadata3 = self._create_subplot_metadata(
            ax3, 0, 2, 'image',
            colormap='plasma', vmin=uncertainty_range[0], vmax=uncertainty_range[1],
            data_type='ensemble_uncertainty', colorbar_label='Uncertainty (m)'
        )
        subplot_metadata.append(metadata3)
        
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
        
        metadata4 = self._create_subplot_metadata(
            ax4, 0, 3, 'image',
            colormap='RdBu_r', vmin=-residual_range, vmax=residual_range,
            data_type='ensemble_residual', colorbar_label='Error (m)'
        )
        subplot_metadata.append(metadata4)
        
        # Row 2: Sample ensemble members (first 4) - using jet colormap
        for i in range(min(4, n_members)):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(ensemble_predictions[i], cmap=self.color_maps['height'],
                          vmin=height_range[0], vmax=height_range[1])
            ax.set_title(f'Member {i+1}', fontweight='bold', fontsize=10)
            ax.axis('off')
            self._configure_image_aspect(ax, ensemble_predictions[i].shape)
            plt.colorbar(im, ax=ax, fraction=0.046)
            
            metadata_member = self._create_subplot_metadata(
                ax, 1, i, 'image',
                colormap='jet', vmin=height_range[0], vmax=height_range[1],
                data_type=f'ensemble_member_{i+1}', colorbar_label='Height (m)'
            )
            subplot_metadata.append(metadata_member)
        
        # Row 3: Statistical analysis (unchanged - these are plots, not images)
        ax_hist = fig.add_subplot(gs[2, :2])
        
        # Histogram of uncertainties
        uncertainty_flat = ensemble_std[np.isfinite(ensemble_std)].flatten()
        ax_hist.hist(uncertainty_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax_hist.set_xlabel('Uncertainty (m)', fontweight='bold')
        ax_hist.set_ylabel('Frequency', fontweight='bold')
        ax_hist.set_title('Distribution of Ensemble Uncertainty', fontweight='bold')
        ax_hist.grid(True, alpha=0.3)
        
        metadata_hist = self._create_subplot_metadata(
            ax_hist, 2, 0, 'histogram',
            hist_data=uncertainty_flat, bins=50, color='skyblue',
            data_type='uncertainty_distribution'
        )
        subplot_metadata.append(metadata_hist)
        
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
        
        metadata_scatter = self._create_subplot_metadata(
            ax_scatter, 2, 1, 'scatter',
            data_type='uncertainty_calibration',
            plot_description='Uncertainty vs actual error scatter plot with perfect calibration line'
        )
        subplot_metadata.append(metadata_scatter)
        
        fig.suptitle(title, fontsize=18, fontweight='bold')
        
        if save_path:
            fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"✓ Ensemble analysis plot saved: {save_path}")
            # Save individual subplots
            self._save_individual_subplots(fig, save_path, subplot_metadata)
        
        return fig