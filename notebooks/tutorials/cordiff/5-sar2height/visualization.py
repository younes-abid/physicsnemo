"""
Data Visualization Module for SAR-to-Height Data Processing

This module provides comprehensive visualization functions for SAR and DSM data,
patches, statistics, and processing results for the cordiff training pipeline.

Author: AI Assistant
Date: 2026-01-06
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import logging
import pandas as pd
from pathlib import Path
from data_loader import DatasetStats, get_data_statistics
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Polygon
import geopandas as gpd
from matplotlib.patches import Polygon as MplPolygon

logger = logging.getLogger(__name__)

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class SARDataVisualizer:
    """
    Comprehensive visualization class for SAR-to-height data processing pipeline.
    
    This class provides methods to:
    - Visualize raw SAR and DSM data
    - Display feature extraction results
    - Show patch generation and quality assessment
    - Plot processing statistics and summaries
    - Create AOI coverage maps and overlap analysis
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100, speed: bool = True):
        """
        Initialize the visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size
            dpi (int): Default figure resolution in dots per inch
            speed (bool): Flag to enable speed optimization
        """
        self.figsize = figsize
        self.dpi = dpi
        self.speed_mode = speed
        
        # Speed optimization settings
        if self.speed_mode:
            self.downsample_factor = 4
            self.max_features_display = 8
            self.patch_display_limit = 100
            self.interpolation = 'nearest'  # Fastest interpolation
            self.skip_colorbars = True
        else:
            self.downsample_factor = 1
            self.max_features_display = 20
            self.patch_display_limit = None
            self.interpolation = 'bilinear'  # Better quality
            self.skip_colorbars = False
        
        logger.info(f"Visualizer initialized: DPI={self.dpi}, Speed mode={'ON' if self.speed_mode else 'OFF'}")
        
    def plot_raw_data_pair(self, intensity_data: np.ndarray, dsm_data: np.ndarray,
                          intensity_meta: Dict, dsm_meta: Dict,
                          title_prefix: str = "Raw Data") -> plt.Figure:
        """
        Plot raw SAR intensity and DSM data side by side.
        
        Args:
            intensity_data (np.ndarray): SAR intensity data
            dsm_data (np.ndarray): DSM height data
            intensity_meta (Dict): Intensity metadata
            dsm_meta (Dict): DSM metadata
            title_prefix (str): Prefix for plot titles
            
        Returns:
            plt.Figure: Figure object
        """
        # Apply downsampling in speed mode
        if self.speed_mode and self.downsample_factor > 1:
            intensity_display = intensity_data[::self.downsample_factor, ::self.downsample_factor]
            dsm_display = dsm_data[::self.downsample_factor, ::self.downsample_factor]
        else:
            intensity_display = intensity_data
            dsm_display = dsm_data
        
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=self.dpi)
        
        # Plot intensity data
        im1 = axes[0].imshow(intensity_display, cmap='gray', aspect='equal', 
                            interpolation=self.interpolation)
        axes[0].set_title(f'{title_prefix}: SAR Intensity (dB)')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        
        if not self.skip_colorbars:
            cbar1 = plt.colorbar(im1, ax=axes[0])
            cbar1.set_label('Intensity (dB)')
        
        # Add intensity statistics
        int_stats = f'Mean: {np.nanmean(intensity_data):.2f} dB\n'
        int_stats += f'Std: {np.nanstd(intensity_data):.2f} dB\n'
        int_stats += f'Range: [{np.nanmin(intensity_data):.2f}, {np.nanmax(intensity_data):.2f}]'
        if self.speed_mode:
            int_stats += f'\n(Downsampled {self.downsample_factor}x)'
        
        axes[0].text(0.02, 0.98, int_stats, transform=axes[0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=9)
        
        # Plot DSM data
        im2 = axes[1].imshow(dsm_display, cmap='terrain', aspect='equal',
                            interpolation=self.interpolation)
        axes[1].set_title(f'{title_prefix}: DSM Heights')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        
        if not self.skip_colorbars:
            cbar2 = plt.colorbar(im2, ax=axes[1])
            cbar2.set_label('Height (m)')
        
        # Add DSM statistics
        dsm_stats = f'Mean: {np.nanmean(dsm_data):.1f} m\n'
        dsm_stats += f'Std: {np.nanstd(dsm_data):.1f} m\n'
        dsm_stats += f'Range: [{np.nanmin(dsm_data):.1f}, {np.nanmax(dsm_data):.1f}]'
        if self.speed_mode:
            dsm_stats += f'\n(Downsampled {self.downsample_factor}x)'
            
        axes[1].text(0.02, 0.98, dsm_stats, transform=axes[1].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    verticalalignment='top', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_extraction_results(self, features: np.ndarray, 
                                      feature_names: List[str],
                                      max_features: int = None) -> plt.Figure:
        """
        Plot extracted SAR features in a grid layout with speed optimizations.
        
        Args:
            features (np.ndarray): Multi-channel feature array (channels, height, width)
            feature_names (List[str]): Names of features
            max_features (int): Maximum number of features to display (auto-set by speed mode)
            
        Returns:
            plt.Figure: Figure object
        """
        # Use speed mode settings if max_features not specified
        if max_features is None:
            max_features = self.max_features_display
            
        n_features = min(features.shape[0], max_features)
        n_cols = 4
        n_rows = int(np.ceil(n_features / n_cols))
        
        # Adjust figure size for speed mode
        if self.speed_mode:
            figure_size = (12, 3*n_rows)  # Smaller figure
        else:
            figure_size = (16, 4*n_rows)  # Larger figure
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figure_size, dpi=self.dpi)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else []
        
        for i in range(n_features):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                continue
                
            feature_data = features[i]
            feature_name = feature_names[i] if i < len(feature_names) else f'Feature {i}'
            
            # Apply downsampling in speed mode
            if self.speed_mode and self.downsample_factor > 1:
                feature_display = feature_data[::self.downsample_factor, ::self.downsample_factor]
            else:
                feature_display = feature_data
            
            # Use appropriate colormap based on feature type
            if 'direction' in feature_name.lower() or 'angle' in feature_name.lower():
                cmap, vmin, vmax = 'hsv', -np.pi, np.pi
            elif 'intensity' in feature_name.lower() and 'db' in feature_name.lower():
                cmap, vmin, vmax = 'gray', -30, 0  # Fixed range for speed
            else:
                cmap, vmin, vmax = 'viridis', None, None
            
            # Optimize range calculation for speed
            if vmin is None or vmax is None:
                if self.speed_mode:
                    # Use simple min/max for speed
                    finite_data = feature_display[np.isfinite(feature_display)]
                    if len(finite_data) > 0:
                        vmin, vmax = np.min(finite_data), np.max(finite_data)
                    else:
                        vmin, vmax = 0, 1
                else:
                    # Use percentiles for quality
                    valid_data = feature_display[np.isfinite(feature_display)]
                    if len(valid_data) > 0:
                        vmin, vmax = np.percentile(valid_data, [2, 98])
                    else:
                        vmin, vmax = 0, 1
            
            im = ax.imshow(feature_display, cmap=cmap, vmin=vmin, vmax=vmax, 
                          aspect='equal', interpolation=self.interpolation)
            
            # Truncate long feature names for speed mode
            display_name = feature_name[:15] + '...' if self.speed_mode and len(feature_name) > 15 else feature_name
            ax.set_title(display_name, fontsize=8 if self.speed_mode else 10)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar only if not in speed mode
            if not self.skip_colorbars:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6 if self.speed_mode else 8)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        title = f'SAR Features ({n_features}/{features.shape[0]})'
        if self.speed_mode:
            title += f' - Speed Mode (Downsampled {self.downsample_factor}x)'
            
        plt.suptitle(title, fontsize=14 if self.speed_mode else 16)
        plt.tight_layout()
        return fig
    
    def plot_patch_grid_overview(self, data: np.ndarray, patches: List[Dict],
                                patch_size: int = 432, title: str = "Patch Grid") -> plt.Figure:
        """
        Plot overview of patch locations on the original data with speed optimizations.
        
        Args:
            data (np.ndarray): Original data array
            patches (List[Dict]): List of patch dictionaries
            patch_size (int): Size of patches
            title (str): Plot title
            
        Returns:
            plt.Figure: Figure object
        """
        # Limit patches in speed mode
        if self.speed_mode and self.patch_display_limit:
            display_patches = patches[:self.patch_display_limit]
            if len(patches) > self.patch_display_limit:
                logger.info(f"Speed mode: Showing {self.patch_display_limit}/{len(patches)} patches")
        else:
            display_patches = patches
        
        fig, ax = plt.subplots(1, 1, figsize=self.figsize, dpi=self.dpi)
        
        # Plot background data
        if data.ndim == 3:
            plot_data = data[0]  # Use first channel
        else:
            plot_data = data
        
        # Apply downsampling to background data
        if self.speed_mode and self.downsample_factor > 1:
            plot_data = plot_data[::self.downsample_factor, ::self.downsample_factor]
            display_patch_size = patch_size // self.downsample_factor
        else:
            display_patch_size = patch_size
        
        im = ax.imshow(plot_data, cmap='gray', aspect='equal', interpolation=self.interpolation)
        
        # Color patches by quality
        qualities = [p['quality']['overall_quality'] for p in display_patches]
        norm = Normalize(vmin=min(qualities) if qualities else 0, 
                        vmax=max(qualities) if qualities else 1)
        
        # Import matplotlib patches with different name to avoid conflicts
        import matplotlib.patches as mpatches
        
        for i, patch in enumerate(display_patches):
            row_start, col_start = patch['grid_position']
            
            # Adjust coordinates for downsampling
            if self.speed_mode and self.downsample_factor > 1:
                row_start = row_start // self.downsample_factor
                col_start = col_start // self.downsample_factor
            
            # Create rectangle for patch
            rect = mpatches.Rectangle(
                (col_start, row_start), display_patch_size, display_patch_size,
                linewidth=1 if self.speed_mode else 2,
                edgecolor=plt.cm.RdYlGn(norm(patch['quality']['overall_quality'])),
                facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add patch ID less frequently in speed mode
            label_interval = 50 if self.speed_mode else 20
            if i % label_interval == 0:
                ax.text(col_start + display_patch_size//2, row_start + display_patch_size//2, 
                       str(patch['patch_id']), 
                       ha='center', va='center', fontsize=5 if self.speed_mode else 6, 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        display_title = f'{title} - {len(display_patches)} patches'
        if self.speed_mode and len(patches) > len(display_patches):
            display_title += f' (of {len(patches)} total)'
        if self.speed_mode:
            display_title += f' - Downsampled {self.downsample_factor}x'
            
        ax.set_title(display_title)
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Add colorbars only if not in speed mode
        if not self.skip_colorbars:
            cbar_data = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar_data.set_label('Data Value')
            
            sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
            sm.set_array([])
            cbar_quality = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.08)
            cbar_quality.set_label('Patch Quality')
        
        plt.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, file_path: str, dpi: int = None) -> None:
        """
        Save a figure to file.
        
        Args:
            fig: Figure object to save
            file_path: Path to save the figure
            dpi: Resolution in dots per inch (uses instance default if None)
        """
        if dpi is None:
            dpi = self.dpi
            
        try:
            fig.savefig(file_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            logger.info(f"Figure saved to {file_path} at {dpi} DPI")
        except Exception as e:
            logger.error(f"Error saving figure to {file_path}: {str(e)}")
            raise