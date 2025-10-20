"""
Visualizer for CorrDiff Prediction Pipeline

Handles all visualization and plotting operations for model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from typing import Tuple, Dict, Any, Optional


class Visualizer:
    """Creates visualizations for CorrDiff prediction results."""
    
    def __init__(self):
        """Initialize Visualizer."""
        pass
    
    def plot_regression_results(self, u10_pred: np.ndarray, v10_pred: np.ndarray,
                              u10_true: np.ndarray, v10_true: np.ndarray,
                              title_prefix: str = "Regression") -> None:
        """
        Plot regression results with proper scaling using xarray.
        
        Args:
            u10_pred: Predicted U10 values
            v10_pred: Predicted V10 values
            u10_true: Ground truth U10 values
            v10_true: Ground truth V10 values
            title_prefix: Prefix for plot titles
        """
        print(f"=== {title_prefix} Visualization ===")
        
        # Create xarray DataArrays with proper coordinates
        H, W = u10_pred.shape
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        
        # Create DataArrays for predictions and ground truth
        u10_pred_da = xr.DataArray(u10_pred, coords=[('y', y_coords), ('x', x_coords)], name='U10_prediction')
        u10_true_da = xr.DataArray(u10_true, coords=[('y', y_coords), ('x', x_coords)], name='U10_ground_truth')
        v10_pred_da = xr.DataArray(v10_pred, coords=[('y', y_coords), ('x', x_coords)], name='V10_prediction')
        v10_true_da = xr.DataArray(v10_true, coords=[('y', y_coords), ('x', x_coords)], name='V10_ground_truth')
        
        # Calculate residuals
        u10_residual_da = u10_pred_da - u10_true_da
        v10_residual_da = v10_pred_da - v10_true_da
        
        # Calculate proper scaling ranges
        u10_min = min(u10_pred_da.min().values, u10_true_da.min().values)
        u10_max = max(u10_pred_da.max().values, u10_true_da.max().values)
        v10_min = min(v10_pred_da.min().values, v10_true_da.min().values)
        v10_max = max(v10_pred_da.max().values, v10_true_da.max().values)
        
        # Calculate residual range for symmetric scaling around zero
        u10_residual_max = max(abs(u10_residual_da.min().values), abs(u10_residual_da.max().values))
        v10_residual_max = max(abs(v10_residual_da.min().values), abs(v10_residual_da.max().values))
        
        print(f"Data ranges:")
        print(f"  U10: [{u10_min:.2f}, {u10_max:.2f}]")
        print(f"  V10: [{v10_min:.2f}, {v10_max:.2f}]")
        print(f"  U10 residual: ±{u10_residual_max:.2f}")
        print(f"  V10 residual: ±{v10_residual_max:.2f}")
        
        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # U10 plots with proper scaling
        u10_pred_da.plot(ax=axes[0, 0], cmap='coolwarm', add_colorbar=True, 
                         vmin=u10_min, vmax=u10_max,
                         cbar_kwargs={'label': 'U10 (m/s)'})
        axes[0, 0].set_title('U10 Prediction')
        axes[0, 0].set_aspect('equal')
        
        u10_true_da.plot(ax=axes[0, 1], cmap='coolwarm', add_colorbar=True, 
                         vmin=u10_min, vmax=u10_max,
                         cbar_kwargs={'label': 'U10 (m/s)'})
        axes[0, 1].set_title('U10 Ground Truth')
        axes[0, 1].set_aspect('equal')
        
        u10_residual_da.plot(ax=axes[0, 2], cmap='RdBu_r', add_colorbar=True, 
                            vmin=-u10_residual_max, vmax=u10_residual_max,
                            cbar_kwargs={'label': 'U10 Residual (m/s)'})
        axes[0, 2].set_title('U10 Residual (Pred - True)')
        axes[0, 2].set_aspect('equal')
        
        # V10 plots with proper scaling
        v10_pred_da.plot(ax=axes[1, 0], cmap='coolwarm', add_colorbar=True, 
                         vmin=v10_min, vmax=v10_max,
                         cbar_kwargs={'label': 'V10 (m/s)'})
        axes[1, 0].set_title('V10 Prediction')
        axes[1, 0].set_aspect('equal')
        
        v10_true_da.plot(ax=axes[1, 1], cmap='coolwarm', add_colorbar=True, 
                         vmin=v10_min, vmax=v10_max,
                         cbar_kwargs={'label': 'V10 (m/s)'})
        axes[1, 1].set_title('V10 Ground Truth')
        axes[1, 1].set_aspect('equal')
        
        v10_residual_da.plot(ax=axes[1, 2], cmap='RdBu_r', add_colorbar=True, 
                            vmin=-v10_residual_max, vmax=v10_residual_max,
                            cbar_kwargs={'label': 'V10 Residual (m/s)'})
        axes[1, 2].set_title('V10 Residual (Pred - True)')
        axes[1, 2].set_aspect('equal')
        
        plt.tight_layout()
        plt.suptitle(f'{title_prefix} Results - Full Domain (432x432)', y=0.98)
        plt.show()
        
        # Additional compact plots
        self._plot_compact_comparison(u10_pred_da, u10_true_da, u10_residual_da, 
                                    u10_min, u10_max, u10_residual_max, "U10")
        self._plot_compact_comparison(v10_pred_da, v10_true_da, v10_residual_da, 
                                    v10_min, v10_max, v10_residual_max, "V10")
        
        print(f"✓ {title_prefix} visualization completed")
    
    def _plot_compact_comparison(self, pred_da: xr.DataArray, true_da: xr.DataArray, 
                               residual_da: xr.DataArray, vmin: float, vmax: float, 
                               residual_max: float, component: str) -> None:
        """Create compact comparison plot for a single component."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Prediction
        pred_da.plot(ax=axes[0], cmap='coolwarm', vmin=vmin, vmax=vmax,
                    cbar_kwargs={'label': f'{component} (m/s)'}, add_colorbar=True)
        axes[0].set_title(f'{component} Prediction')
        axes[0].set_aspect('equal')
        
        # Ground Truth
        true_da.plot(ax=axes[1], cmap='coolwarm', vmin=vmin, vmax=vmax,
                    cbar_kwargs={'label': f'{component} (m/s)'}, add_colorbar=True)
        axes[1].set_title(f'{component} Ground Truth')
        axes[1].set_aspect('equal')
        
        # Residual
        residual_da.plot(ax=axes[2], cmap='RdBu_r', 
                        vmin=-residual_max, vmax=residual_max,
                        cbar_kwargs={'label': f'{component} Residual (m/s)'}, add_colorbar=True)
        axes[2].set_title(f'{component} Residual (Pred - True)')
        axes[2].set_aspect('equal')
        
        plt.tight_layout()
        plt.suptitle(f'{component} Wind Component Comparison', y=1.02)
        plt.show()
    
    def plot_comparison(self, regression_pred: Tuple[np.ndarray, np.ndarray],
                       diffusion_pred: Tuple[np.ndarray, np.ndarray],
                       ground_truth: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Plot comparison between regression, diffusion, and ground truth.
        
        Args:
            regression_pred: Tuple of (u10_reg, v10_reg)
            diffusion_pred: Tuple of (u10_diff, v10_diff)
            ground_truth: Tuple of (u10_true, v10_true)
        """
        u10_reg, v10_reg = regression_pred
        u10_diff, v10_diff = diffusion_pred
        u10_true, v10_true = ground_truth
        
        print("=== Three-Way Comparison Visualization ===")
        
        # Create comprehensive comparison using xarray
        H, W = u10_true.shape
        y_coords = np.arange(H)
        x_coords = np.arange(W)
        
        # Create datasets with 'type' dimension for easy plotting
        ds_u10 = xr.Dataset({
            'U10': (['type', 'y', 'x'], [u10_reg, u10_diff, u10_true]),
            'Residual_U10': (['type', 'y', 'x'], [u10_reg - u10_true, u10_diff - u10_true, np.zeros_like(u10_true)])
        }, coords={
            'type': ['Regression', 'Diffusion', 'Ground Truth'],
            'y': y_coords,
            'x': x_coords
        })
        
        ds_v10 = xr.Dataset({
            'V10': (['type', 'y', 'x'], [v10_reg, v10_diff, v10_true]),
            'Residual_V10': (['type', 'y', 'x'], [v10_reg - v10_true, v10_diff - v10_true, np.zeros_like(v10_true)])
        }, coords={
            'type': ['Regression', 'Diffusion', 'Ground Truth'],
            'y': y_coords,
            'x': x_coords
        })
        
        # Plot U10 comparison
        print("\n=== U10 Wind Component Comparison ===")
        ds_u10.U10.plot(
            col='type',
            col_wrap=3,
            cmap='coolwarm',
            robust=True,
            figsize=(18, 6),
            cbar_kwargs={'label': 'U10 (m/s)'}
        )
        plt.suptitle('U10 Wind Component - Regression vs Diffusion vs Ground Truth', y=1.02)
        plt.show()
        
        # Plot V10 comparison
        print("\n=== V10 Wind Component Comparison ===")
        ds_v10.V10.plot(
            col='type',
            col_wrap=3,
            cmap='coolwarm',
            robust=True,
            figsize=(18, 6),
            cbar_kwargs={'label': 'V10 (m/s)'}
        )
        plt.suptitle('V10 Wind Component - Regression vs Diffusion vs Ground Truth', y=1.02)
        plt.show()
        
        # Plot residuals comparison (excluding ground truth)
        print("\n=== Residuals Comparison ===")
        ds_u10.Residual_U10.sel(type=['Regression', 'Diffusion']).plot(
            col='type',
            col_wrap=2,
            cmap='RdBu_r',
            robust=True,
            figsize=(12, 6),
            cbar_kwargs={'label': 'U10 Residual (m/s)'}
        )
        plt.suptitle('U10 Residuals - Regression vs Diffusion', y=1.02)
        plt.show()
        
        ds_v10.Residual_V10.sel(type=['Regression', 'Diffusion']).plot(
            col='type',
            col_wrap=2,
            cmap='RdBu_r',
            robust=True,
            figsize=(12, 6),
            cbar_kwargs={'label': 'V10 Residual (m/s)'}
        )
        plt.suptitle('V10 Residuals - Regression vs Diffusion', y=1.02)
        plt.show()
        
        print("✓ Three-way comparison visualization completed")
    
    def plot_input_variables(self, data_manager, show_variables: Optional[int] = None, max_cols: int = 4) -> None:
        """
        Visualize input variables from the data manager with dynamic subplot grid.
        
        Args:
            data_manager: DataManager instance with loaded data
            show_variables: Number of variables to display. If None, shows all variables
            max_cols: Maximum number of columns in the subplot grid (default: 4)
        """
        if data_manager.input_data is None:
            print("No input data available to visualize")
            return
        
        print(f"=== Input Variables Visualization ===")
        
        # Get all variables if show_variables is None, otherwise limit to show_variables
        all_variables = list(data_manager.input_data.data_vars)
        if show_variables is None:
            variables = all_variables
        else:
            variables = all_variables[:show_variables]
        
        num_vars = len(variables)
        print(f"Visualizing {num_vars} variables: {variables}")
        
        # Calculate optimal grid dimensions
        cols = min(max_cols, num_vars)
        rows = (num_vars + cols - 1) // cols  # Ceiling division
        
        print(f"Using {rows}x{cols} grid for {num_vars} variables")
        
        # Calculate figure size based on grid
        fig_width = cols * 5
        fig_height = rows * 4
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # Handle single subplot case
        if num_vars == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        for i, var in enumerate(variables):
            data = data_manager.input_data[var]
            data.plot(ax=axes[i], cmap='viridis', add_colorbar=True)
            axes[i].set_title(f'{var}')
            axes[i].set_aspect('equal')
        
        # Hide unused subplots
        for i in range(len(variables), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle(f'Input Atmospheric Variables ({num_vars} variables)', y=0.98)
        plt.show()
        
        print(f"✓ Input variables visualization completed")
    
    def plot_summary_metrics(self, metrics_dict: Dict[str, float], 
                           title: str = "Model Performance") -> None:
        """
        Create a summary plot of key metrics.
        
        Args:
            metrics_dict: Dictionary containing computed metrics
            title: Title for the plot
        """
        # Extract key metrics for plotting
        key_metrics = {
            'R² U10': metrics_dict.get('r2_U10', 0),
            'R² V10': metrics_dict.get('r2_V10', 0),
            'RMSE U10': metrics_dict.get('rmse_U10', 0),
            'RMSE V10': metrics_dict.get('rmse_V10', 0),
            'MAE U10': metrics_dict.get('mae_U10', 0),
            'MAE V10': metrics_dict.get('mae_V10', 0),
        }
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² values (higher is better)
        r2_metrics = {k: v for k, v in key_metrics.items() if 'R²' in k}
        ax1.bar(r2_metrics.keys(), r2_metrics.values(), color=['skyblue', 'lightcoral'])
        ax1.set_title('R² Scores (Higher = Better)')
        ax1.set_ylabel('R² Value')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Error metrics (lower is better)
        error_metrics = {k: v for k, v in key_metrics.items() if 'RMSE' in k or 'MAE' in k}
        colors = ['lightgreen', 'lightgreen', 'orange', 'orange']
        ax2.bar(error_metrics.keys(), error_metrics.values(), color=colors)
        ax2.set_title('Error Metrics (Lower = Better)')
        ax2.set_ylabel('Error Value')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        plt.show()
        
        print(f"✓ Summary metrics visualization completed")