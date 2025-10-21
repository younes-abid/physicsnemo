"""
Data Manager for Weather Pipeline - Streamlit Demo

Handles all data loading, preprocessing, and management operations.
Based on the successful prediction pipeline implementation.
"""

import os
import json
import xarray as xr
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Dict, Any, Optional


class DataManager:
    """Manages data loading and preprocessing for Weather Pipeline."""
    
    def __init__(self, data_file_path: str, stats_dir: str, use_dummy_data: bool = False):
        """
        Initialize DataManager with data paths.
        
        Args:
            data_file_path: Path to the specific data file
            stats_dir: Directory containing statistics files
            use_dummy_data: If True, generate dummy data instead of loading real data
        """
        self.data_file_path = data_file_path
        self.stats_dir = stats_dir
        self.use_dummy_data = use_dummy_data
        self.stats = None
        self.input_data = None
        self.output_data = None
        self.sample_datetime = None
        
    def load_statistics(self) -> Dict[str, Any]:
        """Load normalization statistics."""
        if self.use_dummy_data:
            # Generate dummy statistics
            self.stats = self._generate_dummy_stats()
        else:
            stats_path = os.path.join(self.stats_dir, "stat.json")
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"Statistics file not found: {stats_path}")
            
            with open(stats_path) as f:
                self.stats = json.load(f)
        
        print(f"✓ Statistics loaded")
        return self.stats
    
    def load_sample_data(self, sample_idx: int = 100) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Load a specific sample from the dataset.
        
        Args:
            sample_idx: Index of the sample to load
            
        Returns:
            Tuple of (input_data, output_data) as xarray Datasets
        """
        if self.use_dummy_data:
            self.input_data, self.output_data = self._generate_dummy_data()
            self.sample_datetime = pd.Timestamp.now()
        else:
            if not os.path.exists(self.data_file_path):
                raise FileNotFoundError(f"Data file not found: {self.data_file_path}")
            
            # Load the root dataset to access time coordinate
            root_dataset = xr.open_dataset(self.data_file_path)
            
            # Extract sample datetime if available
            if 'time' in root_dataset.data_vars:
                sample_time = root_dataset.time.isel(sample=sample_idx).values
                self.sample_datetime = pd.to_datetime(sample_time)
                print(f"✓ Sample {sample_idx} corresponds to: {self.sample_datetime}")
                print(f"  - Date: {self.sample_datetime.strftime('%Y-%m-%d')}")
                print(f"  - Time: {self.sample_datetime.strftime('%H:%M:%S')}")
                print(f"  - Day of week: {self.sample_datetime.strftime('%A')}")
            else:
                print(f"✓ Using sample index {sample_idx}")
            
            # Load grouped data
            input_group = xr.open_dataset(self.data_file_path, group="input")
            output_group = xr.open_dataset(self.data_file_path, group="output")
            self.input_data = input_group.isel(sample=sample_idx)
            self.output_data = output_group.isel(sample=sample_idx)
            
            # Close datasets to free memory
            root_dataset.close()
            input_group.close()
            output_group.close()
        
        print(f"✓ Data loaded successfully")
        print(f"  - Input variables: {list(self.input_data.data_vars)}")
        print(f"  - Output variables: {list(self.output_data.data_vars)}")
        print(f"  - Input shape: {self.input_data.dims}")
        print(f"  - Output shape: {self.output_data.dims}")
        
        return self.input_data, self.output_data
    
    def get_variable_statistics(self, show_all: bool = False) -> Dict[str, Any]:
        """Get statistics of loaded variables."""
        if self.input_data is None or self.output_data is None:
            return {"error": "No data loaded. Call load_sample_data() first."}
        
        stats_info = {
            "input_variables": {},
            "output_variables": {}
        }
        
        input_vars = list(self.input_data.data_vars)
        display_vars = input_vars if show_all else input_vars[:5]
        
        for var in display_vars:
            data = self.input_data[var].values
            stats_info["input_variables"][var] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std())
            }
        
        for var in list(self.output_data.data_vars):
            data = self.output_data[var].values
            stats_info["output_variables"][var] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std())
            }
        
        return stats_info
    
    def normalize_input(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize input data using loaded statistics."""
        if self.stats is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        mean = self.stats["input"][var_name]["mean"]
        std = self.stats["input"][var_name]["std"]
        return (data - mean) / std
    
    def denormalize_output(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Denormalize output data using loaded statistics."""
        if self.stats is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        mean = self.stats["output"][var_name]["mean"]
        std = self.stats["output"][var_name]["std"]
        return (data * std) + mean
    
    def normalize_output(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize output data using loaded statistics (for ground truth comparison)."""
        if self.stats is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        mean = self.stats["output"][var_name]["mean"]
        std = self.stats["output"][var_name]["std"]
        return (data - mean) / std
    
    def get_selected_variables(self) -> list:
        """Get the list of selected input variables in the correct order."""
        return [
            't_850', 't_500', 'z_850', 'z_500',  # 4 variables
            'u_850', 'u_500', 'v_850', 'v_500',  # +4 = 8
            'u10', 'v10', 't2m', 'd2m',          # +4 = 12
            "skt", "sp", "tcwv", "tp"            # +4 = 16
        ]
    
    def prepare_input_tensor(self, apply_manual_normalization: bool = False) -> torch.Tensor:
        """
        Prepare input tensor for model inference.
        
        Args:
            apply_manual_normalization: If True, apply manual normalization using stats.json.
        
        Returns:
            Input tensor of shape (1, 16, H, W)
        """
        if self.input_data is None:
            raise ValueError("No data loaded. Call load_sample_data() first.")
        
        selected_vars = self.get_selected_variables()
        
        # Stack input variables
        input_stack = []
        for var in selected_vars:
            var_data = self.input_data[var].values
            
            if apply_manual_normalization:
                var_processed = self.normalize_input(var_data, var)
            else:
                var_processed = var_data
            
            input_stack.append(var_processed)
        
        input_array = np.stack(input_stack, axis=0)  # Shape: (16, H, W)
        input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)  # (1, 16, H, W)
        
        return input_tensor
    
    def get_ground_truth(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get ground truth U10 and V10 data.
        
        Returns:
            Tuple of (u10_true, v10_true) as numpy arrays
        """
        if self.output_data is None:
            raise ValueError("No data loaded. Call load_sample_data() first.")
        
        u10_true = self.output_data["U10"].values
        v10_true = self.output_data["V10"].values
        
        return u10_true, v10_true
    
    def get_datetime_info(self) -> Optional[pd.Timestamp]:
        """Get the datetime information for the loaded sample."""
        return self.sample_datetime
    
    def _generate_dummy_stats(self) -> Dict[str, Any]:
        """Generate dummy statistics for testing."""
        variables = self.get_selected_variables()
        stats = {
            "input": {},
            "output": {"U10": {"mean": 0.0, "std": 1.0}, "V10": {"mean": 0.0, "std": 1.0}}
        }
        
        for var in variables:
            stats["input"][var] = {"mean": 0.0, "std": 1.0}
        
        return stats
    
    def _generate_dummy_data(self) -> Tuple[xr.Dataset, xr.Dataset]:
        """Generate dummy data for testing."""
        H, W = 432, 432
        variables = self.get_selected_variables()
        
        # Create dummy input data
        input_data_dict = {}
        for var in variables:
            input_data_dict[var] = (["y", "x"], np.random.randn(H, W))
        
        input_data = xr.Dataset(input_data_dict, coords={"y": range(H), "x": range(W)})
        
        # Create dummy output data
        output_data_dict = {
            "U10": (["y", "x"], np.random.randn(H, W)),
            "V10": (["y", "x"], np.random.randn(H, W))
        }
        
        output_data = xr.Dataset(output_data_dict, coords={"y": range(H), "x": range(W)})
        
        return input_data, output_data