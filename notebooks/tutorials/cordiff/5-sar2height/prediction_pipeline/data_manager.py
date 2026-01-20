"""
SAR2Height Data Manager

Handles loading, normalization, and preprocessing of SAR2Height data for prediction.
Manages the complex NetCDF structure with input/output/metadata groups.
"""

import numpy as np
import xarray as xr
import torch
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SAR2HeightDataManager:
    """
    Manages SAR2Height data loading, normalization, and preprocessing for prediction pipeline.
    
    Handles the NetCDF structure:
    - input group: SAR features (intensity_db, intensity_percentile_rescaled, etc.)
    - output group: DSM height data (dsm_height)
    - patch_metadata group: Complete patch metadata for filtering
    """
    
    def __init__(self, data_file_path: str, stats_dir: str, use_dummy_data: bool = False):
        """
        Initialize SAR2HeightDataManager.
        
        Args:
            data_file_path: Path to the preprocessed NetCDF file
            stats_dir: Directory containing statistics files
            use_dummy_data: If True, generate dummy data instead of loading real data
        """
        self.data_file_path = data_file_path
        self.stats_dir = stats_dir
        self.use_dummy_data = use_dummy_data
        self.stats = None
        self.input_data = None
        self.output_data = None
        self.metadata = None
        self.input_variables = None
        self.output_variables = None
        
    def load_statistics(self) -> Dict[str, Any]:
        """Load normalization statistics for SAR2Height data."""
        if self.use_dummy_data:
            self.stats = self._generate_dummy_stats()
        else:
            stats_path = Path(self.stats_dir) / "stat.json"
            if not stats_path.exists():
                raise FileNotFoundError(f"Statistics file not found: {stats_path}")
            
            with open(stats_path) as f:
                self.stats = json.load(f)
        
        print(f"✓ SAR2Height statistics loaded")
        print(f"  - Input variables: {list(self.stats['input'].keys())}")
        print(f"  - Output variables: {list(self.stats['output'].keys())}")
        return self.stats
    
    def load_sample_data(self, sample_idx: int = 0) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """
        Load a specific sample from the SAR2Height dataset.
        
        Args:
            sample_idx: Index of the sample to load
            
        Returns:
            Tuple of (input_data, output_data, metadata) as xarray Datasets
        """
        if self.use_dummy_data:
            self.input_data, self.output_data, self.metadata = self._generate_dummy_data()
        else:
            if not Path(self.data_file_path).exists():
                raise FileNotFoundError(f"Data file not found: {self.data_file_path}")
            
            # Load all groups from SAR2Height NetCDF structure
            input_group = xr.open_dataset(self.data_file_path, group="input")
            output_group = xr.open_dataset(self.data_file_path, group="output")
            metadata_group = xr.open_dataset(self.data_file_path, group="patch_metadata")
            
            # Check if sample exists
            n_samples = input_group.dims['sample']
            if sample_idx >= n_samples:
                raise ValueError(f"Sample index {sample_idx} >= number of samples {n_samples}")
            
            # Extract specific sample
            self.input_data = input_group.isel(sample=sample_idx)
            self.output_data = output_group.isel(sample=sample_idx)
            self.metadata = metadata_group.isel(sample=sample_idx)
            
            # Store variable lists
            self.input_variables = list(self.input_data.data_vars)
            self.output_variables = list(self.output_data.data_vars)
            
            # Close datasets to free memory
            input_group.close()
            output_group.close()
            metadata_group.close()
        
        print(f"✓ SAR2Height data loaded successfully")
        print(f"  - Sample index: {sample_idx}")
        print(f"  - Input variables: {self.input_variables}")
        print(f"  - Output variables: {self.output_variables}")
        print(f"  - Input shape: {dict(self.input_data.dims)}")
        print(f"  - Output shape: {dict(self.output_data.dims)}")
        
        return self.input_data, self.output_data, self.metadata
    
    def get_variable_statistics(self, show_all: bool = True) -> Dict[str, Any]:
        """Get statistics of loaded SAR2Height variables."""
        if self.input_data is None or self.output_data is None:
            return {"error": "No data loaded. Call load_sample_data() first."}
        
        stats_info = {
            "input_variables": {},
            "output_variables": {},
            "metadata": {}
        }
        
        # Input variables statistics
        for var in self.input_variables:
            data = self.input_data[var].values
            stats_info["input_variables"][var] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "shape": data.shape
            }
        
        # Output variables statistics
        for var in self.output_variables:
            data = self.output_data[var].values
            stats_info["output_variables"][var] = {
                "min": float(data.min()),
                "max": float(data.max()),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "shape": data.shape
            }
        
        # Metadata information
        if self.metadata is not None:
            for var in self.metadata.data_vars:
                try:
                    data = self.metadata[var].values
                    if np.isscalar(data):
                        stats_info["metadata"][var] = float(data) if np.isfinite(data) else str(data)
                    else:
                        stats_info["metadata"][var] = f"Array shape: {data.shape}"
                except:
                    stats_info["metadata"][var] = "Cannot display"
        
        return stats_info
    
    def normalize_input(self, data: np.ndarray, var_name: str) -> np.ndarray:
        """Normalize SAR input data using loaded statistics."""
        if self.stats is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        if var_name not in self.stats["input"]:
            raise ValueError(f"Variable {var_name} not found in input statistics")
        
        mean = self.stats["input"][var_name]["mean"]
        std = self.stats["input"][var_name]["std"]
        return (data - mean) / std
    
    def denormalize_output(self, data: np.ndarray, var_name: str = "dsm_height") -> np.ndarray:
        """Denormalize DSM height data using loaded statistics."""
        if self.stats is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        if var_name not in self.stats["output"]:
            raise ValueError(f"Variable {var_name} not found in output statistics")
        
        mean = self.stats["output"][var_name]["mean"]
        std = self.stats["output"][var_name]["std"]
        return (data * std) + mean
    
    def normalize_output(self, data: np.ndarray, var_name: str = "dsm_height") -> np.ndarray:
        """Normalize DSM height data using loaded statistics."""
        if self.stats is None:
            raise ValueError("Statistics not loaded. Call load_statistics() first.")
        
        if var_name not in self.stats["output"]:
            raise ValueError(f"Variable {var_name} not found in output statistics")
        
        mean = self.stats["output"][var_name]["mean"]
        std = self.stats["output"][var_name]["std"]
        return (data - mean) / std
    
    def prepare_input_tensor(self, 
                           selected_variables: Optional[List[str]] = None,
                           apply_normalization: bool = True) -> torch.Tensor:
        """
        Prepare input tensor for model inference.
        
        Args:
            selected_variables: List of input variables to use. If None, use all available.
            apply_normalization: If True, apply normalization using stats.json.
        
        Returns:
            Input tensor of shape (1, N_channels, H, W)
        """
        if self.input_data is None:
            raise ValueError("No data loaded. Call load_sample_data() first.")
        
        if selected_variables is None:
            selected_variables = self.input_variables
        
        # Validate selected variables
        for var in selected_variables:
            if var not in self.input_variables:
                raise ValueError(f"Variable {var} not found in loaded data. "
                               f"Available: {self.input_variables}")
        
        # Stack input variables
        input_stack = []
        for var in selected_variables:
            var_data = self.input_data[var].values
            
            if apply_normalization:
                var_processed = self.normalize_input(var_data, var)
            else:
                var_processed = var_data
            
            input_stack.append(var_processed)
        
        input_array = np.stack(input_stack, axis=0)  # Shape: (N_vars, H, W)
        input_tensor = torch.from_numpy(input_array).float().unsqueeze(0)  # (1, N_vars, H, W)
        
        print(f"✓ Input tensor prepared: {input_tensor.shape}")
        print(f"  - Variables: {selected_variables}")
        print(f"  - Normalization applied: {apply_normalization}")
        
        return input_tensor
    
    def get_ground_truth(self) -> np.ndarray:
        """
        Get ground truth DSM height data.
        
        Returns:
            Ground truth DSM height as numpy array
        """
        if self.output_data is None:
            raise ValueError("No data loaded. Call load_sample_data() first.")
        
        dsm_true = self.output_data["dsm_height"].values
        return dsm_true
    
    def get_patch_metadata(self) -> Dict[str, Any]:
        """Get metadata information for the loaded patch."""
        if self.metadata is None:
            return {"error": "No metadata loaded"}
        
        metadata_info = {}
        for var in self.metadata.data_vars:
            try:
                data = self.metadata[var].values
                if np.isscalar(data):
                    metadata_info[var] = float(data) if np.isfinite(data) else str(data)
                else:
                    metadata_info[var] = data.tolist() if data.size < 10 else f"Array({data.shape})"
            except:
                metadata_info[var] = "Cannot access"
        
        return metadata_info
    
    def get_available_files(self, data_dir: str) -> List[str]:
        """
        Get list of available preprocessed NetCDF files.
        
        Args:
            data_dir: Directory containing preprocessed files
            
        Returns:
            List of available file paths
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return []
        
        # Look for preprocessed NetCDF files
        nc_files = list(data_path.glob("preprocessed_patches_*.nc"))
        return [str(f) for f in sorted(nc_files)]
    
    def _generate_dummy_stats(self) -> Dict[str, Any]:
        """Generate dummy statistics for testing."""
        # Common SAR2Height variables
        input_variables = ["intensity_db", "intensity_percentile_rescaled"]
        output_variables = ["dsm_height"]
        
        stats = {
            "input": {},
            "output": {}
        }
        
        # Generate dummy input stats
        for var in input_variables:
            if "db" in var:
                stats["input"][var] = {"mean": -15.0, "std": 8.0}
            else:
                stats["input"][var] = {"mean": 0.5, "std": 0.3}
        
        # Generate dummy output stats
        for var in output_variables:
            stats["output"][var] = {"mean": 100.0, "std": 50.0}
        
        return stats
    
    def _generate_dummy_data(self) -> Tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
        """Generate dummy SAR2Height data for testing."""
        H, W = 432, 432
        
        # Create dummy input data (SAR features)
        input_data_dict = {
            "intensity_db": (["y_lr", "x_lr"], -20 + 10 * np.random.randn(H, W)),
            "intensity_percentile_rescaled": (["y_lr", "x_lr"], np.random.rand(H, W))
        }
        input_data = xr.Dataset(input_data_dict, coords={"y_lr": range(H), "x_lr": range(W)})
        
        # Create dummy output data (DSM height)
        output_data_dict = {
            "dsm_height": (["y_hr", "x_hr"], 50 + 100 * np.random.randn(H, W))
        }
        output_data = xr.Dataset(output_data_dict, coords={"y_hr": range(H), "x_hr": range(W)})
        
        # Create dummy metadata
        metadata_dict = {
            "patch_id": 0,
            "grid_row_start": 0,
            "grid_col_start": 0,
            "dsm_valid_percentage": 95.0,
            "dsm_mean": 100.0,
            "dsm_std": 50.0
        }
        metadata = xr.Dataset(metadata_dict)
        
        return input_data, output_data, metadata