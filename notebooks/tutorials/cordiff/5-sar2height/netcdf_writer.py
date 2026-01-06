"""
NetCDF Writer Module for SAR-to-Height Data Processing

This module provides functionality to save processed patches in NetCDF format
compatible with the cordiff training pipeline, following the same structure
as the weather data in the 2-training.ipynb notebook.

Author: AI Assistant
Date: 2026-01-06
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NetCDFWriter:
    """
    Class to write SAR-to-height patches in cordiff-compatible NetCDF format.
    
    This class provides methods to:
    - Create NetCDF files with input/output groups
    - Write multi-channel SAR features as input
    - Write DSM data as output
    - Include proper metadata and attributes
    - Follow cordiff data structure conventions
    """
    
    def __init__(self):
        """Initialize the NetCDF writer."""
        self.compression = {'zlib': True, 'complevel': 4}
    
    def write_patches_to_netcdf(self, 
                               patches: List[Dict], 
                               output_path: str,
                               feature_names: List[str],
                               base_name: str,
                               aoi_info: Dict,
                               global_attrs: Optional[Dict] = None) -> str:
        """
        Write patches to NetCDF file in cordiff format.
        
        Args:
            patches (List[Dict]): List of patch dictionaries
            output_path (str): Path to output NetCDF file
            feature_names (List[str]): Names of input features
            base_name (str): Base name of the original data
            aoi_info (Dict): AOI information from original data
            global_attrs (Dict, optional): Additional global attributes
            
        Returns:
            str: Path to the created NetCDF file
        """
        if not patches:
            raise ValueError("No patches provided for writing")
        
        logger.info(f"Writing {len(patches)} patches to {output_path}")
        
        # Prepare data arrays
        input_data, output_data, patch_attrs = self._prepare_data_arrays(patches)
        
        # Create NetCDF file
        with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
            # Create dimensions
            self._create_dimensions(ds, input_data, output_data)
            
            # Create input group
            self._create_input_group(ds, input_data, feature_names)
            
            # Create output group
            self._create_output_group(ds, output_data)
            
            # Add global attributes
            self._add_global_attributes(ds, base_name, aoi_info, patch_attrs, global_attrs)
            
            # Add coordinate variables
            self._add_coordinate_variables(ds, input_data.shape)
        
        logger.info(f"Successfully wrote NetCDF file: {output_path}")
        return output_path
    
    def _prepare_data_arrays(self, patches: List[Dict]) -> tuple:
        """
        Prepare input and output data arrays from patches.
        
        Args:
            patches (List[Dict]): List of patch dictionaries
            
        Returns:
            Tuple containing input_data, output_data, and patch_attributes
        """
        n_patches = len(patches)
        
        # Get dimensions from first patch
        first_patch = patches[0]
        n_features, patch_height, patch_width = first_patch['features'].shape
        dsm_height, dsm_width = first_patch['dsm'].shape
        
        if (patch_height, patch_width) != (dsm_height, dsm_width):
            raise ValueError("Feature and DSM patch dimensions don't match")
        
        # Initialize arrays
        input_data = np.zeros((n_patches, n_features, patch_height, patch_width), dtype=np.float32)
        output_data = np.zeros((n_patches, 1, dsm_height, dsm_width), dtype=np.float32)  # 1 output channel (DSM)
        
        # Fill arrays and collect metadata
        patch_attrs = []
        for i, patch in enumerate(patches):
            input_data[i] = patch['features'].astype(np.float32)
            output_data[i, 0] = patch['dsm'].astype(np.float32)
            
            # Collect patch attributes
            patch_attr = {
                'patch_id': patch['patch_id'],
                'grid_position': patch['grid_position'],
                'bounds': patch['bounds'],
                'quality': patch['quality']
            }
            patch_attrs.append(patch_attr)
        
        logger.info(f"Prepared data arrays: input {input_data.shape}, output {output_data.shape}")
        
        return input_data, output_data, patch_attrs
    
    def _create_dimensions(self, ds: nc.Dataset, input_data: np.ndarray, output_data: np.ndarray):
        """Create NetCDF dimensions."""
        n_samples, n_input_channels, height, width = input_data.shape
        _, n_output_channels, _, _ = output_data.shape
        
        # Create dimensions following cordiff convention
        ds.createDimension('sample', n_samples)
        ds.createDimension('x_lr', width)   # Input resolution
        ds.createDimension('y_lr', height)
        ds.createDimension('x_hr', width)   # Output resolution (same as input for SAR-to-height)
        ds.createDimension('y_hr', height)
        
        logger.info(f"Created dimensions: sample={n_samples}, x_lr={width}, y_lr={height}")
    
    def _create_input_group(self, ds: nc.Dataset, input_data: np.ndarray, feature_names: List[str]):
        """Create input group with multi-channel SAR features."""
        input_group = ds.createGroup('input')
        
        n_samples, n_features, height, width = input_data.shape
        
        # Validate feature names
        if len(feature_names) != n_features:
            logger.warning(f"Feature names count ({len(feature_names)}) doesn't match "
                         f"data channels ({n_features}), generating default names")
            feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        
        # Create variables for each feature
        for i, feature_name in enumerate(feature_names):
            # Clean feature name for NetCDF compatibility
            clean_name = self._clean_variable_name(feature_name)
            
            var = input_group.createVariable(
                clean_name, 
                'f4',  # 32-bit float
                ('sample', 'y_lr', 'x_lr'),
                **self.compression
            )
            
            # Write data
            var[:] = input_data[:, i, :, :]
            
            # Add attributes
            var.setncattr('long_name', feature_name)
            var.setncattr('description', f'SAR-derived feature: {feature_name}')
            var.setncattr('channel_index', i)
            
            # Add units if known
            units = self._get_feature_units(feature_name)
            if units:
                var.setncattr('units', units)
        
        # Add group attributes
        input_group.setncattr('description', 'Multi-channel SAR intensity features')
        input_group.setncattr('n_channels', n_features)
        input_group.setncattr('feature_names', ','.join(feature_names))
        
        logger.info(f"Created input group with {n_features} feature variables")
    
    def _create_output_group(self, ds: nc.Dataset, output_data: np.ndarray):
        """Create output group with DSM height data."""
        output_group = ds.createGroup('output')
        
        n_samples, n_outputs, height, width = output_data.shape
        
        # Create DSM height variable
        dsm_var = output_group.createVariable(
            'dsm_height',
            'f4',  # 32-bit float
            ('sample', 'y_hr', 'x_hr'),
            **self.compression
        )
        
        # Write data
        dsm_var[:] = output_data[:, 0, :, :]
        
        # Add attributes
        dsm_var.setncattr('long_name', 'Digital Surface Model Height')
        dsm_var.setncattr('description', 'Height values from radar-derived DSM')
        dsm_var.setncattr('units', 'meters')
        dsm_var.setncattr('standard_name', 'surface_altitude')
        
        # Add group attributes
        output_group.setncattr('description', 'Digital Surface Model height data')
        output_group.setncattr('n_channels', 1)
        
        logger.info("Created output group with DSM height variable")
    
    def _add_coordinate_variables(self, ds: nc.Dataset, data_shape: tuple):
        """Add coordinate variables for spatial dimensions."""
        n_samples, _, height, width = data_shape
        
        # Create sample coordinate
        sample_var = ds.createVariable('sample', 'i4', ('sample',))
        sample_var[:] = np.arange(n_samples)
        sample_var.setncattr('long_name', 'Sample index')
        sample_var.setncattr('description', 'Index of each patch sample')
        
        # Create spatial coordinates (pixel coordinates)
        x_lr_var = ds.createVariable('x_lr', 'i4', ('x_lr',))
        x_lr_var[:] = np.arange(width)
        x_lr_var.setncattr('long_name', 'X coordinate (low resolution)')
        x_lr_var.setncattr('units', 'pixels')
        
        y_lr_var = ds.createVariable('y_lr', 'i4', ('y_lr',))
        y_lr_var[:] = np.arange(height)
        y_lr_var.setncattr('long_name', 'Y coordinate (low resolution)')
        y_lr_var.setncattr('units', 'pixels')
        
        x_hr_var = ds.createVariable('x_hr', 'i4', ('x_hr',))
        x_hr_var[:] = np.arange(width)
        x_hr_var.setncattr('long_name', 'X coordinate (high resolution)')
        x_hr_var.setncattr('units', 'pixels')
        
        y_hr_var = ds.createVariable('y_hr', 'i4', ('y_hr',))
        y_hr_var[:] = np.arange(height)
        y_hr_var.setncattr('long_name', 'Y coordinate (high resolution)')
        y_hr_var.setncattr('units', 'pixels')
    
    def _add_global_attributes(self, ds: nc.Dataset, base_name: str, aoi_info: Dict, 
                             patch_attrs: List[Dict], global_attrs: Optional[Dict] = None):
        """Add global attributes to the NetCDF file."""
        # Basic attributes
        ds.setncattr('title', f'SAR-to-Height Training Data: {base_name}')
        ds.setncattr('description', 'Multi-channel SAR features and DSM height data for cordiff training')
        ds.setncattr('source_data', base_name)
        ds.setncattr('creation_date', datetime.now().isoformat())
        ds.setncattr('created_by', 'SAR-to-Height Data Processing Pipeline')
        ds.setncattr('data_type', 'sar_to_height_patches')
        
        # AOI information
        if aoi_info:
            for key, value in aoi_info.items():
                if isinstance(value, (str, int, float)):
                    ds.setncattr(f'aoi_{key}', value)
        
        # Patch statistics
        n_patches = len(patch_attrs)
        ds.setncattr('n_patches', n_patches)
        
        # Quality statistics
        qualities = [p['quality']['overall_quality'] for p in patch_attrs]
        ds.setncattr('mean_patch_quality', np.mean(qualities))
        ds.setncattr('min_patch_quality', np.min(qualities))
        
        # Spatial bounds (if available)
        bounds_list = []
        for patch_attr in patch_attrs:
            bounds = patch_attr['bounds']
            if not bounds.get('pixel_bounds', False):
                bounds_list.append([bounds['left'], bounds['bottom'], bounds['right'], bounds['top']])
        
        if bounds_list:
            bounds_array = np.array(bounds_list)
            overall_bounds = [
                np.min(bounds_array[:, 0]),  # left
                np.min(bounds_array[:, 1]),  # bottom
                np.max(bounds_array[:, 2]),  # right
                np.max(bounds_array[:, 3])   # top
            ]
            ds.setncattr('spatial_bounds', ','.join(map(str, overall_bounds)))
        
        # Add custom attributes if provided
        if global_attrs:
            for key, value in global_attrs.items():
                try:
                    # Convert Python booleans to integers for NetCDF compatibility
                    if isinstance(value, bool):
                        value = int(value)
                    ds.setncattr(key, value)
                except Exception as e:
                    logger.warning(f"Could not set attribute {key}: {str(e)}")
        
        # Cordiff compatibility attributes - convert boolean to integer
        ds.setncattr('cordiff_compatible', 1)  # Use 1 instead of True
        ds.setncattr('patch_size', patch_attrs[0]['features'].shape[-1] if patch_attrs else 432)
        ds.setncattr('input_channels', len(patch_attrs[0]['features']) if patch_attrs else 0)
        ds.setncattr('output_channels', 1)
    
    def _clean_variable_name(self, name: str) -> str:
        """Clean variable name for NetCDF compatibility."""
        # Replace invalid characters with underscores
        clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        
        # Ensure it starts with a letter or underscore
        if clean_name and clean_name[0].isdigit():
            clean_name = 'var_' + clean_name
        
        # Limit length
        if len(clean_name) > 50:
            clean_name = clean_name[:50]
        
        return clean_name or 'unnamed_var'
    
    def _get_feature_units(self, feature_name: str) -> Optional[str]:
        """Get appropriate units for feature based on name."""
        feature_name_lower = feature_name.lower()
        
        if 'intensity' in feature_name_lower and 'db' in feature_name_lower:
            return 'dB'
        elif 'intensity' in feature_name_lower and ('linear' in feature_name_lower or 'sqrt' in feature_name_lower):
            return 'linear_units'
        elif 'angle' in feature_name_lower or 'direction' in feature_name_lower:
            return 'radians'
        elif 'mean' in feature_name_lower or 'std' in feature_name_lower:
            return 'dB'
        elif 'gradient' in feature_name_lower or 'sobel' in feature_name_lower:
            return 'dB/pixel'
        else:
            return None
    
    def write_skipped_patches(self, 
                             skipped_patches: List[Dict],
                             output_path: str,
                             feature_names: List[str],
                             base_name: str,
                             aoi_info: Dict,
                             skip_reason: str = "spatial_overlap") -> Optional[str]:
        """
        Write skipped patches to NetCDF file for record keeping.
        
        Args:
            skipped_patches (List[Dict]): List of skipped patch dictionaries
            output_path (str): Path to output NetCDF file
            feature_names (List[str]): Names of input features
            base_name (str): Base name of the original data
            aoi_info (Dict): AOI information from original data
            skip_reason (str): Reason why patches were skipped
            
        Returns:
            Optional[str]: Path to the created file or None if no patches
        """
        if not skipped_patches:
            logger.info("No skipped patches to write")
            return None
        
        logger.info(f"Writing {len(skipped_patches)} skipped patches to {output_path}")
        
        # Write using same format as regular patches
        output_file = self.write_patches_to_netcdf(
            skipped_patches, output_path, feature_names, base_name, aoi_info,
            global_attrs={'skip_reason': skip_reason, 'data_status': 'skipped'}
        )
        
        return output_file


def save_patches_to_netcdf(patches: List[Dict], 
                          output_path: str,
                          feature_names: List[str],
                          base_name: str,
                          aoi_info: Dict,
                          skipped_patches: Optional[List[Dict]] = None,
                          skipped_output_path: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to save patches and skipped patches to NetCDF files.
    
    Args:
        patches (List[Dict]): Valid patches to save
        output_path (str): Path for main NetCDF file
        feature_names (List[str]): Names of features
        base_name (str): Base name of original data
        aoi_info (Dict): AOI information
        skipped_patches (List[Dict], optional): Skipped patches
        skipped_output_path (str, optional): Path for skipped patches file
        
    Returns:
        Dict[str, str]: Dictionary with paths to created files
    """
    writer = NetCDFWriter()
    created_files = {}
    
    # Write main patches
    if patches:
        main_file = writer.write_patches_to_netcdf(
            patches, output_path, feature_names, base_name, aoi_info
        )
        created_files['main'] = main_file
    
    # Write skipped patches if provided
    if skipped_patches and skipped_output_path:
        skipped_file = writer.write_skipped_patches(
            skipped_patches, skipped_output_path, feature_names, base_name, aoi_info
        )
        if skipped_file:
            created_files['skipped'] = skipped_file
    
    return created_files


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy patches
    dummy_patches = []
    for i in range(3):
        patch = {
            'patch_id': i,
            'features': np.random.randn(10, 432, 432),  # 10 features
            'dsm': np.random.randn(432, 432) * 100 + 500,  # Height data
            'bounds': {'left': i*1000, 'bottom': i*1000, 'right': (i+1)*1000, 'top': (i+1)*1000, 'pixel_bounds': False},
            'grid_position': (i*432, 0),
            'quality': {'overall_quality': 0.8 + i*0.1}
        }
        dummy_patches.append(patch)
    
    # Feature names
    feature_names = [f'feature_{i:02d}' for i in range(10)]
    
    # Save to NetCDF
    created_files = save_patches_to_netcdf(
        dummy_patches,
        '/tmp/test_sar_patches.nc',
        feature_names,
        'test_data',
        {'satellite': 'ICEYE', 'orbit_id': '12345'}
    )
    
    print("Created files:", created_files)