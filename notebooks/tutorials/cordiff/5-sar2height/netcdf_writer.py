"""
NetCDF Writer Module for SAR-to-Height Data Processing

This module provides functionality to save ALL generated patches in NetCDF format
compatible with the cordiff training pipeline. It preserves complete spatial and
quality metadata for later filtering processes.

Author: AI Assistant
Date: 2026-01-07
"""

import numpy as np
import netCDF4 as nc
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class NetCDFPatchWriter:
    """
    Class to write ALL SAR-to-height patches in cordiff-compatible NetCDF format.
    
    This class saves complete patch information including:
    - Multi-channel SAR features as input
    - DSM data as output
    - Complete spatial metadata (CRS, bounds, polygons)
    - Raw quality components for later filtering
    - Proper cordiff data structure conventions
    """
    
    def __init__(self):
        """Initialize the NetCDF writer."""
        self.compression = {'zlib': True, 'complevel': 4}
    
    def save_all_patches_to_netcdf(self, 
                                  patches: List[Dict], 
                                  output_path: str,
                                  feature_names: List[str],
                                  base_name: str,
                                  source_metadata: Dict,
                                  global_attrs: Optional[Dict] = None) -> str:
        """
        Save ALL patches to NetCDF file in cordiff format with complete metadata.
        
        Args:
            patches (List[Dict]): List of ALL patch dictionaries (no filtering)
            output_path (str): Path to output NetCDF file
            feature_names (List[str]): Names of input features
            base_name (str): Base name of the original data
            source_metadata (Dict): Source data metadata
            global_attrs (Dict, optional): Additional global attributes
            
        Returns:
            str: Path to the created NetCDF file
        """
        if not patches:
            raise ValueError("No patches provided for writing")
        
        logger.info(f"Writing {len(patches)} patches to {output_path}")
        
        # Prepare data arrays
        input_data, output_data = self._prepare_data_arrays(patches)
        
        # Create NetCDF file
        with nc.Dataset(output_path, 'w', format='NETCDF4') as ds:
            # Create dimensions
            self._create_dimensions(ds, input_data, output_data)
            
            # Create coordinate variables
            self._add_coordinate_variables(ds, input_data.shape)
            
            # Create input group
            self._create_input_group(ds, input_data, feature_names)
            
            # Create output group
            self._create_output_group(ds, output_data)
            
            # Create metadata group with complete patch information
            self._create_metadata_group(ds, patches)
            
            # Add global attributes
            self._add_global_attributes(ds, base_name, source_metadata, patches, global_attrs)
        
        logger.info(f"Successfully wrote NetCDF file: {output_path}")
        return output_path
    
    def _prepare_data_arrays(self, patches: List[Dict]) -> tuple:
        """
        Prepare input and output data arrays from patches.
        
        Args:
            patches (List[Dict]): List of patch dictionaries
            
        Returns:
            Tuple containing input_data and output_data arrays
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
        output_data = np.zeros((n_patches, 1, dsm_height, dsm_width), dtype=np.float32)
        
        # Fill arrays
        for i, patch in enumerate(patches):
            input_data[i] = patch['features'].astype(np.float32)
            output_data[i, 0] = patch['dsm'].astype(np.float32)
        
        logger.info(f"Prepared data arrays: input {input_data.shape}, output {output_data.shape}")
        
        return input_data, output_data
    
    def _create_dimensions(self, ds: nc.Dataset, input_data: np.ndarray, output_data: np.ndarray):
        """Create NetCDF dimensions."""
        n_samples, n_input_channels, height, width = input_data.shape
        
        # Create dimensions following cordiff convention
        ds.createDimension('sample', n_samples)
        ds.createDimension('x_lr', width)   # Input resolution
        ds.createDimension('y_lr', height)
        ds.createDimension('x_hr', width)   # Output resolution (same as input for SAR-to-height)
        ds.createDimension('y_hr', height)
        
        logger.info(f"Created dimensions: sample={n_samples}, x_lr={width}, y_lr={height}")
    
    def _add_coordinate_variables(self, ds: nc.Dataset, data_shape: tuple):
        """Add coordinate variables for spatial dimensions."""
        n_samples, _, height, width = data_shape
        
        # Create sample coordinate
        sample_var = ds.createVariable('sample', 'i4', ('sample',))
        sample_var[:] = np.arange(n_samples)
        sample_var.setncattr('long_name', 'Sample index')
        sample_var.setncattr('description', 'Index of each patch sample')
        
        # Create spatial coordinates
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
            clean_name = self._clean_variable_name(feature_name)
            
            var = input_group.createVariable(
                clean_name, 
                'f4',
                ('sample', 'y_lr', 'x_lr'),
                **self.compression
            )
            
            var[:] = input_data[:, i, :, :]
            var.setncattr('long_name', feature_name)
            var.setncattr('description', f'SAR-derived feature: {feature_name}')
            var.setncattr('channel_index', i)
            
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
        
        # Create DSM height variable
        dsm_var = output_group.createVariable(
            'dsm_height',
            'f4',
            ('sample', 'y_hr', 'x_hr'),
            **self.compression
        )
        
        dsm_var[:] = output_data[:, 0, :, :]
        dsm_var.setncattr('long_name', 'Digital Surface Model Height')
        dsm_var.setncattr('description', 'Height values from radar-derived DSM')
        dsm_var.setncattr('units', 'meters')
        dsm_var.setncattr('standard_name', 'surface_altitude')
        
        # Add group attributes
        output_group.setncattr('description', 'Digital Surface Model height data')
        output_group.setncattr('n_channels', 1)
        
        logger.info("Created output group with DSM height variable")
    
    def _create_metadata_group(self, ds: nc.Dataset, patches: List[Dict]):
        """Create metadata group with complete patch information for filtering."""
        metadata_group = ds.createGroup('patch_metadata')
        n_patches = len(patches)
        
        # Create patch ID variable
        patch_id_var = metadata_group.createVariable('patch_id', 'i4', ('sample',))
        patch_id_var[:] = [p['patch_id'] for p in patches]
        patch_id_var.setncattr('description', 'Unique patch identifier')
        
        # Create grid position variables
        grid_row_var = metadata_group.createVariable('grid_row_start', 'i4', ('sample',))
        grid_col_var = metadata_group.createVariable('grid_col_start', 'i4', ('sample',))
        grid_row_var[:] = [p['grid_position'][0] for p in patches]
        grid_col_var[:] = [p['grid_position'][1] for p in patches]
        grid_row_var.setncattr('description', 'Starting row in original grid')
        grid_col_var.setncattr('description', 'Starting column in original grid')
        
        # Create spatial information variables
        if patches[0]['spatial_info']['has_geo_transform']:
            # Geographic bounds
            bounds_left_var = metadata_group.createVariable('bounds_left', 'f8', ('sample',))
            bounds_bottom_var = metadata_group.createVariable('bounds_bottom', 'f8', ('sample',))
            bounds_right_var = metadata_group.createVariable('bounds_right', 'f8', ('sample',))
            bounds_top_var = metadata_group.createVariable('bounds_top', 'f8', ('sample',))
            
            for i, patch in enumerate(patches):
                bounds = patch['spatial_info']['bounds']
                bounds_left_var[i] = bounds['left']
                bounds_bottom_var[i] = bounds['bottom']
                bounds_right_var[i] = bounds['right']
                bounds_top_var[i] = bounds['top']
            
            bounds_left_var.setncattr('description', 'Western boundary in CRS units')
            bounds_bottom_var.setncattr('description', 'Southern boundary in CRS units')
            bounds_right_var.setncattr('description', 'Eastern boundary in CRS units')
            bounds_top_var.setncattr('description', 'Northern boundary in CRS units')
            
            # Store CRS and polygon information as string variables
            crs_var = metadata_group.createVariable('crs', str, ('sample',), fill_value='')
            polygon_var = metadata_group.createVariable('polygon_wkt', str, ('sample',), fill_value='')
            
            for i, patch in enumerate(patches):
                spatial_info = patch['spatial_info']
                crs_var[i] = str(spatial_info.get('crs', ''))
                polygon_var[i] = spatial_info.get('polygon_wkt', '')
            
            crs_var.setncattr('description', 'Coordinate Reference System')
            polygon_var.setncattr('description', 'Patch boundary as WKT polygon')
        
        # Create quality component variables
        quality_components = patches[0]['quality_components']
        
        # DSM quality variables
        dsm_valid_count_var = metadata_group.createVariable('dsm_valid_count', 'i4', ('sample',))
        dsm_valid_pct_var = metadata_group.createVariable('dsm_valid_percentage', 'f4', ('sample',))
        dsm_mean_var = metadata_group.createVariable('dsm_mean', 'f4', ('sample',))
        dsm_std_var = metadata_group.createVariable('dsm_std', 'f4', ('sample',))
        dsm_min_var = metadata_group.createVariable('dsm_min', 'f4', ('sample',))
        dsm_max_var = metadata_group.createVariable('dsm_max', 'f4', ('sample',))
        dsm_range_var = metadata_group.createVariable('dsm_range', 'f4', ('sample',))
        
        for i, patch in enumerate(patches):
            qc = patch['quality_components']
            dsm_valid_count_var[i] = qc['dsm_valid_count']
            dsm_valid_pct_var[i] = qc['dsm_valid_percentage']
            dsm_mean_var[i] = qc['dsm_mean']
            dsm_std_var[i] = qc['dsm_std']
            dsm_min_var[i] = qc['dsm_min']
            dsm_max_var[i] = qc['dsm_max']
            dsm_range_var[i] = qc['dsm_range']
        
        # Add descriptions
        dsm_valid_count_var.setncattr('description', 'Number of valid DSM pixels')
        dsm_valid_pct_var.setncattr('description', 'Percentage of valid DSM pixels')
        dsm_mean_var.setncattr('description', 'Mean DSM value')
        dsm_std_var.setncattr('description', 'Standard deviation of DSM values')
        dsm_min_var.setncattr('description', 'Minimum DSM value')
        dsm_max_var.setncattr('description', 'Maximum DSM value')
        dsm_range_var.setncattr('description', 'Range of DSM values')
        
        # Feature quality variables
        n_features = len(quality_components['feature_valid_percentages'])
        feature_valid_pct_min_var = metadata_group.createVariable('feature_valid_percentage_min', 'f4', ('sample',))
        feature_valid_pct_mean_var = metadata_group.createVariable('feature_valid_percentage_mean', 'f4', ('sample',))
        min_data_completeness_var = metadata_group.createVariable('min_data_completeness', 'f4', ('sample',))
        
        for i, patch in enumerate(patches):
            qc = patch['quality_components']
            feature_valid_pct_min_var[i] = qc['feature_valid_percentage_min']
            feature_valid_pct_mean_var[i] = qc['feature_valid_percentage_mean']
            min_data_completeness_var[i] = qc['min_data_completeness']
        
        feature_valid_pct_min_var.setncattr('description', 'Minimum feature valid percentage across all channels')
        feature_valid_pct_mean_var.setncattr('description', 'Mean feature valid percentage across all channels')
        min_data_completeness_var.setncattr('description', 'Minimum data completeness (DSM and features)')
        
        # Add group attributes
        metadata_group.setncattr('description', 'Complete patch metadata for filtering')
        metadata_group.setncattr('n_patches', n_patches)
        
        logger.info(f"Created metadata group with complete patch information")
    
    def _add_global_attributes(self, ds: nc.Dataset, base_name: str, source_metadata: Dict, 
                             patches: List[Dict], global_attrs: Optional[Dict] = None):
        """Add global attributes to the NetCDF file."""
        # Basic attributes
        ds.setncattr('title', f'SAR-to-Height Complete Patch Data: {base_name}')
        ds.setncattr('description', 'Complete SAR-to-height patches with full metadata for filtering')
        ds.setncattr('source_data', base_name)
        ds.setncattr('creation_date', datetime.now().isoformat())
        ds.setncattr('created_by', 'SAR-to-Height Patch Generation Pipeline')
        ds.setncattr('data_type', 'sar_to_height_complete_patches')
        ds.setncattr('filtering_applied', 0)  # No filtering at this stage
        
        # Source metadata
        if source_metadata:
            for key, value in source_metadata.items():
                if isinstance(value, (str, int, float)):
                    ds.setncattr(f'source_{key}', value)
        
        # Patch statistics
        n_patches = len(patches)
        ds.setncattr('n_patches', n_patches)
        ds.setncattr('patch_size', patches[0]['patch_size'])
        
        # Quality statistics (raw, no filtering applied)
        quality_stats = self._calculate_quality_statistics(patches)
        for key, value in quality_stats.items():
            # Ensure proper data types for NetCDF
            if isinstance(value, bool):
                value = int(value)
            elif isinstance(value, np.bool_):
                value = int(value)
            ds.setncattr(f'quality_stat_{key}', value)
        
        # Spatial extent
        spatial_extent = self._calculate_spatial_extent(patches)
        if spatial_extent:
            for key, value in spatial_extent.items():
                ds.setncattr(f'spatial_{key}', value)
        
        # Add custom attributes if provided
        if global_attrs:
            for key, value in global_attrs.items():
                if isinstance(value, bool):
                    value = int(value)
                elif isinstance(value, np.bool_):
                    value = int(value)
                try:
                    ds.setncattr(key, value)
                except Exception as e:
                    logger.warning(f"Could not set attribute {key}: {str(e)}")
        
        # Cordiff compatibility attributes
        ds.setncattr('cordiff_compatible', 1)
        ds.setncattr('input_channels', patches[0]['features'].shape[0])
        ds.setncattr('output_channels', 1)
    
    def _calculate_quality_statistics(self, patches: List[Dict]) -> Dict:
        """Calculate overall quality statistics from all patches."""
        dsm_valid_pcts = [p['quality_components']['dsm_valid_percentage'] for p in patches]
        feature_valid_mins = [p['quality_components']['feature_valid_percentage_min'] for p in patches]
        dsm_stds = [p['quality_components']['dsm_std'] for p in patches]
        
        return {
            'mean_dsm_valid_percentage': float(np.mean(dsm_valid_pcts)),
            'min_dsm_valid_percentage': float(np.min(dsm_valid_pcts)),
            'max_dsm_valid_percentage': float(np.max(dsm_valid_pcts)),
            'mean_feature_valid_min': float(np.mean(feature_valid_mins)),
            'min_feature_valid_min': float(np.min(feature_valid_mins)),
            'mean_dsm_std': float(np.mean(dsm_stds)),
            'min_dsm_std': float(np.min(dsm_stds)),
            'max_dsm_std': float(np.max(dsm_stds))
        }
    
    def _calculate_spatial_extent(self, patches: List[Dict]) -> Optional[Dict]:
        """Calculate overall spatial extent if geographic information is available."""
        if not patches[0]['spatial_info']['has_geo_transform']:
            return None
        
        all_bounds = [p['spatial_info']['bounds'] for p in patches]
        
        return {
            'extent_left': min(b['left'] for b in all_bounds),
            'extent_bottom': min(b['bottom'] for b in all_bounds),
            'extent_right': max(b['right'] for b in all_bounds),
            'extent_top': max(b['top'] for b in all_bounds),
            'crs': patches[0]['spatial_info']['crs']
        }
    
    def _clean_variable_name(self, name: str) -> str:
        """Clean variable name for NetCDF compatibility."""
        clean_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in name)
        if clean_name and clean_name[0].isdigit():
            clean_name = 'var_' + clean_name
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


def save_all_patches_to_directory(patches: List[Dict], 
                                 output_dir: str,
                                 feature_names: List[str],
                                 base_name: str,
                                 source_metadata: Dict,
                                 global_attrs: Optional[Dict] = None) -> str:
    """
    Save ALL patches to a single NetCDF file in the specified directory.
    
    Args:
        patches (List[Dict]): Complete list of patches (no filtering)
        output_dir (str): Directory to save the NetCDF file
        feature_names (List[str]): Names of features
        base_name (str): Base name for the file
        source_metadata (Dict): Source data metadata
        global_attrs (Dict, optional): Additional global attributes
        
    Returns:
        str: Path to the created NetCDF file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    filename = f"full_patches_{base_name}_{len(patches)}patches.nc"
    output_path = output_dir / filename
    
    # Create writer and save
    writer = NetCDFPatchWriter()
    return writer.save_all_patches_to_netcdf(
        patches=patches,
        output_path=str(output_path),
        feature_names=feature_names,
        base_name=base_name,
        source_metadata=source_metadata,
        global_attrs=global_attrs
    )


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy patches with complete metadata
    dummy_patches = []
    for i in range(3):
        patch = {
            'patch_id': i,
            'features': np.random.randn(10, 432, 432),
            'dsm': np.random.randn(432, 432) * 100 + 500,
            'grid_position': (i*432, 0),
            'pixel_bounds': {'row_start': i*432, 'row_end': (i+1)*432, 'col_start': 0, 'col_end': 432},
            'spatial_info': {
                'has_geo_transform': False,
                'bounds': {'left': i*1000, 'bottom': i*1000, 'right': (i+1)*1000, 'top': (i+1)*1000},
                'polygon_wkt': f'POLYGON(({i*1000} {i*1000}, {(i+1)*1000} {i*1000}, {(i+1)*1000} {(i+1)*1000}, {i*1000} {(i+1)*1000}, {i*1000} {i*1000}))',
                'crs': None
            },
            'quality_components': {
                'dsm_valid_count': 432*432,
                'dsm_valid_percentage': 100.0,
                'dsm_mean': 500.0,
                'dsm_std': 100.0,
                'dsm_min': 300.0,
                'dsm_max': 700.0,
                'dsm_range': 400.0,
                'feature_valid_percentage_min': 99.0,
                'feature_valid_percentage_mean': 99.5,
                'min_data_completeness': 99.0
            },
            'patch_size': 432,
            'shape': (10, 432, 432),
            'source_metadata': {'satellite': 'ICEYE'}
        }
        dummy_patches.append(patch)
    
    # Save to NetCDF
    output_file = save_all_patches_to_directory(
        dummy_patches,
        '/tmp/full_patches',
        [f'feature_{i:02d}' for i in range(10)],
        'test_data',
        {'satellite': 'ICEYE', 'orbit_id': '12345'}
    )
    
    print("Created file:", output_file)