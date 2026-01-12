"""
Coregistration Module for SAR-to-Height Data Processing

This module provides functionality to coregister SAR intensity and DSM data,
ensuring proper spatial alignment for the cordiff training pipeline.

Author: AI Assistant
Date: 2026-01-06
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from rasterio.transform import from_bounds
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CoregistrationProcessor:
    """
    Class to handle coregistration of SAR intensity and DSM data.
    
    This class provides methods to:
    - Align spatial grids between intensity and DSM data
    - Resample data to common resolution
    - Crop data to common extent
    - Validate spatial alignment
    """
    
    def __init__(self):
        """Initialize the coregistration processor."""
        self.resampling_method = Resampling.bilinear
    
    def check_alignment(self, intensity_meta: Dict, dsm_meta: Dict) -> Dict[str, bool]:
        """
        Check if intensity and DSM data are already aligned.
        
        Args:
            intensity_meta (Dict): Metadata from intensity TIFF
            dsm_meta (Dict): Metadata from DSM TIFF
            
        Returns:
            Dict containing alignment status
        """
        alignment = {
            'same_crs': False,
            'same_transform': False,
            'same_dimensions': False,
            'overlapping_bounds': False,
            'aligned': False
        }
        
        # Check CRS
        alignment['same_crs'] = intensity_meta['crs'] == dsm_meta['crs']
        
        # Check transform (pixel size and origin)
        int_transform = intensity_meta['transform']
        dsm_transform = dsm_meta['transform']
        
        # Compare with small tolerance for floating point differences
        transform_equal = (
            abs(int_transform.a - dsm_transform.a) < 1e-6 and
            abs(int_transform.e - dsm_transform.e) < 1e-6 and
            abs(int_transform.c - dsm_transform.c) < 1e-6 and
            abs(int_transform.f - dsm_transform.f) < 1e-6
        )
        alignment['same_transform'] = transform_equal
        
        # Check dimensions
        int_shape = (intensity_meta['height'], intensity_meta['width'])
        dsm_shape = (dsm_meta['height'], dsm_meta['width'])
        alignment['same_dimensions'] = int_shape == dsm_shape
        
        # Check if bounds overlap
        int_bounds = rasterio.transform.array_bounds(
            intensity_meta['height'], intensity_meta['width'], intensity_meta['transform']
        )
        dsm_bounds = rasterio.transform.array_bounds(
            dsm_meta['height'], dsm_meta['width'], dsm_meta['transform']
        )
        
        overlap = (
            max(int_bounds[0], dsm_bounds[0]) < min(int_bounds[2], dsm_bounds[2]) and
            max(int_bounds[1], dsm_bounds[1]) < min(int_bounds[3], dsm_bounds[3])
        )
        alignment['overlapping_bounds'] = overlap
        
        # Overall alignment
        alignment['aligned'] = all([
            alignment['same_crs'],
            alignment['same_transform'],
            alignment['same_dimensions'],
            alignment['overlapping_bounds']
        ])
        
        return alignment
    
    def calculate_common_grid(self, intensity_meta: Dict, dsm_meta: Dict) -> Dict:
        """
        Calculate common grid parameters for coregistration.
        
        Args:
            intensity_meta (Dict): Metadata from intensity TIFF
            dsm_meta (Dict): Metadata from DSM TIFF
            
        Returns:
            Dict containing common grid parameters
        """
        # Get bounds for both datasets
        int_bounds = rasterio.transform.array_bounds(
            intensity_meta['height'], intensity_meta['width'], intensity_meta['transform']
        )
        dsm_bounds = rasterio.transform.array_bounds(
            dsm_meta['height'], dsm_meta['width'], dsm_meta['transform']
        )
        
        # Calculate intersection bounds
        common_bounds = (
            max(int_bounds[0], dsm_bounds[0]),  # left
            max(int_bounds[1], dsm_bounds[1]),  # bottom
            min(int_bounds[2], dsm_bounds[2]),  # right
            min(int_bounds[3], dsm_bounds[3])   # top
        )
        
        # Choose finer resolution between the two datasets
        int_res_x = abs(intensity_meta['transform'].a)
        int_res_y = abs(intensity_meta['transform'].e)
        dsm_res_x = abs(dsm_meta['transform'].a)
        dsm_res_y = abs(dsm_meta['transform'].e)
        
        target_res_x = min(int_res_x, dsm_res_x)
        target_res_y = min(int_res_y, dsm_res_y)
        
        # Calculate target dimensions
        width = int((common_bounds[2] - common_bounds[0]) / target_res_x)
        height = int((common_bounds[3] - common_bounds[1]) / target_res_y)
        
        # Create target transform
        target_transform = from_bounds(
            common_bounds[0], common_bounds[1], common_bounds[2], common_bounds[3],
            width, height
        )
        
        # Use the CRS from intensity data (assuming it's the reference)
        target_crs = intensity_meta['crs']
        
        return {
            'bounds': common_bounds,
            'width': width,
            'height': height,
            'transform': target_transform,
            'crs': target_crs,
            'resolution': (target_res_x, target_res_y)
        }
    
    def coregister_data(self, 
                       intensity_data: np.ndarray, 
                       dsm_data: np.ndarray,
                       intensity_meta: Dict, 
                       dsm_meta: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Coregister intensity and DSM data to a common grid.
        
        Args:
            intensity_data (np.ndarray): SAR intensity data
            dsm_data (np.ndarray): DSM height data
            intensity_meta (Dict): Intensity metadata
            dsm_meta (Dict): DSM metadata
            
        Returns:
            Tuple containing:
            - coregistered_intensity: Resampled intensity data
            - coregistered_dsm: Resampled DSM data
            - common_meta: Metadata for the common grid
        """
        # Check if already aligned
        alignment = self.check_alignment(intensity_meta, dsm_meta)
        
        if alignment['aligned']:
            logger.info("Data already aligned, no coregistration needed")
            return intensity_data, dsm_data, intensity_meta
        
        # Calculate common grid
        common_grid = self.calculate_common_grid(intensity_meta, dsm_meta)
        
        logger.info(f"Coregistering to common grid: {common_grid['width']}x{common_grid['height']}")
        logger.info(f"Target resolution: {common_grid['resolution']}")
        
        # Prepare output arrays
        intensity_coregistered = np.zeros((common_grid['height'], common_grid['width']), dtype=intensity_data.dtype)
        dsm_coregistered = np.zeros((common_grid['height'], common_grid['width']), dtype=dsm_data.dtype)
        
        # Reproject intensity data
        reproject(
            source=intensity_data,
            destination=intensity_coregistered,
            src_transform=intensity_meta['transform'],
            src_crs=intensity_meta['crs'],
            dst_transform=common_grid['transform'],
            dst_crs=common_grid['crs'],
            resampling=self.resampling_method
        )
        
        # Reproject DSM data
        reproject(
            source=dsm_data,
            destination=dsm_coregistered,
            src_transform=dsm_meta['transform'],
            src_crs=dsm_meta['crs'],
            dst_transform=common_grid['transform'],
            dst_crs=common_grid['crs'],
            resampling=self.resampling_method
        )
        
        # Create common metadata
        common_meta = {
            'driver': 'GTiff',
            'dtype': intensity_data.dtype,
            'width': common_grid['width'],
            'height': common_grid['height'],
            'count': 1,
            'crs': common_grid['crs'],
            'transform': common_grid['transform'],
            'bounds': common_grid['bounds'],
            'resolution': common_grid['resolution']
        }
        
        logger.info(f"Coregistration complete. Output shape: {intensity_coregistered.shape}")
        
        return intensity_coregistered, dsm_coregistered, common_meta
    
    def crop_to_valid_region(self, 
                           intensity_data: np.ndarray, 
                           dsm_data: np.ndarray,
                           intensity_nodata: Optional[float] = None,
                           dsm_nodata: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int]]:
        """
        Crop data to region with valid data in both intensity and DSM.
        
        Args:
            intensity_data (np.ndarray): Intensity data
            dsm_data (np.ndarray): DSM data
            intensity_nodata (float, optional): No-data value for intensity
            dsm_nodata (float, optional): No-data value for DSM
            
        Returns:
            Tuple containing:
            - cropped_intensity: Cropped intensity data
            - cropped_dsm: Cropped DSM data
            - crop_bounds: (row_start, row_end, col_start, col_end)
        """
        # Create masks for valid data
        if intensity_nodata is not None:
            intensity_valid = (intensity_data != intensity_nodata) & np.isfinite(intensity_data)
        else:
            intensity_valid = np.isfinite(intensity_data)
        
        if dsm_nodata is not None:
            dsm_valid = (dsm_data != dsm_nodata) & np.isfinite(dsm_data)
        else:
            dsm_valid = np.isfinite(dsm_data)
        
        # Combined valid mask
        valid_mask = intensity_valid & dsm_valid
        
        # Find bounding box of valid data
        valid_rows, valid_cols = np.where(valid_mask)
        
        if len(valid_rows) == 0:
            raise ValueError("No overlapping valid data found")
        
        row_min, row_max = valid_rows.min(), valid_rows.max()
        col_min, col_max = valid_cols.min(), valid_cols.max()
        
        # Add small buffer if possible
        buffer = 5
        row_min = max(0, row_min - buffer)
        row_max = min(intensity_data.shape[0] - 1, row_max + buffer)
        col_min = max(0, col_min - buffer)
        col_max = min(intensity_data.shape[1] - 1, col_max + buffer)
        
        # Crop data
        cropped_intensity = intensity_data[row_min:row_max+1, col_min:col_max+1]
        cropped_dsm = dsm_data[row_min:row_max+1, col_min:col_max+1]
        
        crop_bounds = (row_min, row_max+1, col_min, col_max+1)
        
        logger.info(f"Cropped to valid region: {cropped_intensity.shape}")
        logger.info(f"Crop bounds: rows {row_min}-{row_max}, cols {col_min}-{col_max}")
        
        return cropped_intensity, cropped_dsm, crop_bounds
    
    def validate_coregistration(self, 
                              intensity_data: np.ndarray, 
                              dsm_data: np.ndarray,
                              meta: Dict) -> Dict[str, bool]:
        """
        Validate the coregistration results.
        
        Args:
            intensity_data (np.ndarray): Coregistered intensity data
            dsm_data (np.ndarray): Coregistered DSM data
            meta (Dict): Common metadata
            
        Returns:
            Dict containing validation results
        """
        validation = {
            'same_shape': False,
            'no_empty_data': False,
            'reasonable_overlap': False,
            'valid_metadata': False
        }
        
        # Check shapes
        validation['same_shape'] = intensity_data.shape == dsm_data.shape
        
        # Check for empty data
        intensity_valid = np.isfinite(intensity_data).sum()
        dsm_valid = np.isfinite(dsm_data).sum()
        total_pixels = intensity_data.size
        
        validation['no_empty_data'] = (intensity_valid > 0) and (dsm_valid > 0)
        
        # Check reasonable overlap (at least 50% valid data)
        min_valid = min(intensity_valid, dsm_valid)
        validation['reasonable_overlap'] = (min_valid / total_pixels) > 0.5
        
        # Check metadata validity
        required_keys = ['width', 'height', 'transform', 'crs', 'bounds']
        validation['valid_metadata'] = all(key in meta for key in required_keys)
        
        return validation


def coregister_file_pair(intensity_data: np.ndarray, 
                        dsm_data: np.ndarray,
                        intensity_meta: Dict, 
                        dsm_meta: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Convenience function to coregister a file pair.
    
    Args:
        intensity_data (np.ndarray): SAR intensity data
        dsm_data (np.ndarray): DSM height data
        intensity_meta (Dict): Intensity metadata
        dsm_meta (Dict): DSM metadata
        
    Returns:
        Tuple containing coregistered data and metadata
    """
    processor = CoregistrationProcessor()
    
    # Coregister to common grid
    intensity_coreg, dsm_coreg, common_meta = processor.coregister_data(
        intensity_data, dsm_data, intensity_meta, dsm_meta
    )
    
    # Crop to valid region
    intensity_cropped, dsm_cropped, crop_bounds = processor.crop_to_valid_region(
        intensity_coreg, dsm_coreg
    )
    
    # Update metadata for cropped data
    if crop_bounds[0] > 0 or crop_bounds[2] > 0:
        # Update transform for cropped region
        original_transform = common_meta['transform']
        new_origin_x = original_transform.c + crop_bounds[2] * original_transform.a
        new_origin_y = original_transform.f + crop_bounds[0] * original_transform.e
        
        new_transform = rasterio.transform.from_origin(
            new_origin_x, new_origin_y,
            abs(original_transform.a), abs(original_transform.e)
        )
        
        common_meta.update({
            'width': intensity_cropped.shape[1],
            'height': intensity_cropped.shape[0],
            'transform': new_transform
        })
    
    # Validate results
    validation = processor.validate_coregistration(intensity_cropped, dsm_cropped, common_meta)
    
    if not all(validation.values()):
        logger.warning(f"Coregistration validation issues: {validation}")
    
    return intensity_cropped, dsm_cropped, common_meta


if __name__ == "__main__":
    # Example usage would go here
    pass