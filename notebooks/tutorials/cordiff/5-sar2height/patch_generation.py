"""
Patch Generation Module for SAR-to-Height Data Processing

This module provides functionality to create patches from coregistered SAR and DSM data
for the cordiff training pipeline. It generates ALL patches without filtering and preserves
complete spatial and quality information for later processing.

Author: AI Assistant
Date: 2026-01-07
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Generator
import logging
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import Affine

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    Class to generate patches from SAR features and DSM data.
    
    This class provides methods to:
    - Create non-overlapping patches of specified size
    - Handle edge cases and padding
    - Extract complete patch metadata including spatial bounds and CRS
    - Calculate raw quality metrics without filtering
    """
    
    def __init__(self, patch_size: int = 432, stride: Optional[int] = None):
        """
        Initialize the patch generator.
        
        Args:
            patch_size (int): Size of square patches (default: 432 for cordiff)
            stride (int, optional): Stride between patches. If None, uses patch_size (no overlap)
        """
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        
    def calculate_patch_grid(self, data_shape: Tuple[int, int]) -> Tuple[List[int], List[int], int]:
        """
        Calculate the grid of patch positions for given data shape.
        
        Args:
            data_shape (Tuple[int, int]): Shape of the input data (height, width)
            
        Returns:
            Tuple containing:
            - row_starts: List of row start positions
            - col_starts: List of column start positions
            - n_patches: Total number of patches
        """
        height, width = data_shape
        
        # Calculate patch positions
        row_starts = list(range(0, height - self.patch_size + 1, self.stride))
        col_starts = list(range(0, width - self.patch_size + 1, self.stride))
        
        # If the last patch doesn't fit perfectly, add it
        if len(row_starts) == 0 or row_starts[-1] + self.patch_size < height:
            if height >= self.patch_size:
                row_starts.append(height - self.patch_size)
        
        if len(col_starts) == 0 or col_starts[-1] + self.patch_size < width:
            if width >= self.patch_size:
                col_starts.append(width - self.patch_size)
        
        # Remove duplicates while preserving order
        row_starts = list(dict.fromkeys(row_starts))
        col_starts = list(dict.fromkeys(col_starts))
        
        n_patches = len(row_starts) * len(col_starts)
        
        logger.info(f"Patch grid: {len(row_starts)} rows × {len(col_starts)} cols = {n_patches} patches")
        logger.info(f"Data shape: {data_shape}, Patch size: {self.patch_size}, Stride: {self.stride}")
        
        return row_starts, col_starts, n_patches
    
    def extract_patch(self, data: np.ndarray, row_start: int, col_start: int) -> np.ndarray:
        """
        Extract a single patch from data array.
        
        Args:
            data (np.ndarray): Input data array
            row_start (int): Starting row position
            col_start (int): Starting column position
            
        Returns:
            np.ndarray: Extracted patch
        """
        row_end = row_start + self.patch_size
        col_end = col_start + self.patch_size
        
        # Handle different data dimensions
        if data.ndim == 2:
            patch = data[row_start:row_end, col_start:col_end]
        elif data.ndim == 3:
            patch = data[:, row_start:row_end, col_start:col_end]
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
        
        return patch
    
    def generate_all_patches(self, 
                           features: np.ndarray, 
                           dsm: np.ndarray, 
                           metadata: Dict) -> Generator[Dict, None, None]:
        """
        Generate ALL patches from feature stack and DSM data without any filtering.
        
        Args:
            features (np.ndarray): Multi-channel feature array (channels, height, width)
            dsm (np.ndarray): DSM height data (height, width)
            metadata (Dict): Metadata containing spatial information
            
        Yields:
            Dict: Patch information with complete spatial and quality metadata
        """
        if features.shape[1:] != dsm.shape:
            raise ValueError(f"Feature and DSM shapes don't match: {features.shape[1:]} vs {dsm.shape}")
        
        # Calculate patch grid
        row_starts, col_starts, n_patches = self.calculate_patch_grid(dsm.shape)
        
        patch_id = 0
        for row_start in row_starts:
            for col_start in col_starts:
                # Extract patches
                feature_patch = self.extract_patch(features, row_start, col_start)
                dsm_patch = self.extract_patch(dsm, row_start, col_start)
                
                # Calculate complete spatial information
                spatial_info = self._calculate_spatial_info(row_start, col_start, metadata)
                
                # Calculate raw quality metrics
                quality_components = self._calculate_quality_components(feature_patch, dsm_patch)
                
                # Create comprehensive patch dictionary
                patch_info = {
                    'patch_id': patch_id,
                    'features': feature_patch,
                    'dsm': dsm_patch,
                    'grid_position': (row_start, col_start),
                    'pixel_bounds': {
                        'row_start': row_start,
                        'row_end': row_start + self.patch_size,
                        'col_start': col_start,
                        'col_end': col_start + self.patch_size
                    },
                    'spatial_info': spatial_info,
                    'quality_components': quality_components,
                    'patch_size': self.patch_size,
                    'shape': feature_patch.shape,
                    'source_metadata': metadata.get('aoi_info', {})
                }
                
                logger.debug(f"Generated patch {patch_id} at ({row_start}, {col_start})")
                
                yield patch_info
                patch_id += 1
    
    def _calculate_spatial_info(self, 
                              row_start: int, 
                              col_start: int, 
                              metadata: Dict) -> Dict:
        """
        Calculate complete spatial information for a patch including CRS and polygon.
        
        Args:
            row_start (int): Starting row in pixel coordinates
            col_start (int): Starting column in pixel coordinates
            metadata (Dict): Metadata containing transform and CRS information
            
        Returns:
            Dict containing complete spatial information
        """
        spatial_info = {
            'crs': None,
            'bounds': None,
            'polygon_wkt': None,
            'transform': None,
            'has_geo_transform': False
        }
        
        # Check if we have geospatial transform
        if 'transform' in metadata and metadata['transform'] is not None:
            transform = metadata['transform']
            crs = metadata.get('crs', None)
            
            # Calculate corner coordinates in the CRS
            x_min = transform.c + col_start * transform.a
            y_max = transform.f + row_start * transform.e
            x_max = transform.c + (col_start + self.patch_size) * transform.a
            y_min = transform.f + (row_start + self.patch_size) * transform.e
            
            # Ensure proper ordering
            if transform.e < 0:  # Standard case where y decreases with row
                y_min, y_max = y_max, y_min
            
            # Create bounds dictionary
            bounds = {
                'left': min(x_min, x_max),
                'bottom': min(y_min, y_max),
                'right': max(x_min, x_max),
                'top': max(y_min, y_max)
            }
            
            # Create polygon geometry
            polygon = Polygon([
                (bounds['left'], bounds['bottom']),
                (bounds['right'], bounds['bottom']),
                (bounds['right'], bounds['top']),
                (bounds['left'], bounds['top']),
                (bounds['left'], bounds['bottom'])
            ])
            
            spatial_info.update({
                'crs': crs,
                'bounds': bounds,
                'polygon_wkt': polygon.wkt,
                'transform': transform,
                'has_geo_transform': True
            })
            
        else:
            # Use pixel coordinates as fallback
            bounds = {
                'left': col_start,
                'bottom': row_start + self.patch_size,
                'right': col_start + self.patch_size,
                'top': row_start
            }
            
            polygon = Polygon([
                (bounds['left'], bounds['bottom']),
                (bounds['right'], bounds['bottom']),
                (bounds['right'], bounds['top']),
                (bounds['left'], bounds['top']),
                (bounds['left'], bounds['bottom'])
            ])
            
            spatial_info.update({
                'bounds': bounds,
                'polygon_wkt': polygon.wkt,
                'has_geo_transform': False
            })
            
        return spatial_info
    
    def _calculate_quality_components(self, 
                                    feature_patch: np.ndarray, 
                                    dsm_patch: np.ndarray) -> Dict:
        """
        Calculate raw quality components without applying any filtering logic.
        
        Args:
            feature_patch (np.ndarray): Feature patch data
            dsm_patch (np.ndarray): DSM patch data
            
        Returns:
            Dict containing raw quality components
        """
        components = {}
        
        # Basic patch information
        total_pixels = dsm_patch.size
        components['total_pixels'] = total_pixels
        
        # DSM quality components
        dsm_valid = np.isfinite(dsm_patch)
        dsm_valid_count = np.sum(dsm_valid)
        dsm_valid_pct = (dsm_valid_count / total_pixels) * 100
        
        components['dsm_valid_count'] = int(dsm_valid_count)
        components['dsm_valid_percentage'] = float(dsm_valid_pct)
        components['dsm_invalid_count'] = int(total_pixels - dsm_valid_count)
        
        # DSM statistics (only for valid pixels)
        if dsm_valid_count > 0:
            valid_dsm = dsm_patch[dsm_valid]
            components['dsm_mean'] = float(np.mean(valid_dsm))
            components['dsm_std'] = float(np.std(valid_dsm))
            components['dsm_min'] = float(np.min(valid_dsm))
            components['dsm_max'] = float(np.max(valid_dsm))
            components['dsm_range'] = float(np.max(valid_dsm) - np.min(valid_dsm))
            components['dsm_median'] = float(np.median(valid_dsm))
        else:
            components['dsm_mean'] = 0.0
            components['dsm_std'] = 0.0
            components['dsm_min'] = 0.0
            components['dsm_max'] = 0.0
            components['dsm_range'] = 0.0
            components['dsm_median'] = 0.0
        
        # Feature quality components (per channel)
        n_channels = feature_patch.shape[0]
        feature_valid_counts = []
        feature_valid_percentages = []
        
        for i in range(n_channels):
            channel_data = feature_patch[i]
            channel_valid = np.isfinite(channel_data)
            channel_valid_count = np.sum(channel_valid)
            channel_valid_pct = (channel_valid_count / total_pixels) * 100
            
            feature_valid_counts.append(int(channel_valid_count))
            feature_valid_percentages.append(float(channel_valid_pct))
        
        components['feature_valid_counts'] = feature_valid_counts
        components['feature_valid_percentages'] = feature_valid_percentages
        components['feature_valid_percentage_min'] = float(min(feature_valid_percentages))
        components['feature_valid_percentage_max'] = float(max(feature_valid_percentages))
        components['feature_valid_percentage_mean'] = float(np.mean(feature_valid_percentages))
        
        # Additional useful metrics for quality assessment
        components['min_data_completeness'] = float(min(dsm_valid_pct, min(feature_valid_percentages)))
        components['has_sufficient_variation'] = float(components['dsm_std']) > 0.1  # Can be used later for filtering
        
        return components
    
    def pad_data_if_needed(self, data: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Pad data to minimum patch size if necessary.
        
        Args:
            data (np.ndarray): Input data array
            
        Returns:
            Tuple containing padded data and padding flag
        """
        if data.ndim == 2:
            height, width = data.shape
        elif data.ndim == 3:
            _, height, width = data.shape
        else:
            raise ValueError(f"Unsupported data dimensions: {data.ndim}")
        
        if height >= self.patch_size and width >= self.patch_size:
            return data, False
        
        # Calculate padding needed
        pad_height = max(0, self.patch_size - height)
        pad_width = max(0, self.patch_size - width)
        
        if data.ndim == 2:
            # Pad with edge values (reflect mode)
            padded_data = np.pad(
                data, 
                ((0, pad_height), (0, pad_width)), 
                mode='reflect'
            )
        else:
            # Pad with edge values for multi-channel data
            padded_data = np.pad(
                data, 
                ((0, 0), (0, pad_height), (0, pad_width)), 
                mode='reflect'
            )
        
        logger.info(f"Padded data from {data.shape} to {padded_data.shape}")
        
        return padded_data, True


def generate_all_patches_from_data(features: np.ndarray, 
                                  dsm: np.ndarray, 
                                  metadata: Dict,
                                  patch_size: int = 432,
                                  stride: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to generate ALL patches from feature and DSM data without filtering.
    
    Args:
        features (np.ndarray): Multi-channel feature array
        dsm (np.ndarray): DSM height data
        metadata (Dict): Spatial metadata including CRS and transform
        patch_size (int): Patch size (default: 432)
        stride (int, optional): Stride between patches
        
    Returns:
        List[Dict]: List of ALL generated patches with complete metadata
    """
    generator = PatchGenerator(patch_size, stride)
    
    # Pad data if necessary
    features_padded, features_padded_flag = generator.pad_data_if_needed(features)
    dsm_padded, dsm_padded_flag = generator.pad_data_if_needed(dsm)
    
    if features_padded_flag or dsm_padded_flag:
        logger.info("Data was padded to meet minimum patch size requirements")
    
    # Generate ALL patches
    all_patches = list(generator.generate_all_patches(features_padded, dsm_padded, metadata))
    
    logger.info(f"Generated {len(all_patches)} total patches (no filtering applied)")
    
    return all_patches


if __name__ == "__main__":
    # Example usage
    # Create dummy data
    dummy_features = np.random.randn(10, 500, 500)  # 10 channels, 500x500
    dummy_dsm = np.random.randn(500, 500)
    dummy_metadata = {
        'transform': None,
        'crs': 'EPSG:4326',
        'aoi_info': {'satellite': 'ICEYE'}
    }
    
    # Generate ALL patches
    patches = generate_all_patches_from_data(
        dummy_features, 
        dummy_dsm, 
        dummy_metadata,
        patch_size=432
    )
    
    print(f"Generated {len(patches)} total patches")
    if patches:
        print(f"First patch shape: {patches[0]['features'].shape}")
        print(f"First patch spatial info: {patches[0]['spatial_info']}")
        print(f"First patch quality components: {patches[0]['quality_components']}")