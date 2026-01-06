"""
Patch Generation Module for SAR-to-Height Data Processing

This module provides functionality to create patches from coregistered SAR and DSM data
for the cordiff training pipeline, ensuring 432x432 patch size compatibility.

Author: AI Assistant
Date: 2026-01-06
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Generator
import logging

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    Class to generate patches from SAR features and DSM data.
    
    This class provides methods to:
    - Create non-overlapping patches of specified size
    - Handle edge cases and padding
    - Extract patch metadata (AOI bounds)
    - Validate patch quality
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
    
    def generate_patches(self, 
                        features: np.ndarray, 
                        dsm: np.ndarray, 
                        metadata: Dict) -> Generator[Dict, None, None]:
        """
        Generate patches from feature stack and DSM data.
        
        Args:
            features (np.ndarray): Multi-channel feature array (channels, height, width)
            dsm (np.ndarray): DSM height data (height, width)
            metadata (Dict): Metadata containing spatial information
            
        Yields:
            Dict: Patch information containing features, dsm, bounds, and quality metrics
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
                
                # Calculate patch bounds in original coordinate system
                patch_bounds = self._calculate_patch_bounds(
                    row_start, col_start, metadata
                )
                
                # Calculate quality metrics
                quality_metrics = self._assess_patch_quality(feature_patch, dsm_patch)
                
                # Create patch dictionary
                patch_info = {
                    'patch_id': patch_id,
                    'features': feature_patch,
                    'dsm': dsm_patch,
                    'bounds': patch_bounds,
                    'grid_position': (row_start, col_start),
                    'quality': quality_metrics,
                    'shape': feature_patch.shape,
                    'aoi_info': metadata.get('aoi_info', {})
                }
                
                logger.debug(f"Generated patch {patch_id} at ({row_start}, {col_start})")
                
                yield patch_info
                patch_id += 1
    
    def _calculate_patch_bounds(self, 
                              row_start: int, 
                              col_start: int, 
                              metadata: Dict) -> Dict[str, float]:
        """
        Calculate geographic bounds of a patch.
        
        Args:
            row_start (int): Starting row in pixel coordinates
            col_start (int): Starting column in pixel coordinates
            metadata (Dict): Metadata containing transform information
            
        Returns:
            Dict containing geographic bounds
        """
        if 'transform' not in metadata:
            # Return pixel coordinates if no transform available
            return {
                'left': col_start,
                'bottom': row_start + self.patch_size,
                'right': col_start + self.patch_size,
                'top': row_start,
                'pixel_bounds': True
            }
        
        transform = metadata['transform']
        
        # Calculate corner coordinates
        left = transform.c + col_start * transform.a
        top = transform.f + row_start * transform.e
        right = transform.c + (col_start + self.patch_size) * transform.a
        bottom = transform.f + (row_start + self.patch_size) * transform.e
        
        # Ensure proper ordering (left < right, bottom < top for geographic coords)
        if transform.e < 0:  # Standard case where y decreases with row
            top, bottom = bottom, top
        
        return {
            'left': min(left, right),
            'bottom': min(bottom, top),
            'right': max(left, right),
            'top': max(bottom, top),
            'pixel_bounds': False
        }
    
    def _assess_patch_quality(self, 
                            feature_patch: np.ndarray, 
                            dsm_patch: np.ndarray) -> Dict[str, float]:
        """
        Assess the quality of a patch based on data completeness and statistics.
        
        Args:
            feature_patch (np.ndarray): Feature patch data
            dsm_patch (np.ndarray): DSM patch data
            
        Returns:
            Dict containing quality metrics
        """
        quality = {}
        
        # Calculate valid data percentages
        total_pixels = dsm_patch.size
        
        # DSM quality metrics
        dsm_valid = np.isfinite(dsm_patch)
        dsm_valid_pct = np.sum(dsm_valid) / total_pixels * 100
        
        quality['dsm_valid_percentage'] = dsm_valid_pct
        quality['dsm_mean'] = np.nanmean(dsm_patch)
        quality['dsm_std'] = np.nanstd(dsm_patch)
        quality['dsm_range'] = np.nanmax(dsm_patch) - np.nanmin(dsm_patch)
        
        # Feature quality metrics
        feature_valid_pct = []
        for i in range(feature_patch.shape[0]):
            channel_valid = np.isfinite(feature_patch[i])
            channel_valid_pct = np.sum(channel_valid) / total_pixels * 100
            feature_valid_pct.append(channel_valid_pct)
        
        quality['feature_valid_percentage_min'] = min(feature_valid_pct)
        quality['feature_valid_percentage_mean'] = np.mean(feature_valid_pct)
        
        # Overall quality score (0-1, where 1 is best)
        min_valid_pct = min(dsm_valid_pct, min(feature_valid_pct))
        quality_score = min_valid_pct / 100.0
        
        # Penalize patches with very low variation (likely water or uniform areas)
        if quality['dsm_std'] < 0.1:  # Very flat areas
            quality_score *= 0.5
        
        quality['overall_quality'] = quality_score
        quality['is_valid'] = quality_score > 0.7  # At least 70% valid data
        
        return quality
    
    def filter_valid_patches(self, 
                           patches: List[Dict], 
                           min_quality: float = 0.7,
                           min_valid_pct: float = 70.0) -> List[Dict]:
        """
        Filter patches based on quality criteria.
        1. Data Completeness (70% weight)
                - Percentage of valid (non-NaN) pixels in SAR features
                - Percentage of valid (non-NaN) pixels in DSM data
                - Minimum of both percentages is used
        2.  Terrain Variation (30% weight)
            - DSM standard deviation > 0.1m (avoids flat areas)
            - Patches with very low variation get 50% penalty
            - Helps exclude water bodies, parking lots, etc.
        Why 0.7 threshold?
            - Ensures ≥70% valid pixels in training patches
            - Filters out corrupted/incomplete data
            - Maintains good terrain diversity
            - Balances data quality vs. quantity
         You can adjust this threshold by changing MIN_PATCH_QUALITY
            Lower values = more patches (but potentially lower quality)
            Higher values = fewer patches (but higher quality)
        
        Args:
            patches (List[Dict]): List of patch dictionaries
            min_quality (float): Minimum overall quality score
            min_valid_pct (float): Minimum percentage of valid pixels
            
        Returns:
            List[Dict]: Filtered patches meeting quality criteria
        """
    
             
        valid_patches = []
        
        for patch in patches:
            quality = patch['quality']
            
            # Check quality criteria
            passes_quality = quality['overall_quality'] >= min_quality
            passes_valid_pct = quality['dsm_valid_percentage'] >= min_valid_pct
            
            if passes_quality and passes_valid_pct:
                valid_patches.append(patch)
            else:
                logger.debug(f"Patch {patch['patch_id']} filtered out: "
                           f"quality={quality['overall_quality']:.2f}, "
                           f"valid_pct={quality['dsm_valid_percentage']:.1f}")
        
        logger.info(f"Filtered patches: {len(valid_patches)}/{len(patches)} passed quality criteria")
        
        return valid_patches
    
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


def generate_patches_from_data(features: np.ndarray, 
                             dsm: np.ndarray, 
                             metadata: Dict,
                             patch_size: int = 432,
                             stride: Optional[int] = None,
                             min_quality: float = 0.7) -> List[Dict]:
    """
    Convenience function to generate patches from feature and DSM data.
    
    Args:
        features (np.ndarray): Multi-channel feature array
        dsm (np.ndarray): DSM height data
        metadata (Dict): Spatial metadata
        patch_size (int): Patch size (default: 432)
        stride (int, optional): Stride between patches
        min_quality (float): Minimum quality threshold
        
    Returns:
        List[Dict]: List of valid patches
    """
    generator = PatchGenerator(patch_size, stride)
    
    # Pad data if necessary
    features_padded, features_padded_flag = generator.pad_data_if_needed(features)
    dsm_padded, dsm_padded_flag = generator.pad_data_if_needed(dsm)
    
    if features_padded_flag or dsm_padded_flag:
        logger.info("Data was padded to meet minimum patch size requirements")
    
    # Generate all patches
    all_patches = list(generator.generate_patches(features_padded, dsm_padded, metadata))
    
    # Filter valid patches
    valid_patches = generator.filter_valid_patches(all_patches, min_quality=min_quality)
    
    return valid_patches


if __name__ == "__main__":
    # Example usage
    # Create dummy data
    dummy_features = np.random.randn(10, 500, 500)  # 10 channels, 500x500
    dummy_dsm = np.random.randn(500, 500)
    dummy_metadata = {'transform': None}
    
    # Generate patches
    patches = generate_patches_from_data(
        dummy_features, 
        dummy_dsm, 
        dummy_metadata,
        patch_size=432
    )
    
    print(f"Generated {len(patches)} valid patches")
    if patches:
        print(f"First patch shape: {patches[0]['features'].shape}")
        print(f"First patch quality: {patches[0]['quality']['overall_quality']:.2f}")