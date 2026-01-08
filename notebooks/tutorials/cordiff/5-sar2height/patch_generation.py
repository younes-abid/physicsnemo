"""
Patch Generation Module for SAR-to-Height Data Processing

This module provides functionality to create patches from coregistered SAR and DSM data
for the cordiff training pipeline. It generates ALL patches without filtering and preserves
complete spatial and quality information for later processing.

Author: AI Assistant
Date: 2026-01-07
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Generator, Any, Union
import logging
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import Affine
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import multiprocessing as mp

from data_loader import SARDataLoader

logger = logging.getLogger(__name__)


@dataclass
class GenerationResults:
    """Results from full patch generation phase."""
    success: bool
    files_processed: int
    files_successful: int
    files_failed: int
    files_skipped: int
    total_patches_generated: int
    output_files: List[str]
    generation_time: float
    failed_files: List[str]
    error: Optional[str] = None
    
    @property
    def total_processing_time(self) -> float:
        """Alias for generation_time for backward compatibility."""
        return self.generation_time


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


# Global worker function for multiprocessing
def _generate_patches_worker(args):
    """
    Worker function for multiprocessing patch generation.
    
    Args:
        args: Tuple containing (pair_index, config, file_pairs_info, features, patch_size, save_viz)
        
    Returns:
        Dict: Processing result with success status and details
    """
    pair_index, config_dict, file_pairs_info, features, patch_size, save_viz = args
    
    # Import required modules in worker process
    try:
        from data_loader import SARDataLoader
        from coregistration import coregister_file_pair
        from feature_extraction import SARFeatureExtractor
        from netcdf_writer import save_all_patches_to_directory
    except ImportError as e:
        return {
            'pair_index': pair_index,
            'success': False,
            'error': f"Import error in worker: {str(e)}",
            'n_patches_generated': 0,
            'output_files': {},
            'processing_time': 0,
            'pair_name': f"unknown_{pair_index}"
        }
    
    # Reconstruct basic config info needed for processing
    raw_data_dir = config_dict['raw_data_dir']
    full_patches_dir = config_dict['full_patches_dir']
    override_existing = config_dict.get('override_existing', False)
    
    # Initialize data loader in worker process
    data_loader = SARDataLoader(raw_data_dir)
    
    # Get pair info
    pair_info = file_pairs_info[pair_index]
    base_name = pair_info['base_name']
    aoi_info = pair_info['aoi']
    
    start_time = time.time()
    result = {
        'pair_index': pair_index,
        'pair_name': base_name,
        'success': False,
        'error': None,
        'n_patches_generated': 0,
        'output_files': {},
        'processing_time': 0
    }
    
    try:
        # Check if output file already exists and skip if override is False
        existing_files = list(Path(full_patches_dir).glob(f"full_patches_{base_name}_*patches.nc"))
        
        if existing_files and not override_existing:
            existing_file = existing_files[0]
            result['success'] = True
            result['n_patches_generated'] = 0
            result['output_files'] = {'full_patches': str(existing_file)}
            result['processing_time'] = time.time() - start_time
            result['error'] = 'SKIPPED - Output file exists and override_existing=False'
            return result
        
        # Step 1: Load data
        intensity_data, dsm_data, intensity_meta, dsm_meta = data_loader.load_file_pair(pair_index)
        
        # Step 2: Coregister data
        intensity_coreg, dsm_coreg, common_meta = coregister_file_pair(
            intensity_data, dsm_data, intensity_meta, dsm_meta
        )
        
        # Step 3: Extract features
        feature_extractor = SARFeatureExtractor()
        feature_data, feature_names = feature_extractor.extract_selected_features(
            intensity_coreg, selected_features=features, metadata=common_meta
        )
        
        # Step 4: Generate patches
        all_patches = generate_all_patches_from_data(
            features=feature_data,
            dsm=dsm_coreg,
            metadata=common_meta,
            patch_size=patch_size,
            stride=None
        )
        
        result['n_patches_generated'] = len(all_patches)
        
        if not all_patches:
            result['error'] = "No patches could be generated (insufficient data size)"
            return result
        
        # Step 5: Save patches
        source_metadata = {
            'aoi_info': aoi_info,
            'original_intensity_shape': intensity_data.shape,
            'original_dsm_shape': dsm_data.shape,
            'coregistered_shape': intensity_coreg.shape,
            'n_features_extracted': len(feature_names),
            'patch_size': patch_size
        }
        
        output_file = save_all_patches_to_directory(
            patches=all_patches,
            output_dir=full_patches_dir,
            feature_names=feature_names,
            base_name=base_name,
            source_metadata=source_metadata,
            global_attrs={
                'processing_date': datetime.now().isoformat(),
                'enabled_features': ','.join(features),
                'filtering_applied': False
            }
        )
        
        result['output_files'] = {'full_patches': output_file}
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
    
    return result


def generate_full_patches_pipeline(data_loader, 
                                  config_dict: Dict,
                                  files_to_process: List[int],
                                  features: List[str],
                                  patch_size: int = 432,
                                  save_viz: bool = False) -> GenerationResults:
    """
    Main function to generate full patches from file pairs using multiprocessing.
    
    Args:
        data_loader: SAR data loader instance
        config_dict: Configuration dictionary
        files_to_process: List of file pair indices to process
        features: List of feature names to extract
        patch_size: Size of patches to generate
        save_viz: Whether to save visualizations
        
    Returns:
        GenerationResults: Results of patch generation
    """
    start_time = time.time()
    
    try:
        # Determine processing mode
        use_multiprocessing = config_dict.get('use_multiprocessing', True)
        max_workers = config_dict.get('max_workers', psutil.cpu_count(logical=True))
        use_mp = use_multiprocessing and len(files_to_process) > 1
        n_workers = min(max_workers, len(files_to_process)) if use_mp else 1
        
        print(f"🚀 Processing {len(files_to_process)} file pairs...")
        print(f"  • Patch size: {patch_size}×{patch_size}")
        print(f"  • Features: {len(features)} enabled")
        print(f"  • Visualizations: {'ON' if save_viz else 'OFF'}")
        print(f"  • Multiprocessing: {'ON' if use_mp else 'OFF'}")
        if use_mp:
            print(f"  • Workers: {n_workers}/{psutil.cpu_count(logical=True)} CPU cores")
            print(f"  • Available memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print("-" * 60)
        
        # Process files - use mutable containers for statistics
        output_files = []
        stats = {'successful_files': 0, 'total_patches_generated': 0}
        failed_files = []
        
        if use_mp:
            # Multiprocessing mode
            _generate_patches_with_multiprocessing(
                data_loader, config_dict, files_to_process, features, patch_size, save_viz, n_workers,
                output_files, stats, failed_files
            )
        else:
            # Sequential mode
            _generate_patches_sequentially(
                data_loader, config_dict, files_to_process, features, patch_size, save_viz,
                output_files, stats, failed_files
            )
        
        # Create and return results
        total_time = time.time() - start_time
        
        # Get skipped count from stats (set by processing methods)
        files_skipped = stats.get('skipped_files', 0)
        
        return GenerationResults(
            success=len(failed_files) == 0,
            files_processed=len(files_to_process),
            files_successful=stats['successful_files'],
            files_failed=len(failed_files),
            files_skipped=files_skipped,
            total_patches_generated=stats['total_patches_generated'],
            output_files=output_files,
            generation_time=total_time,
            failed_files=failed_files,
            error=None if len(failed_files) == 0 else f"Failed to process {len(failed_files)} files"
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        error_msg = f"Patch generation failed: {str(e)}"
        logger.error(error_msg)
        
        return GenerationResults(
            success=False,
            files_processed=0,
            files_successful=0,
            files_failed=0,
            files_skipped=0,
            total_patches_generated=0,
            output_files=[],
            generation_time=generation_time,
            failed_files=[],
            error=error_msg
        )


def _generate_patches_with_multiprocessing(data_loader: SARDataLoader, config_dict: Dict, files_to_process: List[int], features: List[str], patch_size: int, save_viz: bool, n_workers: int,
                                output_files: List[str], stats: Dict[str, int], failed_files: List[str]):
    """Generate patches using multiprocessing."""
    print(f"🔄 Starting multiprocessing with {n_workers} workers...")
    
    # Prepare arguments for worker processes
    file_pairs_info = [data_loader.file_pairs[i] for i in files_to_process]
    
    worker_args = [
        (i, config_dict, file_pairs_info, features, patch_size, save_viz) 
        for i in files_to_process
    ]
    
    # Track progress - add skipped_files counter
    completed = 0
    skipped_files = 0
    start_time = time.time()
    
    # Use ProcessPoolExecutor for better control and error handling
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_index = {
            executor.submit(_generate_patches_worker, args): args[0] 
            for args in worker_args
        }
        
        # Process completed jobs as they finish
        for future in as_completed(future_to_index):
            pair_index = future_to_index[future]
            completed += 1
            
            try:
                result = future.result()
                pair_name = result['pair_name']
                
                # Update progress
                elapsed = time.time() - start_time
                eta = elapsed * (len(files_to_process) - completed) / completed if completed > 0 else 0
                progress = completed / len(files_to_process) * 100
                
                print(f"\n🔄 [{completed}/{len(files_to_process)}] ({progress:.1f}%) {pair_name}")
                print(f"    ETA: {eta:.0f}s | Elapsed: {elapsed:.0f}s")
                
                if result['success']:
                    # Check if file was skipped or actually processed
                    if result.get('error') == 'SKIPPED - Output file exists and override_existing=False':
                        skipped_files += 1
                        print(f"    ⏭️  Skipped (file exists)")
                    else:
                        stats['successful_files'] += 1
                        stats['total_patches_generated'] += result['n_patches_generated']
                        print(f"    ✅ Success: {result['n_patches_generated']} patches")
                        print(f"    ⏱️  Processing: {result['processing_time']:.1f}s")
                    
                    # Add output files regardless of skip status
                    if result['output_files']:
                        output_files.extend(result['output_files'].values())
                        
                        # Show output files
                        for file_type, file_path in result['output_files'].items():
                            try:
                                file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
                                print(f"    💾 Output: {Path(file_path).name} ({file_size:.1f} MB)")
                            except:
                                print(f"    💾 Output: {Path(file_path).name}")
                else:
                    failed_files.append(pair_name)
                    error_msg = result['error'] or "Unknown error"
                    print(f"    ❌ Failed: {error_msg[:80]}..." if len(error_msg) > 80 else f"    ❌ Failed: {error_msg}")
                    
            except Exception as e:
                pair_name = f"pair_{pair_index}"
                failed_files.append(pair_name)
                print(f"\n❌ Critical error processing {pair_name}: {str(e)[:80]}...")
    
    # Store skipped count in stats for access by calling method
    stats['skipped_files'] = skipped_files
    print(f"\n✅ Multiprocessing completed in {time.time() - start_time:.1f}s")


def _generate_patches_sequentially(data_loader: SARDataLoader, config_dict: Dict, files_to_process: List[int], features: List[str], patch_size: int, save_viz: bool,
                        output_files: List[str], stats: Dict[str, int], failed_files: List[str]):
    """Generate patches sequentially (original behavior)."""
    print("🔄 Processing files sequentially...")
    
    # Import processing modules
    from coregistration import coregister_file_pair
    from feature_extraction import SARFeatureExtractor
    from netcdf_writer import save_all_patches_to_directory
    
    skipped_files = 0
    
    for i, pair_idx in enumerate(files_to_process):
        pair_name = data_loader.file_pairs[pair_idx]['base_name']
        print(f"\n🔄 Processing {i+1}/{len(files_to_process)}: {pair_name}")
        
        try:
            result = _process_single_file_pair(
                data_loader, config_dict, pair_idx, coregister_file_pair, SARFeatureExtractor, 
                generate_all_patches_from_data, save_all_patches_to_directory,
                features, patch_size, save_viz
            )
            
            if result['success']:
                # Check if file was skipped or actually processed
                if result.get('error') == 'SKIPPED - Output file exists and override_existing=False':
                    skipped_files += 1
                    print(f"  ⏭️  Skipped (file exists)")
                else:
                    stats['successful_files'] += 1
                    stats['total_patches_generated'] += result['n_patches_generated']
                    print(f"  ✅ Success: {result['n_patches_generated']} patches saved")
                    print(f"     Processing time: {result['processing_time']:.1f}s")
                
                # Add output files regardless of skip status
                if result['output_files']:
                    output_files.extend(result['output_files'].values())
                    
                    # Show output files
                    for file_type, file_path in result['output_files'].items():
                        try:
                            file_size = Path(file_path).stat().st_size / (1024*1024)  # MB
                            print(f"     Output: {Path(file_path).name} ({file_size:.1f} MB)")
                        except:
                            print(f"     Output: {Path(file_path).name}")
            else:
                failed_files.append(pair_name)
                error_msg = result['error']
                print(f"  ❌ Failed: {error_msg[:100]}..." if len(error_msg) > 100 else f"  ❌ Failed: {error_msg}")
                
        except Exception as e:
            failed_files.append(pair_name)
            print(f"  ❌ Critical error: {str(e)[:100]}..." if len(str(e)) > 100 else f"  ❌ Critical error: {str(e)}")
    
    # Store skipped count in stats for access by calling method
    stats['skipped_files'] = skipped_files


def _process_single_file_pair(data_loader: SARDataLoader, config_dict: Dict, pair_index: int, coregister_func, extractor_class, 
                             generate_func, save_func, features: List[str], patch_size: int, save_viz: bool) -> Dict:
    """Process a single file pair for patch generation."""
    pair_info = data_loader.file_pairs[pair_index]
    base_name = pair_info['base_name']
    aoi_info = pair_info['aoi']
    full_patches_dir = config_dict['full_patches_dir']
    override_existing = config_dict.get('override_existing', False)
    
    start_time = time.time()
    result = {
        'success': False,
        'error': None,
        'n_patches_generated': 0,
        'output_files': {},
        'processing_time': 0
    }
    
    try:
        # Check if output file already exists and skip if override is False
        existing_files = list(Path(full_patches_dir).glob(f"full_patches_{base_name}_*patches.nc"))
        
        if existing_files and not override_existing:
            existing_file = existing_files[0]  # Take the first match
            logger.info(f"⏭️  Skipping {base_name} - output file already exists: {existing_files[0].name}")
            result['success'] = True
            result['n_patches_generated'] = 0  # We didn't generate, but file exists
            result['output_files'] = {'full_patches': str(existing_file)}
            result['processing_time'] = time.time() - start_time
            result['error'] = 'SKIPPED - Output file exists and override_existing=False'
            return result
        
        # Step 1: Load data
        logger.info(f"Loading data for {base_name}")
        intensity_data, dsm_data, intensity_meta, dsm_meta = data_loader.load_file_pair(pair_index)
        
        # Step 2: Coregister data
        logger.info(f"Coregistering data for {base_name}")
        intensity_coreg, dsm_coreg, common_meta = coregister_func(
            intensity_data, dsm_data, intensity_meta, dsm_meta
        )
        
        # Step 3: Extract features using refactored SARFeatureExtractor
        logger.info(f"Extracting features for {base_name}")
        feature_extractor = extractor_class()
        feature_data, feature_names = feature_extractor.extract_selected_features(
            intensity_coreg, selected_features=features, metadata=common_meta
        )
        
        # Step 4: Generate patches
        logger.info(f"Generating patches for {base_name}")
        all_patches = generate_func(
            features=feature_data,
            dsm=dsm_coreg,
            metadata=common_meta,
            patch_size=patch_size,
            stride=None
        )
        
        result['n_patches_generated'] = len(all_patches)
        
        if not all_patches:
            result['error'] = "No patches could be generated (insufficient data size)"
            return result
        
        # Step 5: Save patches
        logger.info(f"Saving patches for {base_name}")
        source_metadata = {
            'aoi_info': aoi_info,
            'original_intensity_shape': intensity_data.shape,
            'original_dsm_shape': dsm_data.shape,
            'coregistered_shape': intensity_coreg.shape,
            'n_features_extracted': len(feature_names),
            'patch_size': patch_size
        }
        
        output_file = save_func(
            patches=all_patches,
            output_dir=full_patches_dir,
            feature_names=feature_names,
            base_name=base_name,
            source_metadata=source_metadata,
            global_attrs={
                'processing_date': datetime.now().isoformat(),
                'enabled_features': ','.join(features),
                'filtering_applied': False
            }
        )
        
        result['output_files'] = {'full_patches': output_file}
        result['success'] = True
        result['processing_time'] = time.time() - start_time
        
    except Exception as e:
        result['error'] = str(e)
        result['processing_time'] = time.time() - start_time
    
    return result


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