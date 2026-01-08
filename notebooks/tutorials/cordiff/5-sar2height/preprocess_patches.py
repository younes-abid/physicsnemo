"""
SAR2Height Simple Patch Preprocessing Script

This script handles basic preprocessing of SAR2Height patches with two key functionalities:
1. Invalid data handling: Clip invalid values (like -120 dB, -999999) to 0
2. Height normalization: Shift DSM patches so minimum value = 0 (relative height)

Simple, focused preprocessing without feature engineering.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import xarray as xr


@dataclass
class PreprocessingResults:
    """Results from the preprocessing step"""
    success: bool = False
    error: Optional[str] = None
    total_patches_preprocessed: int = 0
    files_processed: int = 0
    files_successful: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    output_files: List[str] = field(default_factory=list)
    total_processing_time: float = 0.0
    processing_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingConfig:
    """Simple configuration for SAR2Height preprocessing."""
    
    # I/O Settings
    input_dir: str
    output_dir: str
    override_existing: bool = False
    
    # Invalid Data Handling
    invalid_data_strategy: str = "clip_to_zero"  # "clip_to_zero" - clip invalid values to 0
    sar_invalid_threshold: float = -100.0        # SAR values below this are invalid (dB)
    dsm_invalid_threshold: float = -999.0        # DSM values below this are invalid (m)
    
    # Height Normalization
    height_normalization: str = "relative_min_zero"  # "none", "relative_min_zero" - shift min to 0
    
    # Processing Settings
    num_workers: Optional[int] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)


def clip_invalid_sar(sar_amplitude_db: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Clip invalid SAR amplitude values to 0.
    
    Args:
        sar_amplitude_db: SAR amplitude in dB
        config: Preprocessing configuration
        
    Returns:
        Clipped SAR amplitude array
    """
    # Find invalid values
    invalid_mask = (
        ~np.isfinite(sar_amplitude_db) |  # NaN, inf
        (sar_amplitude_db < config.sar_invalid_threshold)  # Below threshold
    )
    
    # Clip to 0
    clipped_sar = sar_amplitude_db.copy()
    clipped_sar[invalid_mask] = 0.0
    
    return clipped_sar


def clip_invalid_dsm(dsm_height: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Clip invalid DSM height values to 0.
    
    Args:
        dsm_height: DSM height values
        config: Preprocessing configuration
        
    Returns:
        Clipped DSM height array
    """
    # Find invalid values
    invalid_mask = (
        ~np.isfinite(dsm_height) |  # NaN, inf
        (dsm_height <= config.dsm_invalid_threshold)  # Below or equal to threshold
    )
    
    # Clip to 0
    clipped_dsm = dsm_height.copy()
    clipped_dsm[invalid_mask] = 0.0
    
    return clipped_dsm


def normalize_height_relative(dsm_height: np.ndarray) -> np.ndarray:
    """
    Normalize height to relative values with minimum = 0.
    
    Args:
        dsm_height: DSM height values
        
    Returns:
        Normalized height array with min = 0
    """
    # Only consider finite values for min calculation
    finite_mask = np.isfinite(dsm_height)
    
    if not np.any(finite_mask):
        # No valid data, return as is
        return dsm_height.copy()
    
    # Find minimum of valid values
    min_height = np.min(dsm_height[finite_mask])
    
    # Shift so minimum becomes 0
    normalized_height = dsm_height.copy()
    normalized_height[finite_mask] = dsm_height[finite_mask] - min_height
    
    return normalized_height


def process_single_patch(file_path: str, config: PreprocessingConfig) -> Dict[str, Any]:
    """
    Process a single patch file.
    
    Args:
        file_path: Path to input NetCDF file
        config: Preprocessing configuration
        
    Returns:
        Dictionary with processing results
    """
    try:
        input_path = Path(file_path)
        output_path = Path(config.output_dir) / input_path.name
        
        # Skip if output exists and not overriding
        if output_path.exists() and not config.override_existing:
            return {
                'status': 'skipped',
                'file': str(input_path),
                'reason': 'Output exists'
            }
        
        # Load data
        with xr.open_dataset(input_path) as ds:
            # Get SAR and DSM data
            sar_amplitude = ds['sar_amplitude'].values
            dsm_height = ds['dsm'].values
            
            # Process SAR amplitude
            if config.invalid_data_strategy == "clip_to_zero":
                sar_processed = clip_invalid_sar(sar_amplitude, config)
            else:
                sar_processed = sar_amplitude.copy()
            
            # Process DSM height
            if config.invalid_data_strategy == "clip_to_zero":
                dsm_processed = clip_invalid_dsm(dsm_height, config)
            else:
                dsm_processed = dsm_height.copy()
            
            # Normalize height
            if config.height_normalization == "relative_min_zero":
                dsm_processed = normalize_height_relative(dsm_processed)
            
            # Create output dataset
            ds_out = ds.copy()
            ds_out['sar_amplitude'] = (ds['sar_amplitude'].dims, sar_processed)
            ds_out['dsm'] = (ds['dsm'].dims, dsm_processed)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save processed data
            ds_out.to_netcdf(output_path, engine='netcdf4')
        
        return {
            'status': 'success',
            'file': str(input_path),
            'output': str(output_path)
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'file': str(input_path) if 'input_path' in locals() else file_path,
            'error': str(e)
        }


def preprocess_patches(config: PreprocessingConfig) -> PreprocessingResults:
    """
    Preprocess all patch files with multiprocessing.
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        PreprocessingResults object with processing statistics
    """
    start_time = time.time()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Find all NetCDF files
        input_path = Path(config.input_dir)
        nc_files = list(input_path.glob("*.nc"))
        
        if not nc_files:
            return PreprocessingResults(
                success=False,
                error="No NetCDF files found in input directory"
            )
        
        logger.info(f"Found {len(nc_files)} files to process")
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Process files with multiprocessing
        process_func = partial(process_single_patch, config=config)
        
        with Pool(processes=config.num_workers) as pool:
            results = pool.map(process_func, [str(f) for f in nc_files])
        
        # Collect statistics
        files_successful = sum(1 for r in results if r['status'] == 'success')
        files_failed = sum(1 for r in results if r['status'] == 'failed')
        files_skipped = sum(1 for r in results if r['status'] == 'skipped')
        
        output_files = [r['output'] for r in results if r['status'] == 'success']
        
        # Log results
        logger.info(f"Processing complete: {files_successful} successful, {files_failed} failed, {files_skipped} skipped")
        
        if files_failed > 0:
            failed_files = [r for r in results if r['status'] == 'failed']
            for fail in failed_files:
                logger.error(f"Failed to process {fail['file']}: {fail['error']}")
        
        processing_time = time.time() - start_time
        
        return PreprocessingResults(
            success=True,
            total_patches_preprocessed=files_successful,
            files_processed=len(nc_files),
            files_successful=files_successful,
            files_failed=files_failed,
            files_skipped=files_skipped,
            output_files=output_files,
            total_processing_time=processing_time,
            processing_stats={
                'avg_time_per_file': processing_time / max(len(nc_files), 1),
                'num_workers': config.num_workers
            }
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return PreprocessingResults(
            success=False,
            error=str(e),
            total_processing_time=processing_time
        )


if __name__ == "__main__":
    # Example usage
    config = PreprocessingConfig(
        input_dir="/path/to/input/patches",
        output_dir="/path/to/output/patches",
        override_existing=False,
        invalid_data_strategy="clip_to_zero",
        height_normalization="relative_min_zero",
        num_workers=4
    )
    
    results = preprocess_patches(config)
    
    if results.success:
        print(f"Preprocessing completed successfully!")
        print(f"Processed {results.files_successful}/{results.files_processed} files")
        print(f"Total time: {results.total_processing_time:.2f}s")
    else:
        print(f"Preprocessing failed: {results.error}")