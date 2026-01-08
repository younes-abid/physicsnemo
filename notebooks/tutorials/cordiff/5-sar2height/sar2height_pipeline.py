"""
SAR-to-Height Processing Pipeline - Refactored for Extensibility

This module provides a modular, extensible pipeline for processing SAR intensity 
and DSM data into cordiff-compatible NetCDF patches with three distinct phases:

1. Data Discovery: Find and validate raw SAR/DSM data pairs
2. Full Patch Generation: Generate complete patches without filtering
3. Patch Filtering: Apply quality and spatial filtering to generated patches

The pipeline is designed for easy extension and modification of processing steps.

Author: AI Assistant
Date: 2026-01-07
"""

import os
import sys
from pathlib import Path
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
import time
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil

# Scientific computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import FilteringResults from patch_filter to avoid duplication
from patch_filter import FilteringResults
# Import GenerationResults from patch_generation module
from patch_generation import GenerationResults, generate_full_patches_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class PipelineConfig:
    """Configuration class for the SAR-to-Height pipeline."""
    
    # Paths
    raw_data_dir: str
    full_patches_dir: str
    filtered_patches_dir: str
    visualization_dir: Optional[str] = None
    log_dir: Optional[str] = None
    
    # Processing parameters
    patch_size: int = 432
    process_all_files: bool = True
    max_files_to_process: int = 14
    override_existing: bool = False  # New parameter for skipping existing outputs
    
    # Multiprocessing parameters
    use_multiprocessing: bool = True
    max_workers: Optional[int] = None  # None means use all available cores
    chunk_size: int = 1  # Number of files per worker chunk
    
    # Feature extraction
    enabled_features: List[str] = None
    
    # Visualization
    save_individual_visualizations: bool = False
    visualization_speed_mode: bool = True
    max_features_to_display: int = 20
    figure_dpi: int = 100
    
    # Quality filtering criteria
    min_patch_quality: float = 0.7
    min_dsm_valid_percentage: float = 70.0
    min_feature_valid_percentage: float = 70.0
    min_dsm_std: float = 0.1
    max_dsm_std: float = 1000.0
    min_data_completeness: float = 70.0
    min_dsm_range: float = 0.5
    
    # Spatial filtering
    aoi_overlap_threshold: float = 0.0
    
    def __post_init__(self):
        """Post-initialization to set default features if not provided."""
        if self.enabled_features is None:
            self.enabled_features = [
                'intensity_db',
                'intensity_percentile_rescaled',
                'intensity_linear',
                'intensity_sqrt_linear',
                'intensity_power_transform',
                'intensity_lee_filtered',
                'local_mean_5x5',
                'local_std_5x5'
            ]
        
        # Set max_workers to all available cores if not specified
        if self.max_workers is None:
            self.max_workers = psutil.cpu_count(logical=True)
    
    def get_quality_thresholds(self):
        """Get quality thresholds for filtering."""
        from patch_filter import QualityThresholds
        return QualityThresholds(
            min_dsm_valid_percentage=self.min_dsm_valid_percentage,
            min_feature_valid_percentage=self.min_feature_valid_percentage,
            min_dsm_std=self.min_dsm_std,
            max_dsm_std=self.max_dsm_std,
            min_data_completeness=self.min_data_completeness,
            min_dsm_range=self.min_dsm_range
        )


@dataclass
class DiscoveryResults:
    """Results from data discovery phase."""
    success: bool
    n_intensity_files: int
    n_dsm_files: int
    n_file_pairs: int
    file_pairs_info: List[Dict]
    discovery_time: float
    error: Optional[str] = None


class SAR2HeightPipeline:
    """
    Modular SAR-to-Height processing pipeline.
    
    This pipeline provides three main processing phases:
    1. discover_raw_data(): Find and validate raw data pairs
    2. generate_full_patches(): Create complete patches without filtering
    3. filter_patches(): Apply quality and spatial filtering
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config (PipelineConfig): Pipeline configuration
        """
        self.config = config
        
        # Create directories
        self._create_directories()
        
        # Initialize components (lazy loading)
        self._components_initialized = False
        
        # Storage for pipeline data
        self.discovery_results: Optional[DiscoveryResults] = None
        self.generation_results: Optional[GenerationResults] = None
        self.filtering_results: Optional[FilteringResults] = None
        
        # Component references
        self.data_loader = None
        self.visualizer = None
    
    def _create_directories(self):
        """Create necessary output directories."""
        directories = [
            Path(self.config.full_patches_dir),
            Path(self.config.filtered_patches_dir)
        ]
        
        if self.config.visualization_dir:
            directories.append(Path(self.config.visualization_dir))
        
        if self.config.log_dir:
            directories.append(Path(self.config.log_dir))
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize processing components (lazy loading)."""
        if self._components_initialized:
            return
        
        try:
            # Import modules dynamically
            from data_loader import SARDataLoader
            from visualization import SARDataVisualizer
            
            # Initialize data loader
            self.data_loader = SARDataLoader(self.config.raw_data_dir)
            
            # Initialize visualizer if needed
            if self.config.save_individual_visualizations and self.config.visualization_dir:
                self.visualizer = SARDataVisualizer(
                    figsize=(15, 10), 
                    dpi=self.config.figure_dpi,
                    speed=self.config.visualization_speed_mode
                )
            
            self._components_initialized = True
            logger.info("✅ Pipeline components initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            raise RuntimeError(f"Component initialization failed: {e}")
    
    def discover_raw_data(self) -> DiscoveryResults:
        """
        Phase 1: Discover and validate available SAR and DSM data files.
        
        Returns:
            DiscoveryResults: Results of data discovery
        """
        logger.info("🔍 Phase 1: Discovering SAR and DSM raw data...")
        start_time = time.time()
        
        try:
            # Initialize components if needed
            self._initialize_components()
            
            # Get file information
            n_pairs = len(self.data_loader.file_pairs)
            n_intensity_files = len(self.data_loader.intensity_files)
            n_dsm_files = len(self.data_loader.dsm_files)
            
            discovery_time = time.time() - start_time
            
            # Create results
            self.discovery_results = DiscoveryResults(
                success=n_pairs > 0,
                n_intensity_files=n_intensity_files,
                n_dsm_files=n_dsm_files,
                n_file_pairs=n_pairs,
                file_pairs_info=self.data_loader.get_file_pairs_info() if n_pairs > 0 else [],
                discovery_time=discovery_time
            )
            
            # Log results
            print(f"📊 Data Discovery Results:")
            print(f"  • Found {n_intensity_files} SAR intensity files")
            print(f"  • Found {n_dsm_files} DSM files") 
            print(f"  • Created {n_pairs} matching intensity-DSM pairs")
            print(f"  • Discovery time: {discovery_time:.1f}s")
            
            if n_pairs == 0:
                print(f"\n⚠️  No matching file pairs found. Please check:")
                print(f"    - File naming conventions")
                print(f"    - Directory structure: {self.config.raw_data_dir}")
                print(f"    - File extensions (.tif)")
                self.discovery_results.error = "No matching file pairs found"
            else:
                print(f"\n✓ Ready to process {n_pairs} file pairs")
                
                # Show sample file pairs
                print(f"\n📋 Sample File Pairs:")
                for i, pair in enumerate(self.discovery_results.file_pairs_info[:3]):
                    aoi = pair['aoi']
                    print(f"  {i+1}. {pair['base_name']}")
                    print(f"      Satellite: {aoi['satellite']}, Orbit: {aoi['orbit_id']}, Date: {aoi['datetime'][:8]}")
            
            return self.discovery_results
            
        except Exception as e:
            discovery_time = time.time() - start_time
            error_msg = f"Data discovery failed: {str(e)}"
            logger.error(error_msg)
            
            self.discovery_results = DiscoveryResults(
                success=False,
                n_intensity_files=0,
                n_dsm_files=0, 
                n_file_pairs=0,
                file_pairs_info=[],
                discovery_time=discovery_time,
                error=error_msg
            )
            
            return self.discovery_results
    
    def generate_full_patches(self, 
                             process_all_files: Optional[bool] = None,
                             max_files: Optional[int] = None,
                             enabled_features: Optional[List[str]] = None,
                             save_visualizations: Optional[bool] = None,
                             patch_size: Optional[int] = None) -> GenerationResults:
        """
        Phase 2: Generate complete patches without filtering using multiprocessing.
        
        Args:
            process_all_files: Override config setting for processing all files
            max_files: Override config setting for max files to process
            enabled_features: Override config setting for features to extract
            save_visualizations: Override config setting for visualizations
            patch_size: Override config setting for patch size
            
        Returns:
            GenerationResults: Results of patch generation
        """
        logger.info("🏭 Phase 2: Generating full patches (no filtering)...")
        start_time = time.time()
        
        # Check prerequisites
        if not self.discovery_results or not self.discovery_results.success:
            error_msg = "Cannot generate patches - data discovery not completed or failed"
            logger.error(error_msg)
            return GenerationResults(
                success=False,
                files_processed=0,
                files_successful=0,
                files_failed=0,
                files_skipped=0,
                total_patches_generated=0,
                output_files=[],
                generation_time=time.time() - start_time,
                failed_files=[],
                error=error_msg
            )
        
        try:
            # Use provided parameters or fall back to config
            process_all = process_all_files if process_all_files is not None else self.config.process_all_files
            max_files = max_files if max_files is not None else self.config.max_files_to_process
            features = enabled_features if enabled_features is not None else self.config.enabled_features
            save_viz = save_visualizations if save_visualizations is not None else self.config.save_individual_visualizations
            p_size = patch_size if patch_size is not None else self.config.patch_size
            
            n_pairs = len(self.data_loader.file_pairs)
            
            # Determine files to process
            if process_all:
                files_to_process = list(range(n_pairs))
            else:
                files_to_process = list(range(min(max_files, n_pairs)))
            
            # Convert config to dictionary for the patch generation function
            config_dict = self.config.__dict__.copy()
            
            # Call the refactored patch generation function
            self.generation_results = generate_full_patches_pipeline(
                data_loader=self.data_loader,
                config_dict=config_dict,
                files_to_process=files_to_process,
                features=features,
                patch_size=p_size,
                save_viz=save_viz
            )
            
            return self.generation_results
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_msg = f"Patch generation failed: {str(e)}"
            logger.error(error_msg)
            
            self.generation_results = GenerationResults(
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
            
            return self.generation_results
    
    def filter_patches(self, 
                      full_patches_list: Optional[List[str]] = None,
                      quality_thresholds: Optional[Dict] = None,
                      overlap_tolerance: Optional[float] = None,
                      file_prefix: str = "filtered_patches") -> FilteringResults:
        """
        Phase 3: Apply quality and spatial filtering to generated patches.
        
        Args:
            full_patches_list: List of full patch files to filter (uses generation results if None)
            quality_thresholds: Override config quality thresholds
            overlap_tolerance: Override config spatial overlap tolerance
            file_prefix: Prefix for filtered output files
            
        Returns:
            FilteringResults: Results of patch filtering
        """
        logger.info("🎯 Phase 3: Filtering patches (quality + spatial)...")
        start_time = time.time()
        
        try:
            # Determine input files
            if full_patches_list is not None:
                input_files = full_patches_list
            elif self.generation_results and self.generation_results.success:
                input_files = self.generation_results.output_files
            else:
                error_msg = "Cannot filter patches - no full patches available (run generate_full_patches first)"
                logger.error(error_msg)
                return FilteringResults(
                    success=False,
                    input_files=[],
                    output_files=[],
                    total_patches_processed=0,
                    patches_passed_quality=0,
                    patches_passed_spatial=0,
                    patches_kept=0,
                    filtering_time=time.time() - start_time,
                    filtering_report={},
                    error=error_msg
                )
            
            if not input_files:
                error_msg = "No input files provided for filtering"
                logger.error(error_msg)
                return FilteringResults(
                    success=False,
                    input_files=[],
                    output_files=[],
                    total_patches_processed=0,
                    patches_passed_quality=0,
                    patches_passed_spatial=0,
                    patches_kept=0,
                    filtering_time=time.time() - start_time,
                    filtering_report={},
                    error=error_msg
                )
            
            # Import filtering module with correct interface
            from patch_filter import filter_patch_files, QualityThresholds
            
            # Prepare filtering parameters
            if quality_thresholds is not None:
                q_thresholds = QualityThresholds(**quality_thresholds)
            else:
                q_thresholds = self.config.get_quality_thresholds()
            
            overlap_tol = overlap_tolerance if overlap_tolerance is not None else self.config.aoi_overlap_threshold
            
            print(f"🔍 Filtering {len(input_files)} full patch files...")
            print(f"  • Quality thresholds: DSM valid ≥{q_thresholds.min_dsm_valid_percentage}%, "
                  f"Features valid ≥{q_thresholds.min_feature_valid_percentage}%")
            print(f"  • Spatial overlap tolerance: {overlap_tol}")
            print(f"  • Output directory: {self.config.filtered_patches_dir}")
            print("-" * 60)
            
            # Apply patch filtering with correct function name
            filter_results = filter_patch_files(
                input_files=input_files,
                output_dir=self.config.filtered_patches_dir,
                quality_thresholds=q_thresholds,
                overlap_tolerance=overlap_tol,
                file_prefix=file_prefix,
                override=False
            )
            
            # Convert the results from filter_patch_files to FilteringResults format
            filtering_time = time.time() - start_time
            
            # Create FilteringResults object from the returned data
            self.filtering_results = FilteringResults(
                success=filter_results.success,
                input_files=input_files,
                output_files=filter_results.output_files,  # Fixed: changed from filtered_files to output_files
                total_patches_processed=filter_results.total_patches_processed,
                patches_passed_quality=filter_results.patches_passed_quality,
                patches_passed_spatial=filter_results.patches_passed_spatial,
                patches_kept=filter_results.patches_kept,
                filtering_time=filtering_time,
                filtering_report=filter_results.filtering_report,
                error=filter_results.error
            )
            
            # Print summary
            print("\n" + "=" * 60)
            print("🎯 PATCH FILTERING COMPLETED")
            print("=" * 60)
            print(f"Input files: {len(self.filtering_results.input_files)}")
            print(f"Output files: {len(self.filtering_results.output_files)}")
            print(f"Patches processed: {self.filtering_results.total_patches_processed}")
            print(f"Passed quality: {self.filtering_results.patches_passed_quality}")
            print(f"Passed spatial: {self.filtering_results.patches_passed_spatial}")
            print(f"Final kept: {self.filtering_results.patches_kept}")
            print(f"Filtering time: {self.filtering_results.filtering_time:.1f}s")
            
            if self.filtering_results.output_files:
                total_size = sum(Path(f).stat().st_size for f in self.filtering_results.output_files) / (1024*1024)
                print(f"Total output size: {total_size:.1f} MB")
            
            return self.filtering_results
            
        except Exception as e:
            filtering_time = time.time() - start_time
            error_msg = f"Patch filtering failed: {str(e)}"
            logger.error(error_msg)
            
            self.filtering_results = FilteringResults(
                success=False,
                input_files=full_patches_list or [],
                output_files=[],
                total_patches_processed=0,
                patches_passed_quality=0,
                patches_passed_spatial=0,
                patches_kept=0,
                filtering_time=filtering_time,
                filtering_report={},
                error=error_msg
            )
            
            return self.filtering_results
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline processing summary."""
        return {
            'config': {
                'raw_data_dir': self.config.raw_data_dir,
                'full_patches_dir': self.config.full_patches_dir,
                'filtered_patches_dir': self.config.filtered_patches_dir,
                'patch_size': self.config.patch_size,
                'enabled_features': self.config.enabled_features,
                'process_all_files': self.config.process_all_files
            },
            'discovery_results': self.discovery_results,
            'generation_results': self.generation_results,
            'filtering_results': self.filtering_results,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_pipeline_report(self, output_file: Optional[str] = None) -> str:
        """Save comprehensive pipeline report to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(self.config.full_patches_dir) / f"pipeline_report_{timestamp}.json"
        
        report = self.get_pipeline_summary()
        
        # Convert dataclass objects to dicts for JSON serialization
        def convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                return obj.__dict__
            return str(obj)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=convert_dataclass)
        
        print(f"📄 Pipeline report saved: {output_file}")
        return str(output_file)


# Convenience functions for easy creation
def create_pipeline_config(raw_data_dir: str,
                          output_base_dir: str,
                          patch_size: int = 432,
                          enabled_features: Optional[List[str]] = None,
                          process_all_files: bool = True,
                          save_visualizations: bool = False,
                          **kwargs) -> PipelineConfig:
    """
    Create a pipeline configuration with sensible defaults.
    
    Args:
        raw_data_dir: Directory containing SAR and DSM TIFF files
        output_base_dir: Base output directory  
        patch_size: Size of patches (default 432)
        enabled_features: List of SAR features to extract
        process_all_files: Whether to process all available files
        save_visualizations: Whether to save visualization outputs
        **kwargs: Additional configuration parameters
        
    Returns:
        PipelineConfig: Configured pipeline parameters
    """
    output_base = Path(output_base_dir)
    
    config = PipelineConfig(
        raw_data_dir=raw_data_dir,
        full_patches_dir=str(output_base / "processed" / "full_patches"),
        filtered_patches_dir=str(output_base / "processed" / "filtered_patches"),
        visualization_dir=str(output_base / "visualizations") if save_visualizations else None,
        log_dir=str(output_base / "logs"),
        patch_size=patch_size,
        enabled_features=enabled_features,
        process_all_files=process_all_files,
        save_individual_visualizations=save_visualizations,
        **kwargs
    )
    
    return config


def create_pipeline(config: PipelineConfig) -> SAR2HeightPipeline:
    """
    Create a SAR-to-Height pipeline instance.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        SAR2HeightPipeline: Initialized pipeline instance
    """
    return SAR2HeightPipeline(config)


if __name__ == "__main__":
    # Example usage
    print("SAR-to-Height Modular Pipeline Module Loaded")
    print("Use create_pipeline_config() and create_pipeline() to get started")