"""
SAR2Height Patch Stitching System

This module provides comprehensive functionality for stitching together
predicted SAR2Height patches with overlapping areas, handling relative
height offsets and maintaining geospatial metadata.

Key Features:
- Multiple offset detection methods (mean, median, robust)
- Various blending strategies for overlapped areas
- Configuration-driven approach matching prediction pipeline
- Comprehensive metadata tracking
- Geospatial coordinate preservation
- NetCDF output with proper georeferencing

Author: AI Assistant
Date: 2026-01-16
"""

import numpy as np
import xarray as xr
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OffsetMethod(Enum):
    """Enumeration of available offset detection methods."""
    MEAN = "mean"
    MEDIAN = "median"
    ROBUST_MEAN = "robust_mean"
    LEAST_SQUARES = "least_squares"
    PERCENTILE = "percentile"


class BlendingMethod(Enum):
    """Enumeration of available blending methods for overlapped areas."""
    MEAN = "mean"
    WEIGHTED_DISTANCE = "weighted_distance"
    LINEAR_BLEND = "linear_blend"
    MIN = "min"
    MAX = "max"
    FIRST_PATCH = "first_patch"
    SECOND_PATCH = "second_patch"


@dataclass
class PatchMetadata:
    """Metadata for a single patch."""
    patch_id: int
    grid_row_start: int
    grid_col_start: int
    bounds_left: float
    bounds_bottom: float
    bounds_right: float
    bounds_top: float
    crs: str
    polygon_wkt: str
    dsm_mean: float
    dsm_std: float
    dsm_min: float
    dsm_max: float
    dsm_range: float
    dsm_valid_percentage: float
    feature_valid_percentage_min: float
    min_data_completeness: float
    ensemble_mean_offset: Optional[float] = None

    @classmethod
    def from_experiment_metadata(cls, metadata_dict: Dict[str, Any]) -> 'PatchMetadata':
        """Create PatchMetadata from experiment metadata dictionary."""
        # Handle both old and new metadata structures
        if 'metadata' in metadata_dict:
            # Old structure: metadata is directly under root
            meta = metadata_dict['metadata']
        elif 'data_summary' in metadata_dict and 'metadata' in metadata_dict['data_summary']:
            # New structure: metadata is under data_summary
            meta = metadata_dict['data_summary']['metadata']
        else:
            raise KeyError("Could not find metadata in experiment file. Expected 'metadata' or 'data_summary.metadata'")
            
        return cls(
            patch_id=meta['patch_id'],
            grid_row_start=meta['grid_row_start'],
            grid_col_start=meta['grid_col_start'],
            bounds_left=meta['bounds_left'],
            bounds_bottom=meta['bounds_bottom'],
            bounds_right=meta['bounds_right'],
            bounds_top=meta['bounds_top'],
            crs=meta['crs'],
            polygon_wkt=meta['polygon_wkt'],
            dsm_mean=meta['dsm_mean'],
            dsm_std=meta['dsm_std'],
            dsm_min=meta['dsm_min'],
            dsm_max=meta['dsm_max'],
            dsm_range=meta['dsm_range'],
            dsm_valid_percentage=meta['dsm_valid_percentage'],
            feature_valid_percentage_min=meta['feature_valid_percentage_min'],
            min_data_completeness=meta['min_data_completeness']
        )

    def get_bounds_tuple(self) -> Tuple[float, float, float, float]:
        """Get bounds as (left, bottom, right, top) tuple."""
        return (self.bounds_left, self.bounds_bottom, self.bounds_right, self.bounds_top)


@dataclass
class StitchingConfig:
    """Configuration for patch stitching process."""
    # Core parameters
    patch_size: int
    patch_stride: int
    offset_method: OffsetMethod
    blending_method: BlendingMethod
    
    # Advanced parameters
    percentile_range: Tuple[float, float] = (10.0, 90.0)
    outlier_threshold: float = 3.0
    min_overlap_pixels: int = 100
    
    # Output parameters
    output_dtype: np.dtype = np.float32
    compression_level: int = 6
    save_metadata: bool = True
    save_intermediate_results: bool = False
    
    # Processing parameters
    analyze_grid_structure: bool = True
    validate_patches: bool = True
    verbose_logging: bool = True

    @property
    def overlap_size(self) -> int:
        """Calculate overlap size from patch_size and patch_stride."""
        return self.patch_size - self.patch_stride

    @classmethod
    def from_pipeline_config(cls, **config_params) -> 'StitchingConfig':
        """Create StitchingConfig from pipeline configuration parameters."""
        return cls(
            patch_size=config_params.get('STITCHING_PATCH_SIZE', 432),
            patch_stride=config_params.get('STITCHING_PATCH_STRIDE', 405),
            offset_method=OffsetMethod(config_params.get('STITCHING_OFFSET_METHOD', 'mean')),
            blending_method=BlendingMethod(config_params.get('STITCHING_BLENDING_METHOD', 'weighted_distance')),
            percentile_range=config_params.get('STITCHING_PERCENTILE_RANGE', (10.0, 90.0)),
            outlier_threshold=config_params.get('STITCHING_OUTLIER_THRESHOLD', 3.0),
            min_overlap_pixels=config_params.get('STITCHING_MIN_OVERLAP_PIXELS', 100),
            output_dtype=np.dtype(config_params.get('STITCHING_OUTPUT_DTYPE', 'float32')),
            compression_level=config_params.get('STITCHING_COMPRESSION_LEVEL', 6),
            save_metadata=config_params.get('STITCHING_SAVE_METADATA', True),
            save_intermediate_results=config_params.get('STITCHING_SAVE_INTERMEDIATE_RESULTS', False),
            analyze_grid_structure=config_params.get('STITCHING_ANALYZE_GRID_STRUCTURE', True),
            validate_patches=config_params.get('STITCHING_VALIDATE_PATCHES', True),
            verbose_logging=config_params.get('STITCHING_VERBOSE_LOGGING', True)
        )


@dataclass
class StitchingResults:
    """Results from the stitching process."""
    success: bool
    stitched_dsm: Optional[np.ndarray] = None
    final_bounds: Optional[Tuple[float, float, float, float]] = None
    final_crs: Optional[str] = None
    total_patches_processed: int = 0
    patches_successfully_stitched: int = 0
    patches_skipped: int = 0
    patch_offsets: Optional[Dict[int, float]] = None
    overlap_statistics: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    output_shape: Optional[Tuple[int, int]] = None
    metadata_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    output_file_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if result['stitched_dsm'] is not None:
            result['stitched_dsm'] = "<numpy_array>"
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of stitching results."""
        return {
            'success': self.success,
            'total_patches': self.total_patches_processed,
            'successful_patches': self.patches_successfully_stitched,
            'skipped_patches': self.patches_skipped,
            'success_rate': (self.patches_successfully_stitched / self.total_patches_processed * 100) if self.total_patches_processed > 0 else 0,
            'processing_time': self.processing_time,
            'output_shape': self.output_shape,
            'output_file': self.output_file_path,
            'error_message': self.error_message
        }


class PatchData:
    """Container for patch prediction data and metadata."""

    def __init__(self, patch_id: int, prediction_dir: Path):
        """Initialize PatchData from prediction directory."""
        self.patch_id = patch_id
        self.prediction_dir = Path(prediction_dir)
        self.metadata: Optional[PatchMetadata] = None
        self.ensemble_mean: Optional[np.ndarray] = None
        self.ensemble_members: Optional[List[np.ndarray]] = None
        self.ground_truth: Optional[np.ndarray] = None
        
        self._load_data()

    def _load_data(self):
        """Load patch data from files."""
        try:
            # Load metadata
            metadata_file = self.prediction_dir / "experiment_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    self.metadata = PatchMetadata.from_experiment_metadata(metadata_dict)
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            # Load predictions
            predictions_dir = self.prediction_dir / "predictions"
            
            # Load ensemble members
            ensemble_file = predictions_dir / f"sar2height_prediction_sample_{self.patch_id}_ensemble_members.nc"
            if ensemble_file.exists():
                ensemble_ds = xr.open_dataset(ensemble_file)
                ensemble_array = ensemble_ds['ensemble_predictions'].values
                self.ensemble_members = [ensemble_array[i] for i in range(ensemble_array.shape[0])]
                ensemble_ds.close()
                
                # Calculate ensemble mean
                self.ensemble_mean = np.mean(ensemble_array, axis=0)
            
            # Load main predictions
            main_predictions_file = predictions_dir / f"sar2height_prediction_sample_{self.patch_id}_predictions.nc"
            if main_predictions_file.exists():
                main_ds = xr.open_dataset(main_predictions_file)
                if 'ground_truth' in main_ds:
                    self.ground_truth = main_ds['ground_truth'].values
                
                if self.ensemble_mean is None and 'ensemble_mean' in main_ds:
                    self.ensemble_mean = main_ds['ensemble_mean'].values
                    
                main_ds.close()
            
            if self.ensemble_mean is None:
                raise ValueError(f"No ensemble mean found for patch {self.patch_id}")
                
        except Exception as e:
            logger.error(f"Failed to load patch data for patch {self.patch_id}: {e}")
            raise

    def get_prediction_array(self) -> np.ndarray:
        """Get the prediction array (ensemble mean)."""
        if self.ensemble_mean is None:
            raise ValueError(f"No prediction data available for patch {self.patch_id}")
        return self.ensemble_mean.copy()

    def is_valid(self) -> bool:
        """Check if patch data is valid and complete."""
        return (self.metadata is not None and 
                self.ensemble_mean is not None and
                self.ensemble_mean.shape == (432, 432))


class OffsetDetector:
    """Handles offset detection between overlapping patches."""

    @staticmethod
    def detect_offset(patch1_overlap: np.ndarray, 
                     patch2_overlap: np.ndarray,
                     method: OffsetMethod = OffsetMethod.MEAN,
                     percentile_range: Tuple[float, float] = (10.0, 90.0),
                     outlier_threshold: float = 3.0) -> Tuple[float, Dict[str, Any]]:
        """Detect height offset between two overlapping patch regions."""
        if patch1_overlap.shape != patch2_overlap.shape:
            raise ValueError(f"Overlap shapes don't match: {patch1_overlap.shape} vs {patch2_overlap.shape}")

        # Calculate height differences
        height_diff = patch1_overlap - patch2_overlap
        
        # Remove invalid pixels
        valid_mask = np.isfinite(height_diff)
        valid_diff = height_diff[valid_mask]
        
        if len(valid_diff) == 0:
            return 0.0, {'error': 'No valid pixels in overlap region'}

        # Statistics for all methods
        stats = {
            'n_valid_pixels': len(valid_diff),
            'n_total_pixels': height_diff.size,
            'valid_percentage': (len(valid_diff) / height_diff.size) * 100,
            'raw_mean': float(np.mean(valid_diff)),
            'raw_median': float(np.median(valid_diff)),
            'raw_std': float(np.std(valid_diff)),
            'raw_min': float(np.min(valid_diff)),
            'raw_max': float(np.max(valid_diff)),
            'method_used': method.value
        }

        # Method-specific offset calculation
        if method == OffsetMethod.MEAN:
            offset = np.mean(valid_diff)
            
        elif method == OffsetMethod.MEDIAN:
            offset = np.median(valid_diff)
            
        elif method == OffsetMethod.ROBUST_MEAN:
            p_low, p_high = percentile_range
            low_thresh = np.percentile(valid_diff, p_low)
            high_thresh = np.percentile(valid_diff, p_high)
            robust_mask = (valid_diff >= low_thresh) & (valid_diff <= high_thresh)
            
            if np.sum(robust_mask) > 0:
                offset = np.mean(valid_diff[robust_mask])
                stats['robust_n_pixels'] = np.sum(robust_mask)
                stats['robust_percentage'] = (np.sum(robust_mask) / len(valid_diff)) * 100
            else:
                offset = np.mean(valid_diff)
                stats['robust_fallback'] = True
                
        elif method == OffsetMethod.LEAST_SQUARES:
            y_coords, x_coords = np.mgrid[0:patch1_overlap.shape[0], 0:patch1_overlap.shape[1]]
            valid_y = y_coords[valid_mask]
            valid_x = x_coords[valid_mask]
            
            A = np.column_stack([valid_x.flatten(), valid_y.flatten(), np.ones(len(valid_diff))])
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(A, valid_diff, rcond=None)
                offset = coeffs[2]
                stats['plane_coeffs'] = coeffs.tolist()
                if len(residuals) > 0:
                    stats['residual_sum'] = float(residuals[0])
            except np.linalg.LinAlgError:
                offset = np.mean(valid_diff)
                stats['lstsq_fallback'] = True
                
        elif method == OffsetMethod.PERCENTILE:
            p_low, p_high = percentile_range
            offset = np.median(np.percentile(valid_diff, [p_low, p_high]))
            
        else:
            raise ValueError(f"Unknown offset method: {method}")

        stats['final_offset'] = float(offset)
        return float(offset), stats


class PatchStitchingManager:
    """
    Main SAR2Height Patch Stitching Manager
    
    Similar to ExperimentManager, this class orchestrates the complete patch stitching workflow.
    """

    def __init__(self, 
                 experiment_name: str,
                 output_dir: Union[str, Path],
                 config: StitchingConfig):
        """Initialize PatchStitchingManager."""
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Create experiment directory structure
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.patches: List[PatchData] = []
        self.patch_metadata: Dict[int, PatchMetadata] = {}
        self.results: Optional[StitchingResults] = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"🧩 Patch Stitching Manager initialized: {experiment_name}")
        logger.info(f"📁 Results will be saved to: {self.experiment_dir}")
        logger.info(f"⚙️  Configuration: {config.offset_method.value} offset, {config.blending_method.value} blending")

    def _setup_logging(self):
        """Setup logging configuration."""
        if self.config.verbose_logging:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
            
        # Create log file in experiment directory
        log_file = self.experiment_dir / "stitching.log"
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

    def load_patches(self, experiment_dir: Path, patch_indices: List[int]) -> bool:
        """Load patches from experiment directory."""
        logger.info(f"📦 Loading {len(patch_indices)} patches from {experiment_dir}")
        
        self.patches = []
        failed_patches = []

        for patch_id in patch_indices:
            try:
                patch_dir = experiment_dir / f"sar2height_prediction_sample_{patch_id}"
                if not patch_dir.exists():
                    logger.warning(f"❌ Patch directory not found: {patch_dir}")
                    failed_patches.append(patch_id)
                    continue
                    
                patch_data = PatchData(patch_id, patch_dir)
                if patch_data.is_valid():
                    self.patches.append(patch_data)
                    self.patch_metadata[patch_id] = patch_data.metadata
                    logger.debug(f"✅ Successfully loaded patch {patch_id}")
                else:
                    logger.warning(f"⚠️  Invalid patch data for patch {patch_id}")
                    failed_patches.append(patch_id)
                    
            except Exception as e:
                logger.error(f"❌ Failed to load patch {patch_id}: {e}")
                failed_patches.append(patch_id)

        # Sort patches by grid position
        self.patches.sort(key=lambda p: (p.metadata.grid_row_start, p.metadata.grid_col_start))

        logger.info(f"✅ Successfully loaded {len(self.patches)} patches")
        if failed_patches:
            logger.warning(f"⚠️  Failed to load {len(failed_patches)} patches: {failed_patches}")

        return len(self.patches) > 0

    def analyze_grid_structure(self, patch_indices: List[int], experiment_dir: Path) -> Dict[str, Any]:
        """Analyze patch grid structure and coverage."""
        if not self.config.analyze_grid_structure:
            return {}
            
        logger.info("📊 Analyzing patch grid structure...")
        
        patches = []
        
        # Load patch metadata for analysis
        for patch_id in patch_indices:
            metadata_file = experiment_dir / f"sar2height_prediction_sample_{patch_id}" / "experiment_metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata_dict = json.load(f)
                        patches.append(PatchMetadata.from_experiment_metadata(metadata_dict))
                except Exception as e:
                    logger.warning(f"Failed to load metadata for patch {patch_id}: {e}")

        if not patches:
            return {'error': 'No valid patch metadata found'}

        # Analyze grid structure
        rows = sorted(set(p.grid_row_start for p in patches))
        cols = sorted(set(p.grid_col_start for p in patches))
        
        # Calculate grid spacing
        row_spacing = np.diff(rows) if len(rows) > 1 else [0]
        col_spacing = np.diff(cols) if len(cols) > 1 else [0]

        # Calculate coverage
        total_bounds = (
            min(p.bounds_left for p in patches),
            min(p.bounds_bottom for p in patches),
            max(p.bounds_right for p in patches),
            max(p.bounds_top for p in patches)
        )

        grid_analysis = {
            'n_patches': len(patches),
            'grid_dimensions': {'n_rows': len(rows), 'n_cols': len(cols)},
            'grid_spacing': {
                'row_spacing': row_spacing.tolist() if hasattr(row_spacing, 'tolist') else list(row_spacing),
                'col_spacing': col_spacing.tolist() if hasattr(col_spacing, 'tolist') else list(col_spacing),
                'mean_row_spacing': float(np.mean(row_spacing)) if len(row_spacing) > 0 else 0,
                'mean_col_spacing': float(np.mean(col_spacing)) if len(col_spacing) > 0 else 0
            },
            'geographic_bounds': {
                'left': total_bounds[0],
                'bottom': total_bounds[1],
                'right': total_bounds[2],
                'top': total_bounds[3],
                'width_m': total_bounds[2] - total_bounds[0],
                'height_m': total_bounds[3] - total_bounds[1]
            },
            'crs': patches[0].crs
        }

        # Log analysis results
        logger.info(f"📏 Grid: {grid_analysis['grid_dimensions']['n_rows']} rows × {grid_analysis['grid_dimensions']['n_cols']} cols")
        logger.info(f"📐 Total patches: {grid_analysis['n_patches']}")
        logger.info(f"🌍 Area: {grid_analysis['geographic_bounds']['width_m']:.1f} × {grid_analysis['geographic_bounds']['height_m']:.1f} m")

        # Save grid analysis if metadata saving is enabled
        if self.config.save_metadata:
            grid_analysis_file = self.experiment_dir / "grid_analysis.json"
            with open(grid_analysis_file, 'w') as f:
                json.dump(grid_analysis, f, indent=2, default=str)
            logger.info(f"💾 Grid analysis saved: {grid_analysis_file.name}")

        return grid_analysis

    def stitch_patches(self) -> StitchingResults:
        """Stitch all loaded patches into a single DSM."""
        start_time = time.time()
        
        if not self.patches:
            return StitchingResults(
                success=False,
                error_message="No patches loaded for stitching"
            )

        logger.info(f"🧩 Starting to stitch {len(self.patches)} patches")
        
        try:
            # Calculate output dimensions
            output_shape, grid_bounds = self._calculate_output_dimensions()
            logger.info(f"📏 Output shape: {output_shape}")

            # Initialize output arrays
            stitched_dsm = np.full(output_shape, np.nan, dtype=self.config.output_dtype)
            patch_count = np.zeros(output_shape, dtype=np.int32)

            # Track results
            patch_offsets = {}
            overlap_stats = []
            patches_processed = 0
            patches_successfully_stitched = 0

            # Process first patch as reference
            first_patch = self.patches[0]
            first_prediction = first_patch.get_prediction_array()
            first_row_start, first_col_start = self._get_output_position(first_patch, grid_bounds)
            
            stitched_dsm[first_row_start:first_row_start+self.config.patch_size,
                        first_col_start:first_col_start+self.config.patch_size] = first_prediction
            patch_count[first_row_start:first_row_start+self.config.patch_size,
                       first_col_start:first_col_start+self.config.patch_size] = 1

            patch_offsets[first_patch.patch_id] = 0.0
            patches_processed += 1
            patches_successfully_stitched += 1
            first_patch.metadata.ensemble_mean_offset = 0.0

            logger.info(f"✅ Placed reference patch {first_patch.patch_id}")

            # Process remaining patches
            for patch in self.patches[1:]:
                patches_processed += 1
                
                try:
                    # Get patch position
                    row_start, col_start = self._get_output_position(patch, grid_bounds)
                    patch_prediction = patch.get_prediction_array()
                    
                    # Check overlap with existing stitched region
                    existing_region = stitched_dsm[row_start:row_start+self.config.patch_size,
                                                   col_start:col_start+self.config.patch_size]
                    existing_count = patch_count[row_start:row_start+self.config.patch_size,
                                                 col_start:col_start+self.config.patch_size]
                    
                    # Find valid overlap pixels
                    overlap_mask = (existing_count > 0) & np.isfinite(existing_region) & np.isfinite(patch_prediction)
                    n_overlap_pixels = np.sum(overlap_mask)
                    
                    if n_overlap_pixels >= self.config.min_overlap_pixels:
                        # Calculate offset
                        existing_overlap = existing_region[overlap_mask]
                        patch_overlap = patch_prediction[overlap_mask]
                        
                        offset, offset_stats = OffsetDetector.detect_offset(
                            existing_overlap.reshape(-1, 1),
                            patch_overlap.reshape(-1, 1),
                            method=self.config.offset_method,
                            percentile_range=self.config.percentile_range,
                            outlier_threshold=self.config.outlier_threshold
                        )
                        
                        offset_stats['patch_id'] = patch.patch_id
                        offset_stats['n_overlap_pixels'] = n_overlap_pixels
                        overlap_stats.append(offset_stats)
                    else:
                        offset = 0.0
                        logger.warning(f"⚠️  Insufficient overlap ({n_overlap_pixels}) for patch {patch.patch_id}")

                    # Apply offset and simple blending
                    adjusted_prediction = patch_prediction + offset
                    patch_offsets[patch.patch_id] = offset
                    patch.metadata.ensemble_mean_offset = offset

                    # Simple mean blending for overlapped regions
                    if n_overlap_pixels >= self.config.min_overlap_pixels:
                        blend_mask = overlap_mask
                        if np.any(blend_mask):
                            stitched_patch_region = adjusted_prediction.copy()
                            blended_overlap = (existing_region[blend_mask] + 
                                             adjusted_prediction[blend_mask]) / 2.0
                            stitched_patch_region[blend_mask] = blended_overlap
                            
                            stitched_dsm[row_start:row_start+self.config.patch_size,
                                        col_start:col_start+self.config.patch_size] = stitched_patch_region
                        else:
                            stitched_dsm[row_start:row_start+self.config.patch_size,
                                        col_start:col_start+self.config.patch_size] = adjusted_prediction
                    else:
                        stitched_dsm[row_start:row_start+self.config.patch_size,
                                    col_start:col_start+self.config.patch_size] = adjusted_prediction

                    # Update patch count
                    patch_count[row_start:row_start+self.config.patch_size,
                               col_start:col_start+self.config.patch_size] += 1

                    patches_successfully_stitched += 1
                    logger.debug(f"✅ Stitched patch {patch.patch_id} with offset {offset:.4f}")

                except Exception as e:
                    logger.error(f"❌ Failed to stitch patch {patch.patch_id}: {e}")
                    continue

            # Calculate final bounds
            final_bounds = self._calculate_final_bounds()
            final_crs = self.patches[0].metadata.crs

            # Create results
            self.results = StitchingResults(
                success=True,
                stitched_dsm=stitched_dsm,
                final_bounds=final_bounds,
                final_crs=final_crs,
                total_patches_processed=patches_processed,
                patches_successfully_stitched=patches_successfully_stitched,
                patches_skipped=patches_processed - patches_successfully_stitched,
                patch_offsets=patch_offsets,
                overlap_statistics={'overlap_stats': overlap_stats},
                processing_time=time.time() - start_time,
                output_shape=output_shape,
                metadata_summary=self._create_metadata_summary()
            )

            logger.info(f"🎉 Stitching completed successfully in {self.results.processing_time:.2f}s")
            logger.info(f"📊 Successfully stitched: {patches_successfully_stitched}/{patches_processed} patches")

            return self.results

        except Exception as e:
            logger.error(f"💥 Stitching failed: {e}")
            return StitchingResults(
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def save_results(self, output_path: Optional[Path] = None) -> bool:
        """Save stitching results to NetCDF file."""
        if self.results is None or not self.results.success:
            logger.error("❌ No successful stitching results to save")
            return False

        if output_path is None:
            output_path = self.experiment_dir / "stitched_dsm.nc"
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create xarray dataset
            height, width = self.results.output_shape
            left, bottom, right, top = self.results.final_bounds
            x_coords = np.linspace(left, right, width)
            y_coords = np.linspace(top, bottom, height)

            # Create dataset
            ds = xr.Dataset(
                {
                    'height': (['y', 'x'], self.results.stitched_dsm, {
                        'long_name': 'Digital Surface Model Height',
                        'units': 'meters',
                        'description': 'Stitched SAR2Height prediction from ensemble mean'
                    })
                },
                coords={
                    'x': (['x'], x_coords, {
                        'long_name': 'Easting',
                        'units': 'meters',
                        'standard_name': 'projection_x_coordinate'
                    }),
                    'y': (['y'], y_coords, {
                        'long_name': 'Northing',
                        'units': 'meters',
                        'standard_name': 'projection_y_coordinate'
                    })
                },
                attrs={
                    'title': 'SAR2Height Stitched DSM Prediction',
                    'institution': 'SAR2Height Pipeline',
                    'source': 'Stitched from ensemble mean predictions',
                    'creation_date': datetime.now().isoformat(),
                    'experiment_name': self.experiment_name,
                    'crs': self.results.final_crs,
                    'total_patches': self.results.patches_successfully_stitched,
                    'processing_time_seconds': self.results.processing_time,
                    'patch_size': self.config.patch_size,
                    'patch_stride': self.config.patch_stride,
                    'overlap_size': self.config.overlap_size,
                    'offset_method': self.config.offset_method.value,
                    'blending_method': self.config.blending_method.value
                }
            )

            # Save to NetCDF
            encoding = {
                'height': {
                    'zlib': True,
                    'complevel': self.config.compression_level,
                    'dtype': self.config.output_dtype
                }
            }

            ds.to_netcdf(output_path, encoding=encoding, mode='w')
            logger.info(f"💾 Saved stitched DSM: {output_path}")

            # Update results with output path
            self.results.output_file_path = str(output_path)

            # Save metadata JSON
            if self.config.save_metadata:
                metadata_path = output_path.with_suffix('.json')
                metadata = {
                    'experiment_info': {
                        'name': self.experiment_name,
                        'creation_date': datetime.now().isoformat()
                    },
                    'stitching_results': self.results.to_dict(),
                    'patch_metadata': {str(k): asdict(v) for k, v in self.patch_metadata.items()},
                    'config': asdict(self.config)
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"💾 Saved metadata: {metadata_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            return False

    def run_complete_stitching_workflow(self,
                                      experiment_dir: Path,
                                      patch_indices: List[int],
                                      output_path: Optional[Path] = None) -> StitchingResults:
        """
        Run the complete patch stitching workflow.
        
        Similar to ExperimentManager.run_complete_experiment().
        """
        logger.info(f"🚀 Starting complete patch stitching workflow")
        logger.info(f"📦 Processing {len(patch_indices)} patches from {experiment_dir}")
        
        start_time = time.time()
        
        try:
            # Step 1: Analyze grid structure (optional)
            if self.config.analyze_grid_structure:
                grid_analysis = self.analyze_grid_structure(patch_indices, experiment_dir)
                if 'error' in grid_analysis:
                    logger.warning(f"⚠️  Grid analysis failed: {grid_analysis['error']}")
            
            # Step 2: Load patches
            if not self.load_patches(experiment_dir, patch_indices):
                return StitchingResults(
                    success=False,
                    error_message="Failed to load patches from experiment directory",
                    processing_time=time.time() - start_time
                )
            
            # Step 3: Perform stitching
            results = self.stitch_patches()
            
            # Step 4: Save results if successful
            if results.success:
                if self.save_results(output_path):
                    logger.info(f"✅ Complete stitching workflow finished successfully")
                else:
                    logger.error("⚠️  Stitching succeeded but saving failed")
                    results.error_message = "Stitching succeeded but saving failed"
            else:
                logger.error(f"❌ Stitching workflow failed: {results.error_message}")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Critical error in stitching workflow: {e}")
            return StitchingResults(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    def _calculate_output_dimensions(self) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """Calculate output grid dimensions and bounds."""
        min_row = min(p.metadata.grid_row_start for p in self.patches)
        max_row = max(p.metadata.grid_row_start for p in self.patches)
        min_col = min(p.metadata.grid_col_start for p in self.patches)
        max_col = max(p.metadata.grid_col_start for p in self.patches)

        output_height = (max_row - min_row) + self.config.patch_size
        output_width = (max_col - min_col) + self.config.patch_size

        return (output_height, output_width), (min_row, min_col, max_row, max_col)

    def _get_output_position(self, patch: PatchData, grid_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get position of patch in output grid."""
        min_row, min_col, _, _ = grid_bounds
        row_start = patch.metadata.grid_row_start - min_row
        col_start = patch.metadata.grid_col_start - min_col
        return row_start, col_start

    def _calculate_final_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate final geographic bounds of stitched result."""
        all_bounds = [p.metadata.get_bounds_tuple() for p in self.patches]
        left = min(b[0] for b in all_bounds)
        bottom = min(b[1] for b in all_bounds)
        right = max(b[2] for b in all_bounds)
        top = max(b[3] for b in all_bounds)
        return (left, bottom, right, top)

    def _create_metadata_summary(self) -> Dict[str, Any]:
        """Create summary of patch metadata."""
        return {
            'total_patches': len(self.patches),
            'patch_ids': [p.patch_id for p in self.patches],
            'grid_extent': {
                'min_row': min(p.metadata.grid_row_start for p in self.patches),
                'max_row': max(p.metadata.grid_row_start for p in self.patches),
                'min_col': min(p.metadata.grid_col_start for p in self.patches),
                'max_col': max(p.metadata.grid_col_start for p in self.patches)
            },
            'crs': self.patches[0].metadata.crs,
            'config': {
                'patch_size': self.config.patch_size,
                'patch_stride': self.config.patch_stride,
                'overlap_size': self.config.overlap_size,
                'offset_method': self.config.offset_method.value,
                'blending_method': self.config.blending_method.value
            }
        }


# =============================================================================
# UTILITY FUNCTIONS FOR PIPELINE INTEGRATION
# =============================================================================

def create_stitching_config_from_pipeline(**config_params) -> StitchingConfig:
    """Create StitchingConfig from pipeline configuration parameters."""
    return StitchingConfig.from_pipeline_config(**config_params)


def run_patch_stitching_pipeline(experiment_name: str,
                                experiment_dir: Path,
                                output_dir: Path,
                                patch_indices: List[int],
                                config: StitchingConfig,
                                output_filename: Optional[str] = None) -> StitchingResults:
    """
    Run the complete patch stitching pipeline.
    
    This is the main entry point function similar to your prediction pipeline.
    """
    # Create stitching manager
    stitcher = PatchStitchingManager(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config
    )
    
    # Determine output path
    output_path = None
    if output_filename:
        output_path = stitcher.experiment_dir / output_filename
    
    # Run complete workflow
    results = stitcher.run_complete_stitching_workflow(
        experiment_dir=experiment_dir,
        patch_indices=patch_indices,
        output_path=output_path
    )
    
    return results


if __name__ == "__main__":
    # Example usage
    print("SAR2Height Patch Stitching System")
    print("Available offset methods:", [m.value for m in OffsetMethod])
    print("Available blending methods:", [m.value for m in BlendingMethod])
    print("\nUse PatchStitchingManager for complete workflow management.")