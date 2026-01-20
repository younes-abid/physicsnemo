"""
SAR2Height Ensemble Patch Stitching System

This module provides functionality for stitching individual ensemble members
from SAR2Height patches to create full-scene ensemble predictions and uncertainty maps.

Key Features:
- Stitches all ensemble members individually to create full-scene ensemble
- Generates uncertainty maps (std deviation) from ensemble predictions
- Supports multiple ensemble ranking strategies
- Configuration-driven approach matching prediction pipeline
- Comprehensive metadata tracking and visualization

Author: AI Assistant
Date: 2026-01-18
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
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EnsembleRankingMethod(Enum):
    """Enumeration of available ensemble ranking methods."""
    PATCH_MEAN = "patch_mean"  # Rank by patch mean height
    GLOBAL_MEAN = "global_mean"  # Rank by global mean height
    VARIANCE = "variance"  # Rank by patch variance
    RANDOM = "random"  # Random ranking


class UncertaintyMethod(Enum):
    """Enumeration of uncertainty calculation methods."""
    STD = "std"  # Standard deviation
    VAR = "variance"  # Variance
    IQR = "interquartile_range"  # Interquartile range
    RANGE = "range"  # Min-max range


@dataclass
class EnsembleStitchingConfig:
    """Configuration for ensemble patch stitching process."""
    # Core parameters
    patch_size: int
    patch_stride: int
    ensemble_size: int
    
    # Ranking and uncertainty
    ranking_method: EnsembleRankingMethod = EnsembleRankingMethod.PATCH_MEAN
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.STD
    
    # Output parameters
    output_dtype: np.dtype = np.float32
    compression_level: int = 6
    save_metadata: bool = True
    save_individual_ensemble_members: bool = False
    save_uncertainty_maps: bool = True
    save_netcdf_files: bool = True  # NEW: Option to skip large NetCDF file writing
    
    # Visualization parameters
    save_labelless_figures: bool = True  # NEW: Save figures without labels
    
    # Performance parameters
    use_multiprocessing: bool = True  # NEW: Enable simple multiprocessing
    max_workers: int = 4  # NEW: Number of worker processes
    
    # Processing parameters
    analyze_grid_structure: bool = True
    validate_patches: bool = True
    verbose_logging: bool = True

    @property
    def overlap_size(self) -> int:
        """Calculate overlap size from patch_size and patch_stride."""
        return self.patch_size - self.patch_stride

    @classmethod
    def from_pipeline_config(cls, **config_params) -> 'EnsembleStitchingConfig':
        """Create EnsembleStitchingConfig from pipeline configuration parameters."""
        return cls(
            patch_size=config_params.get('STITCHING_PATCH_SIZE', 432),
            patch_stride=config_params.get('STITCHING_PATCH_STRIDE', 405),
            ensemble_size=config_params.get('ENSEMBLE_SIZE', 10),
            ranking_method=EnsembleRankingMethod(config_params.get('ENSEMBLE_RANKING_METHOD', 'patch_mean')),
            uncertainty_method=UncertaintyMethod(config_params.get('UNCERTAINTY_METHOD', 'std')),
            output_dtype=np.dtype(config_params.get('STITCHING_OUTPUT_DTYPE', 'float32')),
            compression_level=config_params.get('STITCHING_COMPRESSION_LEVEL', 6),
            save_metadata=config_params.get('STITCHING_SAVE_METADATA', True),
            save_individual_ensemble_members=config_params.get('SAVE_INDIVIDUAL_ENSEMBLE_MEMBERS', False),
            save_uncertainty_maps=config_params.get('SAVE_UNCERTAINTY_MAPS', True),
            save_netcdf_files=config_params.get('SAVE_NETCDF_FILES', True),  # NEW
            save_labelless_figures=config_params.get('SAVE_LABELLESS_FIGURES', True),  # NEW
            use_multiprocessing=config_params.get('USE_MULTIPROCESSING', True),  # NEW
            max_workers=config_params.get('MAX_WORKERS', 4),  # NEW
            analyze_grid_structure=config_params.get('STITCHING_ANALYZE_GRID_STRUCTURE', True),
            validate_patches=config_params.get('STITCHING_VALIDATE_PATCHES', True),
            verbose_logging=config_params.get('STITCHING_VERBOSE_LOGGING', True)
        )


@dataclass
class EnsembleStitchingResults:
    """Results from the ensemble stitching process."""
    success: bool
    ensemble_members: Optional[List[np.ndarray]] = None  # List of stitched ensemble members
    ensemble_mean: Optional[np.ndarray] = None
    ensemble_std: Optional[np.ndarray] = None
    ensemble_uncertainty_map: Optional[np.ndarray] = None
    final_bounds: Optional[Tuple[float, float, float, float]] = None
    final_crs: Optional[str] = None
    total_patches_processed: int = 0
    patches_successfully_stitched: int = 0
    patches_skipped: int = 0
    ensemble_statistics: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    output_shape: Optional[Tuple[int, int]] = None
    metadata_summary: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    output_files: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        # Replace numpy arrays with descriptions for JSON compatibility
        if result['ensemble_members'] is not None:
            result['ensemble_members'] = f"<{len(result['ensemble_members'])} ensemble members>"
        if result['ensemble_mean'] is not None:
            result['ensemble_mean'] = "<numpy_array>"
        if result['ensemble_std'] is not None:
            result['ensemble_std'] = "<numpy_array>"
        if result['ensemble_uncertainty_map'] is not None:
            result['ensemble_uncertainty_map'] = "<numpy_array>"
        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of ensemble stitching results."""
        return {
            'success': self.success,
            'total_patches': self.total_patches_processed,
            'successful_patches': self.patches_successfully_stitched,
            'skipped_patches': self.patches_skipped,
            'success_rate': (self.patches_successfully_stitched / self.total_patches_processed * 100) if self.total_patches_processed > 0 else 0,
            'processing_time': self.processing_time,
            'output_shape': self.output_shape,
            'ensemble_size': len(self.ensemble_members) if self.ensemble_members else 0,
            'output_files': self.output_files,
            'error_message': self.error_message
        }


class EnsemblePatchData:
    """Container for ensemble patch prediction data and metadata."""

    def __init__(self, patch_id: int, prediction_dir: Path):
        """Initialize EnsemblePatchData from prediction directory."""
        self.patch_id = patch_id
        self.prediction_dir = Path(prediction_dir)
        self.metadata = None
        self.ensemble_members: Optional[List[np.ndarray]] = None
        self.ensemble_mean: Optional[np.ndarray] = None
        self.ground_truth: Optional[np.ndarray] = None
        
        self._load_data()

    def _load_data(self):
        """Load ensemble patch data from files."""
        try:
            # Load metadata
            metadata_file = self.prediction_dir / "experiment_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata_dict = json.load(f)
                    # Handle both metadata structures
                    if 'metadata' in metadata_dict:
                        self.metadata = metadata_dict['metadata']
                    elif 'data_summary' in metadata_dict and 'metadata' in metadata_dict['data_summary']:
                        self.metadata = metadata_dict['data_summary']['metadata']
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

            # Load ensemble predictions
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
            else:
                raise FileNotFoundError(f"Ensemble file not found: {ensemble_file}")
            
            # Load ground truth if available
            main_predictions_file = predictions_dir / f"sar2height_prediction_sample_{self.patch_id}_predictions.nc"
            if main_predictions_file.exists():
                main_ds = xr.open_dataset(main_predictions_file)
                if 'ground_truth' in main_ds:
                    self.ground_truth = main_ds['ground_truth'].values
                main_ds.close()
                
        except Exception as e:
            logger.error(f"Failed to load ensemble patch data for patch {self.patch_id}: {e}")
            raise

    def get_ensemble_member(self, member_idx: int) -> np.ndarray:
        """Get a specific ensemble member."""
        if self.ensemble_members is None or member_idx >= len(self.ensemble_members):
            raise ValueError(f"Ensemble member {member_idx} not available for patch {self.patch_id}")
        return self.ensemble_members[member_idx].copy()

    def get_ensemble_statistics(self) -> Dict[str, float]:
        """Get statistics for this patch's ensemble."""
        if self.ensemble_members is None:
            return {}
        
        ensemble_array = np.array(self.ensemble_members)
        patch_means = [np.mean(member) for member in self.ensemble_members]
        
        return {
            'patch_mean_min': float(np.min(patch_means)),
            'patch_mean_max': float(np.max(patch_means)),
            'patch_mean_std': float(np.std(patch_means)),
            'ensemble_size': len(self.ensemble_members),
            'spatial_std_mean': float(np.mean(np.std(ensemble_array, axis=0)))
        }

    def rank_ensemble_members(self, method: EnsembleRankingMethod) -> List[int]:
        """Rank ensemble members by specified method."""
        if self.ensemble_members is None:
            return []
        
        if method == EnsembleRankingMethod.PATCH_MEAN:
            # Rank by patch mean height (low to high)
            patch_means = [np.mean(member) for member in self.ensemble_members]
            return np.argsort(patch_means).tolist()
            
        elif method == EnsembleRankingMethod.VARIANCE:
            # Rank by patch variance (low to high)
            patch_vars = [np.var(member) for member in self.ensemble_members]
            return np.argsort(patch_vars).tolist()
            
        elif method == EnsembleRankingMethod.RANDOM:
            # Random ranking
            indices = list(range(len(self.ensemble_members)))
            np.random.shuffle(indices)
            return indices
            
        else:  # GLOBAL_MEAN - will be handled at global level
            return list(range(len(self.ensemble_members)))

    def is_valid(self) -> bool:
        """Check if ensemble patch data is valid and complete."""
        return (self.metadata is not None and 
                self.ensemble_members is not None and
                len(self.ensemble_members) > 0 and
                all(member.shape == (432, 432) for member in self.ensemble_members))


class EnsemblePatchStitchingManager:
    """
    Main SAR2Height Ensemble Patch Stitching Manager
    
    Handles stitching of ensemble predictions to create full-scene ensemble
    predictions and uncertainty maps.
    """

    def __init__(self, 
                 experiment_name: str,
                 output_dir: Union[str, Path],
                 config: EnsembleStitchingConfig):
        """Initialize EnsemblePatchStitchingManager."""
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Create experiment directory structure
        self.experiment_dir = self.output_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.patches: List[EnsemblePatchData] = []
        self.patch_metadata: Dict[int, Dict] = {}
        self.results: Optional[EnsembleStitchingResults] = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"🧩 Ensemble Patch Stitching Manager initialized: {experiment_name}")
        logger.info(f"📁 Results will be saved to: {self.experiment_dir}")
        logger.info(f"⚙️  Configuration: {config.ensemble_size} ensemble members, {config.ranking_method.value} ranking")

    def _setup_logging(self):
        """Setup logging configuration."""
        if self.config.verbose_logging:
            log_level = logging.INFO
        else:
            log_level = logging.WARNING
            
        # Create log file in experiment directory
        log_file = self.experiment_dir / "ensemble_stitching.log"
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )

    def load_ensemble_patches(self, experiment_dir: Path, patch_indices: List[int]) -> bool:
        """Load ensemble patches from experiment directory with optional multiprocessing."""
        logger.info(f"📦 Loading {len(patch_indices)} ensemble patches from {experiment_dir}")
        
        self.patches = []
        failed_patches = []

        # Simple multiprocessing for patch loading (I/O bound operation)
        if self.config.use_multiprocessing and len(patch_indices) > 10:
            logger.info(f"⚡ Using multiprocessing with {self.config.max_workers} workers for patch loading")
            
            # Helper function for loading a single patch
            def load_single_patch(patch_id):
                try:
                    patch_dir = experiment_dir / f"sar2height_prediction_sample_{patch_id}"
                    if not patch_dir.exists():
                        return None, f"Directory not found: {patch_dir}"
                        
                    patch_data = EnsemblePatchData(patch_id, patch_dir)
                    if patch_data.is_valid():
                        return patch_data, None
                    else:
                        return None, f"Invalid patch data for patch {patch_id}"
                        
                except Exception as e:
                    return None, f"Failed to load patch {patch_id}: {e}"
            
            # Use ThreadPoolExecutor for I/O bound operations (file loading)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all patch loading tasks
                future_to_patch_id = {
                    executor.submit(load_single_patch, patch_id): patch_id 
                    for patch_id in patch_indices
                }
                
                # Collect results
                for future in as_completed(future_to_patch_id):
                    patch_id = future_to_patch_id[future]
                    try:
                        patch_data, error_msg = future.result()
                        if patch_data is not None:
                            self.patches.append(patch_data)
                            self.patch_metadata[patch_id] = patch_data.metadata
                            logger.debug(f"✅ Successfully loaded ensemble patch {patch_id}")
                        else:
                            logger.warning(f"⚠️  {error_msg}")
                            failed_patches.append(patch_id)
                    except Exception as e:
                        logger.error(f"❌ Exception loading patch {patch_id}: {e}")
                        failed_patches.append(patch_id)
                        
        else:
            # Sequential loading for small numbers of patches
            logger.info("📝 Using sequential loading")
            for patch_id in patch_indices:
                try:
                    patch_dir = experiment_dir / f"sar2height_prediction_sample_{patch_id}"
                    if not patch_dir.exists():
                        logger.warning(f"❌ Patch directory not found: {patch_dir}")
                        failed_patches.append(patch_id)
                        continue
                        
                    patch_data = EnsemblePatchData(patch_id, patch_dir)
                    if patch_data.is_valid():
                        self.patches.append(patch_data)
                        self.patch_metadata[patch_id] = patch_data.metadata
                        logger.debug(f"✅ Successfully loaded ensemble patch {patch_id}")
                    else:
                        logger.warning(f"⚠️  Invalid ensemble patch data for patch {patch_id}")
                        failed_patches.append(patch_id)
                        
                except Exception as e:
                    logger.error(f"❌ Failed to load ensemble patch {patch_id}: {e}")
                    failed_patches.append(patch_id)

        # Sort patches by grid position
        self.patches.sort(key=lambda p: (p.metadata['grid_row_start'], p.metadata['grid_col_start']))

        logger.info(f"✅ Successfully loaded {len(self.patches)} ensemble patches")
        if failed_patches:
            logger.warning(f"⚠️  Failed to load {len(failed_patches)} patches: {failed_patches[:10]}{'...' if len(failed_patches) > 10 else ''}")

        return len(self.patches) > 0

    def rank_global_ensemble_members(self) -> List[int]:
        """Rank ensemble members globally across all patches."""
        if not self.patches or self.config.ranking_method != EnsembleRankingMethod.GLOBAL_MEAN:
            return list(range(self.config.ensemble_size))
        
        logger.info("📊 Ranking ensemble members globally by mean height...")
        
        # Calculate global mean for each ensemble member
        global_means = []
        for member_idx in range(self.config.ensemble_size):
            member_means = []
            for patch in self.patches:
                if patch.ensemble_members and member_idx < len(patch.ensemble_members):
                    member_means.append(np.mean(patch.ensemble_members[member_idx]))
            
            if member_means:
                global_means.append(np.mean(member_means))
            else:
                global_means.append(0.0)
        
        # Return indices sorted by global mean (low to high)
        ranked_indices = np.argsort(global_means).tolist()
        
        logger.info(f"📈 Global ensemble ranking: {ranked_indices}")
        logger.info(f"   Global means: {[f'{m:.3f}' for m in sorted(global_means)]}")
        
        return ranked_indices

    def stitch_ensemble_members(self) -> EnsembleStitchingResults:
        """Stitch all ensemble members to create full-scene ensemble predictions."""
        start_time = time.time()
        
        if not self.patches:
            return EnsembleStitchingResults(
                success=False,
                error_message="No ensemble patches loaded for stitching"
            )

        logger.info(f"🧩 Starting to stitch {len(self.patches)} ensemble patches")
        logger.info(f"🔢 Ensemble size: {self.config.ensemble_size}")
        
        try:
            # Calculate output dimensions
            output_shape, grid_bounds = self._calculate_output_dimensions()
            logger.info(f"📏 Output shape: {output_shape}")

            # Get global ensemble ranking if needed
            if self.config.ranking_method == EnsembleRankingMethod.GLOBAL_MEAN:
                global_ranking = self.rank_global_ensemble_members()
            else:
                global_ranking = list(range(self.config.ensemble_size))

            # Initialize output arrays for all ensemble members
            stitched_ensemble_members = []
            for i in range(self.config.ensemble_size):
                stitched_ensemble_members.append(
                    np.full(output_shape, np.nan, dtype=self.config.output_dtype)
                )

            # Track results
            patches_processed = 0
            patches_successfully_stitched = 0
            ensemble_stats = {'patch_rankings': {}, 'global_ranking': global_ranking}

            # Process each patch
            for patch in self.patches:
                patches_processed += 1
                
                try:
                    # Get patch position in output grid
                    row_start, col_start = self._get_output_position(patch, grid_bounds)
                    
                    # Get patch-specific ranking if not using global ranking
                    if self.config.ranking_method == EnsembleRankingMethod.GLOBAL_MEAN:
                        patch_ranking = global_ranking
                    else:
                        patch_ranking = patch.rank_ensemble_members(self.config.ranking_method)
                    
                    ensemble_stats['patch_rankings'][patch.patch_id] = patch_ranking
                    
                    # Stitch each ensemble member according to ranking
                    for rank_idx, member_idx in enumerate(patch_ranking):
                        if member_idx < len(patch.ensemble_members):
                            member_data = patch.ensemble_members[member_idx]
                            
                            # Place in corresponding ranked position
                            stitched_ensemble_members[rank_idx][
                                row_start:row_start+self.config.patch_size,
                                col_start:col_start+self.config.patch_size
                            ] = member_data

                    patches_successfully_stitched += 1
                    logger.debug(f"✅ Stitched ensemble patch {patch.patch_id}")

                except Exception as e:
                    logger.error(f"❌ Failed to stitch ensemble patch {patch.patch_id}: {e}")
                    continue

            # Calculate ensemble statistics
            logger.info("📊 Calculating ensemble statistics...")
            ensemble_array = np.array(stitched_ensemble_members)
            ensemble_mean = np.mean(ensemble_array, axis=0)
            
            # Calculate uncertainty map
            if self.config.uncertainty_method == UncertaintyMethod.STD:
                uncertainty_map = np.std(ensemble_array, axis=0)
            elif self.config.uncertainty_method == UncertaintyMethod.VAR:
                uncertainty_map = np.var(ensemble_array, axis=0)
            elif self.config.uncertainty_method == UncertaintyMethod.IQR:
                q75 = np.percentile(ensemble_array, 75, axis=0)
                q25 = np.percentile(ensemble_array, 25, axis=0)
                uncertainty_map = q75 - q25
            elif self.config.uncertainty_method == UncertaintyMethod.RANGE:
                uncertainty_map = np.max(ensemble_array, axis=0) - np.min(ensemble_array, axis=0)

            ensemble_std = np.std(ensemble_array, axis=0)

            # Calculate final bounds and CRS
            final_bounds = self._calculate_final_bounds()
            final_crs = self.patches[0].metadata['crs']

            # Collect ensemble statistics
            valid_uncertainty = uncertainty_map[np.isfinite(uncertainty_map)]
            ensemble_statistics = {
                'uncertainty_stats': {
                    'method': self.config.uncertainty_method.value,
                    'mean': float(np.mean(valid_uncertainty)) if len(valid_uncertainty) > 0 else 0.0,
                    'std': float(np.std(valid_uncertainty)) if len(valid_uncertainty) > 0 else 0.0,
                    'min': float(np.min(valid_uncertainty)) if len(valid_uncertainty) > 0 else 0.0,
                    'max': float(np.max(valid_uncertainty)) if len(valid_uncertainty) > 0 else 0.0,
                    'median': float(np.median(valid_uncertainty)) if len(valid_uncertainty) > 0 else 0.0
                },
                'ensemble_member_stats': {
                    'global_means': [float(np.mean(member[np.isfinite(member)])) 
                                   for member in stitched_ensemble_members 
                                   if np.any(np.isfinite(member))],
                    'global_stds': [float(np.std(member[np.isfinite(member)])) 
                                  for member in stitched_ensemble_members 
                                  if np.any(np.isfinite(member))]
                },
                'ranking_method': self.config.ranking_method.value
            }

            # Create results
            self.results = EnsembleStitchingResults(
                success=True,
                ensemble_members=stitched_ensemble_members,
                ensemble_mean=ensemble_mean,
                ensemble_std=ensemble_std,
                ensemble_uncertainty_map=uncertainty_map,
                final_bounds=final_bounds,
                final_crs=final_crs,
                total_patches_processed=patches_processed,
                patches_successfully_stitched=patches_successfully_stitched,
                patches_skipped=patches_processed - patches_successfully_stitched,
                ensemble_statistics=ensemble_statistics,
                processing_time=time.time() - start_time,
                output_shape=output_shape,
                metadata_summary=self._create_metadata_summary()
            )

            logger.info(f"🎉 Ensemble stitching completed successfully in {self.results.processing_time:.2f}s")
            logger.info(f"📊 Successfully stitched: {patches_successfully_stitched}/{patches_processed} patches")
            logger.info(f"📈 Uncertainty stats: mean={ensemble_statistics['uncertainty_stats']['mean']:.4f}, "
                       f"std={ensemble_statistics['uncertainty_stats']['std']:.4f}")

            return self.results

        except Exception as e:
            logger.error(f"💥 Ensemble stitching failed: {e}")
            return EnsembleStitchingResults(
                success=False,
                processing_time=time.time() - start_time,
                error_message=str(e)
            )

    def save_results(self, output_dir: Optional[Path] = None) -> bool:
        """Save ensemble stitching results to NetCDF files (conditional based on config)."""
        if self.results is None or not self.results.success:
            logger.error("❌ No successful ensemble stitching results to save")
            return False

        if output_dir is None:
            output_dir = self.experiment_dir
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        try:
            # Skip NetCDF saving if disabled for performance
            if not self.config.save_netcdf_files:
                logger.info("⚡ Skipping NetCDF file creation (save_netcdf_files=False)")
                
                # Save only metadata JSON
                if self.config.save_metadata:
                    metadata_path = output_dir / "ensemble_stitching_metadata.json"
                    metadata = {
                        'experiment_info': {
                            'name': self.experiment_name,
                            'creation_date': datetime.now().isoformat()
                        },
                        'stitching_results': self.results.to_dict(),
                        'config': asdict(self.config),
                        'note': 'NetCDF files skipped for performance (save_netcdf_files=False)'
                    }
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    logger.info(f"💾 Saved metadata only: {metadata_path}")
                    output_files['metadata'] = str(metadata_path)
                
                # Update results with output paths
                self.results.output_files = output_files
                return True
            
            # Continue with normal NetCDF saving if enabled
            logger.info("💾 Creating NetCDF files (this may take time for large datasets)...")
            
            # Create coordinate arrays
            height, width = self.results.output_shape
            left, bottom, right, top = self.results.final_bounds
            x_coords = np.linspace(left, right, width)
            y_coords = np.linspace(top, bottom, height)

            # Common attributes
            common_attrs = {
                'title': 'SAR2Height Ensemble Stitched Predictions',
                'institution': 'SAR2Height Pipeline',
                'creation_date': datetime.now().isoformat(),
                'experiment_name': self.experiment_name,
                'crs': self.results.final_crs,
                'total_patches': self.results.patches_successfully_stitched,
                'processing_time_seconds': self.results.processing_time,
                'patch_size': self.config.patch_size,
                'patch_stride': self.config.patch_stride,
                'overlap_size': self.config.overlap_size,
                'ensemble_size': self.config.ensemble_size,
                'ranking_method': self.config.ranking_method.value,
                'uncertainty_method': self.config.uncertainty_method.value
            }

            # Encoding settings (optimized for speed)
            encoding_template = {
                'zlib': True,
                'complevel': self.config.compression_level,
                'dtype': self.config.output_dtype,
                'shuffle': True  # Better compression
            }

            # 1. Save ensemble mean (always save this as it's most important)
            ensemble_mean_path = output_dir / "ensemble_mean.nc"
            ds_mean = xr.Dataset(
                {
                    'height': (['y', 'x'], self.results.ensemble_mean, {
                        'long_name': 'Ensemble Mean Height',
                        'units': 'meters',
                        'description': 'Mean of stitched ensemble predictions'
                    })
                },
                coords={
                    'x': (['x'], x_coords, {'long_name': 'Easting', 'units': 'meters'}),
                    'y': (['y'], y_coords, {'long_name': 'Northing', 'units': 'meters'})
                },
                attrs={**common_attrs, 'source': 'Ensemble mean from stitched predictions'}
            )
            
            ds_mean.to_netcdf(ensemble_mean_path, encoding={'height': encoding_template})
            output_files['ensemble_mean'] = str(ensemble_mean_path)
            logger.info(f"💾 Saved ensemble mean: {ensemble_mean_path}")

            # 2. Save uncertainty map (conditional)
            if self.config.save_uncertainty_maps:
                uncertainty_path = output_dir / "ensemble_uncertainty.nc"
                ds_uncertainty = xr.Dataset(
                    {
                        'uncertainty': (['y', 'x'], self.results.ensemble_uncertainty_map, {
                            'long_name': f'Ensemble Uncertainty ({self.config.uncertainty_method.value.upper()})',
                            'units': 'meters',
                            'description': f'Uncertainty calculated as {self.config.uncertainty_method.value} of ensemble predictions'
                        }),
                        'ensemble_std': (['y', 'x'], self.results.ensemble_std, {
                            'long_name': 'Ensemble Standard Deviation',
                            'units': 'meters',
                            'description': 'Standard deviation of ensemble predictions'
                        })
                    },
                    coords={
                        'x': (['x'], x_coords, {'long_name': 'Easting', 'units': 'meters'}),
                        'y': (['y'], y_coords, {'long_name': 'Northing', 'units': 'meters'})
                    },
                    attrs={**common_attrs, 'source': 'Uncertainty from stitched ensemble predictions'}
                )
                
                uncertainty_encoding = {
                    'uncertainty': encoding_template,
                    'ensemble_std': encoding_template
                }
                ds_uncertainty.to_netcdf(uncertainty_path, encoding=uncertainty_encoding)
                output_files['ensemble_uncertainty'] = str(uncertainty_path)
                logger.info(f"💾 Saved uncertainty map: {uncertainty_path}")

            # 3. Save individual ensemble members (conditional - this is the biggest file)
            if self.config.save_individual_ensemble_members:
                logger.info("💾 Saving large ensemble members file (this will take time)...")
                ensemble_members_path = output_dir / "ensemble_members.nc"
                
                # Create ensemble member dataset
                ensemble_data = np.array(self.results.ensemble_members)
                ds_ensemble = xr.Dataset(
                    {
                        'ensemble_predictions': (['ensemble_member', 'y', 'x'], ensemble_data, {
                            'long_name': 'Ensemble Member Predictions',
                            'units': 'meters',
                            'description': f'Individual stitched ensemble members (ranked by {self.config.ranking_method.value})'
                        })
                    },
                    coords={
                        'ensemble_member': (['ensemble_member'], range(self.config.ensemble_size), {
                            'long_name': 'Ensemble Member Index',
                            'description': f'Ranked by {self.config.ranking_method.value}'
                        }),
                        'x': (['x'], x_coords, {'long_name': 'Easting', 'units': 'meters'}),
                        'y': (['y'], y_coords, {'long_name': 'Northing', 'units': 'meters'})
                    },
                    attrs={**common_attrs, 'source': 'Individual stitched ensemble member predictions'}
                )
                
                ds_ensemble.to_netcdf(ensemble_members_path, encoding={'ensemble_predictions': encoding_template})
                output_files['ensemble_members'] = str(ensemble_members_path)
                logger.info(f"💾 Saved ensemble members: {ensemble_members_path}")

            # Update results with output paths
            self.results.output_files = output_files

            # Save metadata JSON
            if self.config.save_metadata:
                metadata_path = output_dir / "ensemble_stitching_metadata.json"
                metadata = {
                    'experiment_info': {
                        'name': self.experiment_name,
                        'creation_date': datetime.now().isoformat()
                    },
                    'stitching_results': self.results.to_dict(),
                    'config': asdict(self.config),
                    'output_files': output_files
                }
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                logger.info(f"💾 Saved metadata: {metadata_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to save ensemble results: {e}")
            return False

    def run_complete_ensemble_stitching_workflow(self,
                                                experiment_dir: Path,
                                                patch_indices: List[int],
                                                output_dir: Optional[Path] = None) -> EnsembleStitchingResults:
        """Run the complete ensemble stitching workflow."""
        logger.info(f"🚀 Starting complete ensemble stitching workflow")
        logger.info(f"📦 Processing {len(patch_indices)} patches from {experiment_dir}")
        
        start_time = time.time()
        
        try:
            # Step 1: Load ensemble patches
            if not self.load_ensemble_patches(experiment_dir, patch_indices):
                return EnsembleStitchingResults(
                    success=False,
                    error_message="Failed to load ensemble patches from experiment directory",
                    processing_time=time.time() - start_time
                )
            
            # Step 2: Stitch ensemble members
            results = self.stitch_ensemble_members()
            
            # Step 3: Save results if successful
            if results.success:
                if self.save_results(output_dir):
                    logger.info(f"✅ Complete ensemble stitching workflow finished successfully")
                else:
                    logger.error("⚠️  Stitching succeeded but saving failed")
                    results.error_message = "Stitching succeeded but saving failed"
            else:
                logger.error(f"❌ Ensemble stitching workflow failed: {results.error_message}")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 Critical error in ensemble stitching workflow: {e}")
            return EnsembleStitchingResults(
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time
            )

    def visualize_ensemble_results(self, save_figures: bool = True, colormap: str = 'jet') -> None:
        """
        Visualize ensemble stitching results with comprehensive grid layouts.
        
        Creates two main figure grids:
        1. All 10 ensemble members (ranked from lowest to highest) 
        2. Uncertainty and statistical analysis plots
        
        Also creates labelless versions in a 'labelless' subfolder.
        """
        if self.results is None or not self.results.success:
            logger.error("❌ No successful results to visualize")
            return

        # Create labelless subfolder if enabled
        labelless_dir = None
        if self.config.save_labelless_figures:
            labelless_dir = self.experiment_dir / "labelless"
            labelless_dir.mkdir(exist_ok=True)

        # Get geographic bounds for projected plotting
        left, bottom, right, top = self.results.final_bounds
        
        # =============================================================================
        # FIGURE 1: ALL ENSEMBLE MEMBERS GRID (2x5 layout)
        # =============================================================================
        
        print("🎨 Creating ensemble members visualization...")
        
        # Create both labeled and labelless versions
        for is_labelless in [False, True]:
            if is_labelless and not self.config.save_labelless_figures:
                continue
                
            fig1, axes1 = plt.subplots(2, 5, figsize=(25, 10))
            axes1 = axes1.flatten()
            
            # Plot each ensemble member with projected coordinates
            for i, ensemble_member in enumerate(self.results.ensemble_members):
                ax = axes1[i]
                
                # Use projected plotting with extent for correct pixel sizes
                im = ax.imshow(ensemble_member, 
                              extent=[left, right, bottom, top],
                              cmap=colormap, 
                              aspect='equal',
                              origin='upper')
                
                if not is_labelless:
                    # Determine ranking label
                    if self.config.ranking_method.value == 'patch_mean':
                        rank_label = f"#{i+1} (Lowest → Highest Mean)"
                    elif self.config.ranking_method.value == 'global_mean':
                        rank_label = f"#{i+1} (Global Ranking)"
                    elif self.config.ranking_method.value == 'variance':
                        rank_label = f"#{i+1} (Lowest → Highest Variance)"
                    else:
                        rank_label = f"Ensemble Member #{i+1}"
                    
                    ax.set_title(rank_label, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Easting (m)', fontsize=10)
                    ax.set_ylabel('Northing (m)', fontsize=10)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Height (m)', fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    
                    # Calculate and display statistics
                    valid_data = ensemble_member[np.isfinite(ensemble_member)]
                    if len(valid_data) > 0:
                        stats_text = f"μ={np.mean(valid_data):.2f}m\nσ={np.std(valid_data):.2f}m"
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', fontsize=9,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    # Remove all labels for labelless version
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title('')
                    ax.set_xlabel('')
                    ax.set_ylabel('')
            
            if not is_labelless:
                # Overall title for ensemble grid
                ranking_method_title = self.config.ranking_method.value.replace('_', ' ').title()
                fig1.suptitle(f'Ensemble Members Ranked by {ranking_method_title}\n'
                             f'{self.results.patches_successfully_stitched} Patches • '
                             f'{self.config.ensemble_size} Members • {colormap.upper()} Colormap', 
                             fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            if save_figures:
                if is_labelless:
                    ensemble_grid_path = labelless_dir / "ensemble_members_grid_labelless.png"
                else:
                    ensemble_grid_path = self.experiment_dir / "ensemble_members_grid.png"
                plt.savefig(ensemble_grid_path, dpi=150, bbox_inches='tight', facecolor='white')  # Reduced DPI for speed
                logger.info(f"💾 Saved ensemble grid: {ensemble_grid_path}")
            
            if not is_labelless:  # Only show labeled version
                plt.show()
            else:
                plt.close()

        # =============================================================================
        # FIGURE 2: UNCERTAINTY AND STATISTICS GRID (2x3 layout)
        # =============================================================================
        
        print("📊 Creating uncertainty and statistics visualization...")
        
        # Create both labeled and labelless versions
        for is_labelless in [False, True]:
            if is_labelless and not self.config.save_labelless_figures:
                continue
                
            fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
            
            # Plot 1: Ensemble Mean
            im1 = axes2[0,0].imshow(self.results.ensemble_mean, 
                                   extent=[left, right, bottom, top],
                                   cmap=colormap, aspect='equal', origin='upper')
            if not is_labelless:
                axes2[0,0].set_title('📈 Ensemble Mean Height', fontsize=14, fontweight='bold')
                axes2[0,0].set_xlabel('Easting (m)')
                axes2[0,0].set_ylabel('Northing (m)')
                cbar1 = plt.colorbar(im1, ax=axes2[0,0], shrink=0.8)
                cbar1.set_label('Height (m)')
                
                # Add ensemble mean statistics
                valid_mean = self.results.ensemble_mean[np.isfinite(self.results.ensemble_mean)]
                if len(valid_mean) > 0:
                    mean_stats = f"""Mean: {np.mean(valid_mean):.3f}m
Std: {np.std(valid_mean):.3f}m
Range: {np.ptp(valid_mean):.3f}m"""
                    axes2[0,0].text(0.02, 0.98, mean_stats, transform=axes2[0,0].transAxes,
                                   verticalalignment='top', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            else:
                axes2[0,0].set_xticks([])
                axes2[0,0].set_yticks([])

            # Plot 2: Uncertainty Map (Using MAGMA colormap as requested)
            im2 = axes2[0,1].imshow(self.results.ensemble_uncertainty_map, 
                                   extent=[left, right, bottom, top],
                                   cmap='magma', aspect='equal', origin='upper')  # MAGMA for uncertainty
            if not is_labelless:
                uncertainty_title = f'🎯 Uncertainty ({self.config.uncertainty_method.value.upper()})'
                axes2[0,1].set_title(uncertainty_title, fontsize=14, fontweight='bold')
                axes2[0,1].set_xlabel('Easting (m)')
                axes2[0,1].set_ylabel('Northing (m)')
                cbar2 = plt.colorbar(im2, ax=axes2[0,1], shrink=0.8)
                cbar2.set_label(f'Uncertainty (m)')

                # Add uncertainty statistics
                valid_uncertainty = self.results.ensemble_uncertainty_map[np.isfinite(self.results.ensemble_uncertainty_map)]
                if len(valid_uncertainty) > 0:
                    unc_stats = f"""Mean: {np.mean(valid_uncertainty):.4f}m
Median: {np.median(valid_uncertainty):.4f}m
Max: {np.max(valid_uncertainty):.4f}m"""
                    axes2[0,1].text(0.02, 0.98, unc_stats, transform=axes2[0,1].transAxes,
                                   verticalalignment='top', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            else:
                axes2[0,1].set_xticks([])
                axes2[0,1].set_yticks([])

            # Plot 3: Standard Deviation Map (Using MAGMA for consistency)
            im3 = axes2[0,2].imshow(self.results.ensemble_std, 
                                   extent=[left, right, bottom, top],
                                   cmap='magma', aspect='equal', origin='upper')  # MAGMA for std
            if not is_labelless:
                axes2[0,2].set_title('📊 Ensemble Standard Deviation', fontsize=14, fontweight='bold')
                axes2[0,2].set_xlabel('Easting (m)')
                axes2[0,2].set_ylabel('Northing (m)')
                cbar3 = plt.colorbar(im3, ax=axes2[0,2], shrink=0.8)
                cbar3.set_label('Std Dev (m)')
            else:
                axes2[0,2].set_xticks([])
                axes2[0,2].set_yticks([])

            # Plot 4: Ensemble Range (Max - Min)
            ensemble_array = np.array(self.results.ensemble_members)
            ensemble_min = np.min(ensemble_array, axis=0)
            ensemble_max = np.max(ensemble_array, axis=0)
            ensemble_range = ensemble_max - ensemble_min
            
            im4 = axes2[1,0].imshow(ensemble_range, 
                                   extent=[left, right, bottom, top],
                                   cmap='magma', aspect='equal', origin='upper')  # MAGMA for range
            if not is_labelless:
                axes2[1,0].set_title('📏 Ensemble Range (Max - Min)', fontsize=14, fontweight='bold')
                axes2[1,0].set_xlabel('Easting (m)')
                axes2[1,0].set_ylabel('Northing (m)')
                cbar4 = plt.colorbar(im4, ax=axes2[1,0], shrink=0.8)
                cbar4.set_label('Range (m)')
            else:
                axes2[1,0].set_xticks([])
                axes2[1,0].set_yticks([])

            # Plot 5: Uncertainty Histogram
            if not is_labelless:
                valid_uncertainty = self.results.ensemble_uncertainty_map[np.isfinite(self.results.ensemble_uncertainty_map)]
                axes2[1,1].hist(valid_uncertainty, bins=50, alpha=0.8, color='purple', 
                               edgecolor='darkmagenta', linewidth=0.5, density=True)
                axes2[1,1].set_title('📊 Uncertainty Distribution', fontsize=14, fontweight='bold')
                axes2[1,1].set_xlabel(f'Uncertainty (m)')
                axes2[1,1].set_ylabel('Density')
                axes2[1,1].grid(True, alpha=0.3)
                
                # Add distribution statistics
                if len(valid_uncertainty) > 0:
                    hist_stats = f"""Samples: {len(valid_uncertainty):,}
Mean: {np.mean(valid_uncertainty):.4f}
Std: {np.std(valid_uncertainty):.4f}
Q95: {np.percentile(valid_uncertainty, 95):.4f}"""
                    axes2[1,1].text(0.98, 0.98, hist_stats, transform=axes2[1,1].transAxes,
                                   verticalalignment='top', horizontalalignment='right', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
            else:
                axes2[1,1].axis('off')

            # Plot 6: Ensemble Member Statistics
            if not is_labelless and self.results.ensemble_statistics and 'ensemble_member_stats' in self.results.ensemble_statistics:
                member_stats = self.results.ensemble_statistics['ensemble_member_stats']
                
                if 'global_means' in member_stats and 'global_stds' in member_stats:
                    member_indices = range(len(member_stats['global_means']))
                    
                    # Bar plot of global means
                    bars = axes2[1,2].bar(member_indices, member_stats['global_means'], 
                                         alpha=0.7, color='lightcoral', edgecolor='darkred')
                    axes2[1,2].set_title('📈 Global Mean by Ensemble Member', fontsize=14, fontweight='bold')
                    axes2[1,2].set_xlabel('Ensemble Member (Ranked)')
                    axes2[1,2].set_ylabel('Global Mean Height (m)')
                    axes2[1,2].grid(True, alpha=0.3)
                    
                    # Add value labels on bars
                    for i, (bar, value) in enumerate(zip(bars, member_stats['global_means'])):
                        height = bar.get_height()
                        axes2[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                    
                    # Add ranking method note
                    ranking_note = f"Ranked by: {self.config.ranking_method.value.replace('_', ' ').title()}"
                    axes2[1,2].text(0.02, 0.98, ranking_note, transform=axes2[1,2].transAxes,
                                   verticalalignment='top', fontsize=10,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
                else:
                    axes2[1,2].axis('off')
            elif not is_labelless:
                # Fallback: show processing summary
                axes2[1,2].axis('off')
                summary_text = f"""🎉 Ensemble Processing Summary

✅ Total Patches: {self.results.total_patches_processed}
✅ Successfully Stitched: {self.results.patches_successfully_stitched}
❌ Skipped: {self.results.patches_skipped}
📈 Success Rate: {(self.results.patches_successfully_stitched/self.results.total_patches_processed*100):.1f}%

⏱️ Processing Time: {self.results.processing_time:.2f}s
📏 Output Shape: {self.results.output_shape}
🎯 Ensemble Size: {self.config.ensemble_size}
📊 Ranking Method: {self.config.ranking_method.value.replace('_', ' ').title()}
📈 Uncertainty Method: {self.config.uncertainty_method.value.upper()}"""
                
                axes2[1,2].text(0.1, 0.9, summary_text, transform=axes2[1,2].transAxes,
                               verticalalignment='top', fontsize=11, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
            else:
                axes2[1,2].axis('off')

            if not is_labelless:
                # Overall title for uncertainty grid
                fig2.suptitle(f'Ensemble Uncertainty Analysis\n'
                             f'{self.config.uncertainty_method.value.upper()} Method • '
                             f'{self.results.patches_successfully_stitched} Patches • '
                             f'{self.config.ensemble_size} Members', 
                             fontsize=16, fontweight='bold', y=0.98)
            
            plt.tight_layout()
            
            if save_figures:
                if is_labelless:
                    uncertainty_grid_path = labelless_dir / "uncertainty_analysis_grid_labelless.png"
                else:
                    uncertainty_grid_path = self.experiment_dir / "uncertainty_analysis_grid.png"
                plt.savefig(uncertainty_grid_path, dpi=150, bbox_inches='tight', facecolor='white')  # Reduced DPI for speed
                logger.info(f"💾 Saved uncertainty grid: {uncertainty_grid_path}")
            
            if not is_labelless:  # Only show labeled version
                plt.show()
            else:
                plt.close()
        
        # =============================================================================
        # SUMMARY STATISTICS PRINT
        # =============================================================================
        
        print("\n" + "="*80)
        print("📊 ENSEMBLE VISUALIZATION SUMMARY")
        print("="*80)
        
        print(f"🎯 Configuration:")
        print(f"  • Ensemble Size: {self.config.ensemble_size}")
        print(f"  • Ranking Method: {self.config.ranking_method.value.replace('_', ' ').title()}")
        print(f"  • Uncertainty Method: {self.config.uncertainty_method.value.upper()}")
        print(f"  • Height Colormap: {colormap.upper()}")
        print(f"  • Uncertainty Colormap: MAGMA")
        print(f"  • Labelless Figures: {'ON' if self.config.save_labelless_figures else 'OFF'}")
        
        print(f"\n📈 Ensemble Statistics:")
        if self.results.ensemble_statistics:
            unc_stats = self.results.ensemble_statistics.get('uncertainty_stats', {})
            print(f"  • Uncertainty Mean: {unc_stats.get('mean', 0):.4f} m")
            print(f"  • Uncertainty Std: {unc_stats.get('std', 0):.4f} m")
            print(f"  • Uncertainty Range: [{unc_stats.get('min', 0):.4f}, {unc_stats.get('max', 0):.4f}] m")
            
            member_stats = self.results.ensemble_statistics.get('ensemble_member_stats', {})
            if 'global_means' in member_stats:
                global_means = member_stats['global_means']
                print(f"  • Member Mean Range: [{min(global_means):.3f}, {max(global_means):.3f}] m")
                print(f"  • Member Mean Spread: {max(global_means) - min(global_means):.3f} m")
        
        print(f"\n📁 Output Files:")
        if self.results.output_files:
            for file_type, file_path in self.results.output_files.items():
                print(f"  • {file_type.replace('_', ' ').title()}: {Path(file_path).name}")
        
        print(f"\n🎨 Visualizations Generated:")
        print(f"  • Ensemble Members Grid (2×5): All {self.config.ensemble_size} ranked members")
        print(f"  • Uncertainty Analysis (2×3): Mean, uncertainty, std, range, histogram, stats")
        if self.config.save_labelless_figures:
            print(f"  • Labelless Versions: Saved in 'labelless' subfolder")
        print(f"  • Geographic Projection: Correct pixel sizes with extent=[{left:.0f}, {right:.0f}, {bottom:.0f}, {top:.0f}]")
        print(f"  • Reduced DPI (150): Faster plotting for large images")
        
        print("="*80)

    def _calculate_output_dimensions(self) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """Calculate output grid dimensions and bounds."""
        min_row = min(p.metadata['grid_row_start'] for p in self.patches)
        max_row = max(p.metadata['grid_row_start'] for p in self.patches)
        min_col = min(p.metadata['grid_col_start'] for p in self.patches)
        max_col = max(p.metadata['grid_col_start'] for p in self.patches)

        output_height = (max_row - min_row) + self.config.patch_size
        output_width = (max_col - min_col) + self.config.patch_size

        return (output_height, output_width), (min_row, min_col, max_row, max_col)

    def _get_output_position(self, patch: EnsemblePatchData, grid_bounds: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get position of patch in output grid."""
        min_row, min_col, _, _ = grid_bounds
        row_start = patch.metadata['grid_row_start'] - min_row
        col_start = patch.metadata['grid_col_start'] - min_col
        return row_start, col_start

    def _calculate_final_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate final geographic bounds of stitched result."""
        all_bounds = [(p.metadata['bounds_left'], p.metadata['bounds_bottom'], 
                      p.metadata['bounds_right'], p.metadata['bounds_top']) for p in self.patches]
        left = min(b[0] for b in all_bounds)
        bottom = min(b[1] for b in all_bounds)
        right = max(b[2] for b in all_bounds)
        top = max(b[3] for b in all_bounds)
        return (left, bottom, right, top)

    def _create_metadata_summary(self) -> Dict[str, Any]:
        """Create summary of ensemble patch metadata."""
        return {
            'total_patches': len(self.patches),
            'patch_ids': [p.patch_id for p in self.patches],
            'ensemble_size': self.config.ensemble_size,
            'ranking_method': self.config.ranking_method.value,
            'uncertainty_method': self.config.uncertainty_method.value,
            'grid_extent': {
                'min_row': min(p.metadata['grid_row_start'] for p in self.patches),
                'max_row': max(p.metadata['grid_row_start'] for p in self.patches),
                'min_col': min(p.metadata['grid_col_start'] for p in self.patches),
                'max_col': max(p.metadata['grid_col_start'] for p in self.patches)
            },
            'crs': self.patches[0].metadata['crs'],
            'config': {
                'patch_size': self.config.patch_size,
                'patch_stride': self.config.patch_stride,
                'overlap_size': self.config.overlap_size,
                'ensemble_size': self.config.ensemble_size,
                'ranking_method': self.config.ranking_method.value,
                'uncertainty_method': self.config.uncertainty_method.value
            }
        }


# =============================================================================
# UTILITY FUNCTIONS FOR PIPELINE INTEGRATION
# =============================================================================

def create_ensemble_stitching_config_from_pipeline(**config_params) -> EnsembleStitchingConfig:
    """Create EnsembleStitchingConfig from pipeline configuration parameters."""
    return EnsembleStitchingConfig.from_pipeline_config(**config_params)


def run_ensemble_patch_stitching_pipeline(experiment_name: str,
                                         experiment_dir: Path,
                                         output_dir: Path,
                                         patch_indices: List[int],
                                         config: EnsembleStitchingConfig) -> EnsembleStitchingResults:
    """Run the complete ensemble patch stitching pipeline."""
    # Create ensemble stitching manager
    stitcher = EnsemblePatchStitchingManager(
        experiment_name=experiment_name,
        output_dir=output_dir,
        config=config
    )
    
    # Run complete workflow
    results = stitcher.run_complete_ensemble_stitching_workflow(
        experiment_dir=experiment_dir,
        patch_indices=patch_indices,
        output_dir=stitcher.experiment_dir
    )
    
    # Generate visualization if successful
    if results.success:
        stitcher.visualize_ensemble_results(save_figures=True)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("SAR2Height Ensemble Patch Stitching System")
    print("Available ranking methods:", [m.value for m in EnsembleRankingMethod])
    print("Available uncertainty methods:", [m.value for m in UncertaintyMethod])
    print("\nUse EnsemblePatchStitchingManager for complete workflow management.")