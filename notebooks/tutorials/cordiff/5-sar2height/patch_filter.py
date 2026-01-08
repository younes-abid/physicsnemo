"""
NetCDF Patch Filtering with Group Structure Preservation

This module filters SAR-to-height patches while preserving the complete NetCDF group structure:
- input group: Multi-channel SAR features
- output group: DSM height data
- patch_metadata group: Complete spatial and quality metadata

The filtered files maintain the same structure as full_patch.nc for cordiff compatibility.
Uses consistent thresholds and fast NetCDF writing approach.
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
import os  # Add missing os import
from typing import List, Optional, Tuple, Dict, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import time

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    # Fallback to simple rectangular overlap detection
    class Polygon:
        def __init__(self, coords):
            # Simple rectangular bounds
            x_coords = [c[0] for c in coords]
            y_coords = [c[1] for c in coords]
            self.bounds = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            self.area = (self.bounds[2] - self.bounds[0]) * (self.bounds[3] - self.bounds[1])
        
        def intersects(self, other):
            if hasattr(other, 'bounds'):
                return not (self.bounds[2] <= other.bounds[0] or 
                           self.bounds[0] >= other.bounds[2] or
                           self.bounds[3] <= other.bounds[1] or
                           self.bounds[1] >= other.bounds[3])
            return False
        
        def intersection(self, other):
            if not self.intersects(other):
                return Polygon([(0,0), (0,0), (0,0), (0,0)])
            
            x1 = max(self.bounds[0], other.bounds[0])
            y1 = max(self.bounds[1], other.bounds[1])
            x2 = min(self.bounds[2], other.bounds[2])
            y2 = min(self.bounds[3], other.bounds[3])
            
            return Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def unary_union(polygons):
        if not polygons:
            return None
        
        # Simple union using bounding box
        all_bounds = [p.bounds for p in polygons]
        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)
        
        return Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

logger = logging.getLogger(__name__)

@dataclass
class FilteringResults:
    """Results from patch filtering operation"""
    success: bool
    error: Optional[str] = None
    input_files: List[str] = None
    output_files: List[str] = None
    total_patches_processed: int = 0
    patches_kept: int = 0
    patches_passed_quality: int = 0  # Add missing attribute
    patches_passed_spatial: int = 0   # Add missing attribute
    filtering_time: float = 0.0       # Add missing attribute
    filtering_report: Dict = None     # Add missing attribute (alias for processing_details)
    processing_details: Dict = None
    
    def __post_init__(self):
        """Ensure filtering_report is set if processing_details exists"""
        if self.processing_details is not None and self.filtering_report is None:
            self.filtering_report = self.processing_details

@dataclass
class QualityThresholds:
    """Quality filtering thresholds - CONSISTENT across all filtering."""
    min_dsm_valid_percentage: float = 70.0
    min_feature_valid_percentage: float = 70.0
    min_dsm_std: float = 0.1
    max_dsm_std: float = 1000.0
    min_data_completeness: float = 70.0
    min_dsm_range: float = 0.5

def filter_netcdf_with_groups_fast(src_file, dst_file, quality_thresholds, selected_indices, existing_coverage=None):
    """
    Simple and fast filtering function that preserves the original NetCDF structure.
    Uses the same clean approach as the original NetCDF writer.
    """
    
    logger = logging.getLogger(__name__)
    
    # Simple compression settings (same as writer)
    compression_opts = {'zlib': True, 'complevel': 4}
    
    start_time = time.time()
    n_filtered = len(selected_indices)
    
    logger.info(f"Starting filtering: {n_filtered} patches to keep")
    
    try:
        # Open files
        open_start = time.time()
        src = nc.Dataset(src_file, 'r')
        dst = nc.Dataset(dst_file, 'w', format='NETCDF4')
        logger.info(f"  File opening: {time.time() - open_start:.3f}s")
        
        # Copy global attributes
        attr_start = time.time()
        logger.info("  Copying global attributes...")
        for attr_name in src.ncattrs():
            try:
                dst.setncattr(attr_name, src.getncattr(attr_name))
                logger.info(f"    Copied global attribute: {attr_name}")
            except:
                logger.warning(f"  Could not copy global attribute: {attr_name}")
        logger.info(f"  Global attributes: {time.time() - attr_start:.3f}s")
        
        # Copy dimensions
        dim_start = time.time()
        logger.info("  Copying dimensions...")
        dst.createDimension('sample', n_filtered)
        for dim_name, dim in src.dimensions.items():
            if dim_name != 'sample':
                dst.createDimension(dim_name, len(dim) if not dim.isunlimited() else None)
        logger.info(f"  Dimensions: {time.time() - dim_start:.3f}s")
        
        # Pre-convert indices for faster indexing
        indices = np.array(selected_indices, dtype=np.int32)
        
        # Copy variables (fast and simple)
        var_start = time.time()
        logger.info("  Copying variables...")
        total_data = 0
        
        for var_name, var in src.variables.items():
            # Create variable with same properties
            dst_var = dst.createVariable(
                var_name, 
                var.dtype, 
                var.dimensions,
                **compression_opts
            )
            
            # Copy attributes
            for attr_name in var.ncattrs():
                try:
                    dst_var.setncattr(attr_name, var.getncattr(attr_name))
                    logger.info(f"    Copied attribute '{attr_name}' for variable '{var_name}'")
                except:
                    logger.warning(f"  Could not copy attribute '{attr_name}' for variable '{var_name}'")
            
            # Copy data with filtering
            if 'sample' in var.dimensions:
                if var_name == 'sample':
                    # Special case: sample index
                    dst_var[:] = np.arange(n_filtered, dtype=var.dtype)
                else:
                    # Filtered data
                    data = var[indices]
                    dst_var[:] = data
                    total_data += data.nbytes
            else:
                # Non-filtered data
                data = var[:]
                dst_var[:] = data
                total_data += data.nbytes
        
        logger.info(f"  Variables: {time.time() - var_start:.3f}s, "
                   f"data: {total_data/1024/1024:.1f}MB")
        
        # Copy groups (simplified and fast)
        groups_start = time.time()
        logger.info("  Copying groups...")
        
        for group_name in ['input', 'output', 'patch_metadata']:
            logger.info(f"    Processing group: {group_name}")
            if group_name not in src.groups:
                continue
            
            group_start = time.time()
            src_group = src.groups[group_name]
            dst_group = dst.createGroup(group_name)
            
            # Copy group attributes
            for attr_name in src_group.ncattrs():
                try:
                    dst_group.setncattr(attr_name, src_group.getncattr(attr_name))
                    logger.info(f"    Copied attribute '{attr_name}' for group '{group_name}'")
                except:
                    logger.warning(f"  Could not copy attribute '{attr_name}' for group '{group_name}'")    
            
            # Update patch count for metadata group
            if group_name == 'patch_metadata':
                try:
                    dst_group.setncattr('n_patches', n_filtered)
                    logger.info("    Set 'n_patches' attribute for group 'patch_metadata'")
                except:
                    logger.warning("  Could not set 'n_patches' attribute for group 'patch_metadata'")
            
            # Copy variables in group
            group_data = 0
            
            for var_name, var in src_group.variables.items():
                # Create variable
                dst_var = dst_group.createVariable(
                    var_name, 
                    var.dtype, 
                    var.dimensions,
                    **compression_opts
                )
                
                # Copy attributes
                for attr_name in var.ncattrs():
                    try:
                        dst_var.setncattr(attr_name, var.getncattr(attr_name))
                        logger.info(f"    Copied attribute '{attr_name}' for variable '{var_name}'")
                    except:
                        logger.warning(f"  Could not copy attribute '{attr_name}' for variable '{var_name}'")
                
                # Copy data
                if 'sample' in var.dimensions:
                    data = var[indices]
                    dst_var[:] = data
                    group_data += data.nbytes
                else:
                    data = var[:]
                    dst_var[:] = data
                    group_data += data.nbytes
            
            logger.info(f"    Group '{group_name}': {time.time() - group_start:.3f}s, "
                       f"data: {group_data/1024/1024:.1f}MB")
        
        logger.info(f"  Groups: {time.time() - groups_start:.3f}s")
        
        # Update global attributes for filtering
        try:
            dst.setncattr('filtering_applied', 1)
            dst.setncattr('patches_kept_after_filtering', n_filtered)
        except:
            logger.warning("  Could not set global filtering attributes")
        
        # Close files
        src.close()
        dst.close()
        
        total_time = time.time() - start_time
        logger.info(f"Filtering complete in {total_time:.3f}s")
        
        return n_filtered
        
    except Exception as e:
        logger.error(f"Error in filtering: {e}")
        
        # Clean up on error
        if 'src' in locals():
            try:
                src.close()
            except:
                pass
        
        if 'dst' in locals():
            try:
                dst.close()
            except:
                pass
        
        if os.path.exists(dst_file):
            try:
                os.remove(dst_file)
                logger.info(f"Cleaned up corrupted file: {dst_file}")
            except:
                pass
        
        raise
    
def get_patch_bounds(input_file: str) -> List[Polygon]:
    """Extract spatial bounds for all patches in a file as Polygon objects"""
    
    with nc.Dataset(input_file, 'r') as ds:
        metadata = ds.groups['patch_metadata']
        
        # Get patch bounds (handle different variable names for backward compatibility)
        if 'bounds_left' in metadata.variables:
            # New structure from netcdf_writer.py
            left = metadata.variables['bounds_left'][:]
            right = metadata.variables['bounds_right'][:]
            bottom = metadata.variables['bounds_bottom'][:]
            top = metadata.variables['bounds_top'][:]
        elif 'min_x' in metadata.variables:
            # Legacy structure
            left = metadata.variables['min_x'][:]
            right = metadata.variables['max_x'][:]
            bottom = metadata.variables['min_y'][:]
            top = metadata.variables['max_y'][:]
        else:
            raise ValueError("No spatial bounds variables found in patch metadata")
        
        # Create polygon for each patch
        polygons = []
        for i in range(len(left)):
            # Create rectangle polygon from bounds
            polygon = Polygon([
                (left[i], bottom[i]),  # bottom-left
                (right[i], bottom[i]), # bottom-right
                (right[i], top[i]),    # top-right
                (left[i], top[i])      # top-left
            ])
            polygons.append(polygon)
        
        return polygons


def get_existing_coverage(output_dir: str, file_prefix: str) -> Optional[Polygon]:
    """Get union of all existing filtered patch coverage areas"""
    
    output_path = Path(output_dir)
    
    # Find all existing filtered files
    pattern = f"{file_prefix}_*.nc"
    existing_files = list(output_path.glob(pattern))
    
    if not existing_files:
        return None
    
    logger.info(f"  Found {len(existing_files)} existing filtered files")
    
    all_polygons = []
    
    for file_path in existing_files:
        try:
            polygons = get_patch_bounds(str(file_path))
            all_polygons.extend(polygons)
        except Exception as e:
            logger.warning(f"  Could not read bounds from {file_path.name}: {e}")
            continue
    
    if not all_polygons:
        return None
    
    # Create union of all existing patch areas
    try:
        existing_coverage = unary_union(all_polygons)
        logger.info(f"  Existing coverage: {len(all_polygons)} patches combined")
        return existing_coverage
    except Exception as e:
        logger.warning(f"  Could not create coverage union: {e}")
        return None


def get_spatial_filter_mask(input_file: str, existing_coverage: Optional[Polygon], overlap_tolerance: float) -> np.ndarray:
    """Get boolean mask for spatial filtering based on overlap with existing patches"""
    
    with nc.Dataset(input_file, 'r') as ds:
        n_total = ds.dimensions['sample'].size
    
    # If no existing coverage, all patches pass spatial filter
    if existing_coverage is None:
        logger.info(f"  No existing coverage found - all patches pass spatial filter")
        return np.ones(n_total, dtype=bool)
    
    # Get patch bounds for current file
    current_polygons = get_patch_bounds(input_file)
    
    # Create mask based on overlap tolerance
    spatial_mask = np.ones(len(current_polygons), dtype=bool)
    
    for i, patch_polygon in enumerate(current_polygons):
        try:
            # Check if patch intersects with existing coverage
            if patch_polygon.intersects(existing_coverage):
                # Calculate overlap area
                intersection = patch_polygon.intersection(existing_coverage)
                overlap_area = intersection.area
                patch_area = patch_polygon.area
                
                # Calculate overlap percentage
                overlap_percentage = (overlap_area / patch_area) if patch_area > 0 else 1.0
                
                # Filter out if overlap exceeds tolerance
                if overlap_percentage > overlap_tolerance:
                    spatial_mask[i] = False
                    
        except Exception as e:
            logger.warning(f"  Error checking overlap for patch {i}: {e}")
            # On error, be conservative and filter out the patch
            spatial_mask[i] = False
    
    n_passed = np.sum(spatial_mask)
    logger.info(f"  Spatial filter: {n_passed}/{len(spatial_mask)} patches passed (overlap_tolerance={overlap_tolerance})")
    
    return spatial_mask


def get_filter_mask(input_file: str, quality_thresholds: QualityThresholds) -> np.ndarray:
    """Get boolean mask for quality filtering"""
    
    with nc.Dataset(input_file, 'r') as ds:
        metadata = ds.groups['patch_metadata']
        n_total = ds.dimensions['sample'].size
        
        # Default mask - all pass
        mask = np.ones(n_total, dtype=bool)
        
        # Apply quality filters if variables exist
        if 'dsm_valid_percentage' in metadata.variables:
            mask &= (metadata.variables['dsm_valid_percentage'][:] >= quality_thresholds.min_dsm_valid_percentage)
            
        if 'feature_valid_percentage_min' in metadata.variables:
            mask &= (metadata.variables['feature_valid_percentage_min'][:] >= quality_thresholds.min_feature_valid_percentage)
            
        if 'dsm_std' in metadata.variables:
            dsm_std = metadata.variables['dsm_std'][:]
            mask &= (dsm_std >= quality_thresholds.min_dsm_std) & (dsm_std <= quality_thresholds.max_dsm_std)
            
        if 'min_data_completeness' in metadata.variables:
            mask &= (metadata.variables['min_data_completeness'][:] >= quality_thresholds.min_data_completeness)
            
        if 'dsm_range' in metadata.variables:
            mask &= (metadata.variables['dsm_range'][:] >= quality_thresholds.min_dsm_range)
        
        return mask


def filter_single_file(input_file: str, 
                      output_file: str, 
                      quality_thresholds: QualityThresholds,
                      existing_coverage: Optional[Polygon] = None,
                      overlap_tolerance: float = 0.0) -> Tuple[int, Dict]:
    """Filter one file while preserving NetCDF group structure - FAST version with proper error handling"""
    
    output_path = Path(output_file)
    
    try:
        # Get quality mask
        quality_mask = get_filter_mask(input_file, quality_thresholds)
        n_quality_passed = np.sum(quality_mask)
        
        # Get spatial mask
        spatial_mask = get_spatial_filter_mask(input_file, existing_coverage, overlap_tolerance)
        n_spatial_passed = np.sum(spatial_mask)
        
        # Combined mask
        combined_mask = quality_mask & spatial_mask
        indices = np.where(combined_mask)[0]
        n_kept = len(indices)
        
        stats = {
            'total_patches': len(quality_mask),
            'quality_passed': n_quality_passed,
            'spatial_passed': n_spatial_passed,
            'final_kept': n_kept
        }
        
        logger.info(f"  {Path(input_file).name}: {n_kept}/{len(combined_mask)} patches kept "
                   f"(quality: {n_quality_passed}, spatial: {n_spatial_passed})")
        
        if n_kept > 0:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save filtered file with FAST preserved group structure
            filter_netcdf_with_groups_fast(input_file, str(output_path), quality_thresholds, indices, existing_coverage)
            
            # Verify the file was created successfully
            if not output_path.exists():
                raise RuntimeError(f"Output file was not created: {output_path}")
                
            return n_kept, stats
        else:
            return 0, stats
            
    except Exception as e:
        logger.error(f"  Error processing {Path(input_file).name}: {e}")
        
        # Clean up any partially created output file
        if output_path.exists():
            try:
                output_path.unlink()
                logger.info(f"  Cleaned up partial file: {output_path}")
            except Exception as cleanup_error:
                logger.warning(f"  Could not clean up partial file {output_path}: {cleanup_error}")
        
        return 0, {'error': str(e), 'total_patches': 0, 'quality_passed': 0, 'spatial_passed': 0, 'final_kept': 0}


def filter_patch_files(input_files: List[str],
                      output_dir: str,
                      quality_thresholds: Optional[QualityThresholds] = None,
                      overlap_tolerance: float = 0.0,
                      file_prefix: str = "filtered") -> FilteringResults:
    """
    SIMPLE patch filtering with spatial overlap support and CONSISTENT thresholds
    
    Args:
        input_files: List of input NetCDF files to filter
        output_dir: Output directory for filtered files
        quality_thresholds: Quality filtering criteria (CONSISTENT - no adaptation)
        overlap_tolerance: Spatial overlap tolerance (0.0 = no overlap allowed, 1.0 = full overlap allowed)
        file_prefix: Prefix for output filenames
        
    Returns:
        FilteringResults object with details of the filtering operation
    """
    
    if quality_thresholds is None:
        quality_thresholds = QualityThresholds()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_files = []
    total_stats = {
        'total_patches_processed': 0,
        'patches_passed_quality': 0,
        'patches_passed_spatial': 0,
        'patches_kept': 0,
        'files_processed': 0,
        'files_created': 0
    }
    
    logger.info(f"🚀 FAST filtering {len(input_files)} files with CONSISTENT thresholds")
    logger.info(f"Thresholds: DSM≥{quality_thresholds.min_dsm_valid_percentage}%, "
               f"Features≥{quality_thresholds.min_feature_valid_percentage}%, "
               f"Overlap≤{overlap_tolerance*100:.1f}%")
    
    if not SHAPELY_AVAILABLE:
        logger.warning("Shapely not available - using simple rectangular overlap detection")
    
    for input_file in input_files:
        input_path = Path(input_file)
        
        # Create output filename
        base_name = input_path.stem.replace('full_patches_', '').replace('_patches', '')
        output_name = f"{file_prefix}_{base_name}.nc"
        output_path = output_dir / output_name
        
        # Get existing coverage before processing this file
        existing_coverage = get_existing_coverage(str(output_dir), file_prefix)
        
        # Filter file with FAST processing
        n_kept, file_stats = filter_single_file(
            str(input_path), 
            str(output_path), 
            quality_thresholds,
            existing_coverage,
            overlap_tolerance
        )
        
        # Update total statistics
        total_stats['files_processed'] += 1
        if 'error' not in file_stats:
            total_stats['total_patches_processed'] += file_stats['total_patches']
            total_stats['patches_passed_quality'] += file_stats['quality_passed']
            total_stats['patches_passed_spatial'] += file_stats['spatial_passed']
            total_stats['patches_kept'] += file_stats['final_kept']
        
        if n_kept > 0:
            output_files.append(str(output_path))
            total_stats['files_created'] += 1
        else:
            # Remove empty output file if it was created
            if output_path.exists():
                output_path.unlink()
    
    # Create filtering report
    filtering_report = {
        'filtering_stats': total_stats,
        'thresholds_used': {
            'min_dsm_valid_percentage': quality_thresholds.min_dsm_valid_percentage,
            'min_feature_valid_percentage': quality_thresholds.min_feature_valid_percentage,
            'min_dsm_std': quality_thresholds.min_dsm_std,
            'max_dsm_std': quality_thresholds.max_dsm_std,
            'min_data_completeness': quality_thresholds.min_data_completeness,
            'min_dsm_range': quality_thresholds.min_dsm_range,
            'overlap_tolerance': overlap_tolerance
        },
        'output_files': output_files,
        'processing_summary': {
            'input_files': len(input_files),
            'output_files': len(output_files),
            'total_patches_kept': total_stats['patches_kept'],
            'filtering_efficiency': total_stats['patches_kept'] / max(1, total_stats['total_patches_processed'])
        }
    }
    
    logger.info("✅ FAST filtering complete!")
    logger.info(f"📊 Results: {total_stats['patches_kept']}/{total_stats['total_patches_processed']} patches kept "
               f"({filtering_report['processing_summary']['filtering_efficiency']*100:.1f}% efficiency)")
    logger.info(f"📁 Created {len(output_files)} filtered files")
    
    return FilteringResults(
        success=True,
        input_files=input_files,
        output_files=output_files,
        total_patches_processed=total_stats['total_patches_processed'],
        patches_kept=total_stats['patches_kept'],
        processing_details=filtering_report
    )


