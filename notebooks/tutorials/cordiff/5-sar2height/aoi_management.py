"""
AOI (Area of Interest) Management Module for SAR-to-Height Data Processing

This module provides functionality to manage Areas of Interest (AOI) to prevent
overlapping patches across different datasets in the cordiff training pipeline.

Author: AI Assistant
Date: 2026-01-06
"""

import os
import glob
import numpy as np
import xarray as xr
from typing import List, Tuple, Dict, Optional, Set
from pathlib import Path
import logging
from shapely.geometry import box, Polygon
from shapely.ops import unary_union
import json

logger = logging.getLogger(__name__)


class AOIManager:
    """
    Class to manage Areas of Interest (AOI) and prevent spatial overlap between patches.
    
    This class provides methods to:
    - Load existing AOI information from NetCDF files
    - Check for spatial overlaps between patches
    - Manage AOI database for conflict resolution
    - Filter patches to avoid overlap with existing data
    """
    
    def __init__(self, patches_dir: str, skipped_patches_dir: str):
        """
        Initialize the AOI manager.
        
        Args:
            patches_dir (str): Directory containing existing patch NetCDF files
            skipped_patches_dir (str): Directory to store skipped patches
        """
        self.patches_dir = Path(patches_dir)
        self.skipped_patches_dir = Path(skipped_patches_dir)
        
        # Create directories if they don't exist
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.skipped_patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing AOI information
        self.existing_aois = self._load_existing_aois()
        
        logger.info(f"AOI Manager initialized with {len(self.existing_aois)} existing AOIs")
    
    def _load_existing_aois(self) -> List[Dict]:
        """
        Load AOI information from existing NetCDF files in the patches directory.
        
        Returns:
            List[Dict]: List of existing AOI information
        """
        existing_aois = []
        
        # Find all NetCDF files in patches directory
        nc_files = glob.glob(str(self.patches_dir / "*.nc"))
        
        for nc_file in nc_files:
            try:
                # Open NetCDF file and extract AOI information
                with xr.open_dataset(nc_file) as ds:
                    # Extract AOI information from global attributes
                    aoi_info = self._extract_aoi_from_netcdf(ds, nc_file)
                    if aoi_info:
                        existing_aois.append(aoi_info)
                        
            except Exception as e:
                logger.warning(f"Error reading AOI from {nc_file}: {str(e)}")
        
        logger.info(f"Loaded {len(existing_aois)} existing AOIs from {len(nc_files)} files")
        return existing_aois
    
    def _extract_aoi_from_netcdf(self, dataset: xr.Dataset, file_path: str) -> Optional[Dict]:
        """
        Extract AOI information from a NetCDF dataset.
        
        Args:
            dataset (xr.Dataset): Opened NetCDF dataset
            file_path (str): Path to the NetCDF file
            
        Returns:
            Optional[Dict]: AOI information or None if not found
        """
        try:
            # Try to extract from global attributes
            attrs = dataset.attrs
            
            # Look for AOI-related attributes
            aoi_keys = ['aoi_bounds', 'spatial_bounds', 'geographic_bounds']
            bounds = None
            
            for key in aoi_keys:
                if key in attrs:
                    bounds_str = attrs[key]
                    if isinstance(bounds_str, str):
                        # Parse bounds string (format: "left,bottom,right,top")
                        bounds = [float(x.strip()) for x in bounds_str.split(',')]
                        break
            
            if bounds is None:
                # Try to extract from coordinate variables
                if 'x_lr' in dataset.dims and 'y_lr' in dataset.dims:
                    # Assume regular grid - calculate bounds from dimensions
                    # This is a fallback method
                    logger.warning(f"No explicit bounds found in {file_path}, using dimension info")
                    return None
            
            # Extract other metadata
            aoi_info = {
                'file_path': file_path,
                'bounds': bounds,
                'polygon': box(*bounds) if bounds and len(bounds) == 4 else None,
                'satellite': attrs.get('satellite', 'unknown'),
                'orbit_id': attrs.get('orbit_id', 'unknown'),
                'datetime': attrs.get('acquisition_datetime', 'unknown'),
                'n_samples': dataset.dims.get('sample', 0)
            }
            
            return aoi_info
            
        except Exception as e:
            logger.warning(f"Error extracting AOI from {file_path}: {str(e)}")
            return None
    
    def check_overlap(self, new_bounds: Dict[str, float], 
                     overlap_threshold: float = 0.1) -> Tuple[bool, List[Dict]]:
        """
        Check if new bounds overlap with existing AOIs.
        
        Args:
            new_bounds (Dict[str, float]): New bounds to check
            overlap_threshold (float): Minimum overlap ratio to consider as conflict
            
        Returns:
            Tuple containing:
            - has_overlap: Boolean indicating if overlap exists
            - overlapping_aois: List of overlapping AOI information
        """
        if new_bounds.get('pixel_bounds', False):
            # Can't check overlap for pixel coordinates
            logger.warning("Cannot check overlap for pixel-based bounds")
            return False, []
        
        # Create polygon from new bounds
        new_polygon = box(
            new_bounds['left'], new_bounds['bottom'],
            new_bounds['right'], new_bounds['top']
        )
        
        overlapping_aois = []
        
        for existing_aoi in self.existing_aois:
            if existing_aoi['polygon'] is None:
                continue
            
            # Calculate intersection
            intersection = new_polygon.intersection(existing_aoi['polygon'])
            
            if intersection.is_empty:
                continue
            
            # Calculate overlap ratio
            overlap_area = intersection.area
            new_area = new_polygon.area
            existing_area = existing_aoi['polygon'].area
            
            # Overlap ratio relative to smaller polygon
            min_area = min(new_area, existing_area)
            overlap_ratio = overlap_area / min_area if min_area > 0 else 0
            
            if overlap_ratio > overlap_threshold:
                overlap_info = existing_aoi.copy()
                overlap_info['overlap_ratio'] = overlap_ratio
                overlap_info['overlap_area'] = overlap_area
                overlapping_aois.append(overlap_info)
        
        has_overlap = len(overlapping_aois) > 0
        
        if has_overlap:
            logger.debug(f"Found {len(overlapping_aois)} overlapping AOIs")
        
        return has_overlap, overlapping_aois
    
    def filter_non_overlapping_patches(self, patches: List[Dict],
                                     overlap_threshold: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter patches to remove those that overlap with existing AOIs.
        
        Args:
            patches (List[Dict]): List of patches to check
            overlap_threshold (float): Minimum overlap ratio to consider as conflict
            
        Returns:
            Tuple containing:
            - non_overlapping_patches: Patches that don't overlap
            - overlapping_patches: Patches that overlap with existing AOIs
        """
        non_overlapping_patches = []
        overlapping_patches = []
        
        for patch in patches:
            bounds = patch['bounds']
            has_overlap, overlapping_aois = self.check_overlap(bounds, overlap_threshold)
            
            if has_overlap:
                # Add overlap information to patch
                patch['overlap_info'] = overlapping_aois
                overlapping_patches.append(patch)
                
                logger.debug(f"Patch {patch['patch_id']} overlaps with "
                           f"{len(overlapping_aois)} existing AOIs")
            else:
                non_overlapping_patches.append(patch)
        
        logger.info(f"Filtered patches: {len(non_overlapping_patches)} non-overlapping, "
                   f"{len(overlapping_patches)} overlapping")
        
        return non_overlapping_patches, overlapping_patches
    
    def register_new_aoi(self, patches: List[Dict], file_path: str, 
                        base_name: str, satellite: str = 'unknown',
                        orbit_id: str = 'unknown', datetime: str = 'unknown'):
        """
        Register a new AOI after saving patches to NetCDF.
        
        Args:
            patches (List[Dict]): List of patches that were saved
            file_path (str): Path to the saved NetCDF file
            base_name (str): Base name of the data file
            satellite (str): Satellite identifier
            orbit_id (str): Orbit identifier
            datetime (str): Acquisition datetime
        """
        if not patches:
            logger.warning("No patches provided for AOI registration")
            return
        
        # Calculate combined bounds from all patches
        all_bounds = [patch['bounds'] for patch in patches 
                     if not patch['bounds'].get('pixel_bounds', False)]
        
        if not all_bounds:
            logger.warning("No geographic bounds available for AOI registration")
            return
        
        # Calculate overall bounding box
        left = min(bounds['left'] for bounds in all_bounds)
        bottom = min(bounds['bottom'] for bounds in all_bounds)
        right = max(bounds['right'] for bounds in all_bounds)
        top = max(bounds['top'] for bounds in all_bounds)
        
        # Create AOI information
        aoi_info = {
            'file_path': file_path,
            'bounds': [left, bottom, right, top],
            'polygon': box(left, bottom, right, top),
            'satellite': satellite,
            'orbit_id': orbit_id,
            'datetime': datetime,
            'n_samples': len(patches),
            'base_name': base_name
        }
        
        # Add to existing AOIs
        self.existing_aois.append(aoi_info)
        
        logger.info(f"Registered new AOI: {base_name} with {len(patches)} patches")
    
    def get_aoi_summary(self) -> Dict:
        """
        Get summary statistics about existing AOIs.
        
        Returns:
            Dict: Summary information about AOIs
        """
        if not self.existing_aois:
            return {
                'total_aois': 0,
                'total_patches': 0,
                'satellites': [],
                'date_range': None,
                'spatial_coverage': None
            }
        
        # Calculate statistics
        total_aois = len(self.existing_aois)
        total_patches = sum(aoi.get('n_samples', 0) for aoi in self.existing_aois)
        
        # Get unique satellites
        satellites = list(set(aoi.get('satellite', 'unknown') for aoi in self.existing_aois))
        
        # Get date range
        dates = [aoi.get('datetime', '') for aoi in self.existing_aois if aoi.get('datetime', '')]
        date_range = None
        if dates:
            try:
                # Extract date part (first 8 characters: YYYYMMDD)
                date_strings = [d[:8] for d in dates if len(d) >= 8]
                if date_strings:
                    date_range = {'earliest': min(date_strings), 'latest': max(date_strings)}
            except Exception as e:
                logger.warning(f"Error parsing dates: {str(e)}")
        
        # Calculate spatial coverage (union of all AOI polygons)
        valid_polygons = [aoi['polygon'] for aoi in self.existing_aois 
                         if aoi['polygon'] is not None]
        
        spatial_coverage = None
        if valid_polygons:
            try:
                union_polygon = unary_union(valid_polygons)
                bounds = union_polygon.bounds  # (minx, miny, maxx, maxy)
                spatial_coverage = {
                    'total_area': union_polygon.area,
                    'bounds': bounds,
                    'coverage_polygons': len(valid_polygons)
                }
            except Exception as e:
                logger.warning(f"Error calculating spatial coverage: {str(e)}")
        
        return {
            'total_aois': total_aois,
            'total_patches': total_patches,
            'satellites': satellites,
            'date_range': date_range,
            'spatial_coverage': spatial_coverage
        }
    
    def save_aoi_database(self, output_path: Optional[str] = None):
        """
        Save AOI database to JSON file for backup and analysis.
        
        Args:
            output_path (str, optional): Path to save JSON file
        """
        if output_path is None:
            output_path = self.patches_dir / "aoi_database.json"
        
        # Convert AOI information to JSON-serializable format
        serializable_aois = []
        for aoi in self.existing_aois:
            aoi_copy = aoi.copy()
            # Convert polygon to bounds for serialization
            if aoi_copy['polygon'] is not None:
                aoi_copy['polygon'] = list(aoi_copy['polygon'].bounds)
            serializable_aois.append(aoi_copy)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'aois': serializable_aois,
                'summary': self.get_aoi_summary(),
                'created_date': str(np.datetime64('now'))
            }, f, indent=2)
        
        logger.info(f"Saved AOI database to {output_path}")
    
    def load_aoi_database(self, input_path: str):
        """
        Load AOI database from JSON file.
        
        Args:
            input_path (str): Path to JSON file
        """
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct AOI information
            self.existing_aois = []
            for aoi_data in data.get('aois', []):
                aoi_copy = aoi_data.copy()
                # Reconstruct polygon from bounds
                if aoi_copy['polygon'] is not None and isinstance(aoi_copy['polygon'], list):
                    bounds = aoi_copy['polygon']
                    if len(bounds) == 4:
                        aoi_copy['polygon'] = box(*bounds)
                    else:
                        aoi_copy['polygon'] = None
                self.existing_aois.append(aoi_copy)
            
            logger.info(f"Loaded {len(self.existing_aois)} AOIs from {input_path}")
            
        except Exception as e:
            logger.error(f"Error loading AOI database from {input_path}: {str(e)}")


def check_patch_overlaps(patches: List[Dict], 
                        patches_dir: str,
                        overlap_threshold: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
    """
    Convenience function to check patch overlaps with existing AOIs.
    
    Args:
        patches (List[Dict]): List of patches to check
        patches_dir (str): Directory containing existing patches
        overlap_threshold (float): Overlap threshold for filtering
        
    Returns:
        Tuple of non-overlapping and overlapping patches
    """
    # Create temporary skipped directory name
    skipped_dir = str(Path(patches_dir).parent / "patches_skipped")
    
    # Initialize AOI manager
    aoi_manager = AOIManager(patches_dir, skipped_dir)
    
    # Filter patches
    return aoi_manager.filter_non_overlapping_patches(patches, overlap_threshold)


if __name__ == "__main__":
    # Example usage
    patches_dir = "/home/younes.abid/git/physicsnemo/data/sar2height/patches"
    skipped_dir = "/home/younes.abid/git/physicsnemo/data/sar2height/patches_skipped"
    
    # Initialize AOI manager
    aoi_manager = AOIManager(patches_dir, skipped_dir)
    
    # Get summary
    summary = aoi_manager.get_aoi_summary()
    print("AOI Summary:", summary)
    
    # Save database
    aoi_manager.save_aoi_database()