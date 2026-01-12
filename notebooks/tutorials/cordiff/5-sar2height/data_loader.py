"""
SAR-to-Height Data Loader Module

This module provides functionality to load, inspect, and manage SAR intensity and DSM TIFF files
for the cordiff training data preparation pipeline.

Author: AI Assistant
Date: 2026-01-06
"""

import os
import glob
import rasterio
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime
from rasterio.warp import transform_bounds
from rasterio.crs import CRS
from shapely.geometry import box, Polygon
from shapely.ops import transform
import pyproj

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataRange:
    """Data range statistics container"""
    min_val: float
    max_val: float
    mean_val: float
    std_val: float
    
    
@dataclass
class FileCounts:
    """File count statistics container"""
    sar_files: int
    dsm_files: int
    paired_files: int


@dataclass
class QualityMetrics:
    """Data quality metrics container"""
    valid_data_percentage: float
    correlation_score: Optional[float] = None
    spatial_alignment_score: Optional[float] = None


@dataclass
class DataRanges:
    """Container for all data ranges"""
    sar_intensity: Optional[DataRange] = None
    dsm_elevation: Optional[DataRange] = None


@dataclass
class AOIInfo:
    """Area of Interest information"""
    satellite: str
    orbit_id: str
    datetime: str
    date: str
    time: str
    full_name: str
    bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy) in original CRS
    bounds_wgs84: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy) in WGS84
    polygon_wgs84: Optional[Polygon] = None  # Shapely polygon in WGS84
    crs: Optional[str] = None  # Original CRS
    area_km2: Optional[float] = None  # Area in square kilometers
    
    
class DatasetStats:
    """
    Comprehensive statistics container for SAR dataset analysis.
    
    This class organizes all dataset statistics in a structured way
    that's compatible with the visualization module.
    """
    
    def __init__(self, raw_stats: Dict):
        """
        Initialize DatasetStats from raw statistics dictionary.
        
        Args:
            raw_stats: Raw statistics dictionary from get_data_statistics()
        """
        self.raw_stats = raw_stats
        self._process_stats()
    
    def _process_stats(self):
        """Process raw statistics into structured format"""
        stats = self.raw_stats
        
        # File counts
        self.file_counts = FileCounts(
            sar_files=len(stats.get('data_ranges', {}).get('intensity', {}).get('min', [])),
            dsm_files=len(stats.get('data_ranges', {}).get('dsm', {}).get('min', [])),
            paired_files=stats.get('validation_summary', {}).get('valid_pairs', 0)
        )
        
        # Data ranges
        self.data_ranges = DataRanges()
        
        if stats.get('aggregated_stats', {}).get('intensity'):
            int_stats = stats['aggregated_stats']['intensity']
            self.data_ranges.sar_intensity = DataRange(
                min_val=int_stats.get('global_min', 0),
                max_val=int_stats.get('global_max', 0),
                mean_val=int_stats.get('mean_of_means', 0),
                std_val=int_stats.get('mean_of_stds', 0)
            )
        
        if stats.get('aggregated_stats', {}).get('dsm'):
            dsm_stats = stats['aggregated_stats']['dsm']
            self.data_ranges.dsm_elevation = DataRange(
                min_val=dsm_stats.get('global_min', 0),
                max_val=dsm_stats.get('global_max', 0),
                mean_val=dsm_stats.get('mean_of_means', 0),
                std_val=dsm_stats.get('mean_of_stds', 0)
            )
        
        # Quality metrics
        valid_pairs = stats.get('validation_summary', {}).get('valid_pairs', 0)
        total_pairs = stats.get('validation_summary', {}).get('total_pairs', 1)
        valid_percentage = (valid_pairs / max(total_pairs, 1)) * 100
        
        self.quality_metrics = QualityMetrics(
            valid_data_percentage=valid_percentage
        )
        
        # Satellite distribution
        self.satellite_distribution = {}
        if 'unique_values' in stats and 'satellites' in stats['unique_values']:
            satellites = stats['summary']['satellites']
            for satellite in stats['unique_values']['satellites']:
                self.satellite_distribution[satellite] = satellites.count(satellite)
        
        # AOI information
        self.aoi_list = []
        if 'pair_details' in stats:
            for pair in stats['pair_details']:
                aoi_info = pair.get('aoi_info', {})
                aoi = AOIInfo(
                    satellite=aoi_info.get('satellite', 'unknown'),
                    orbit_id=aoi_info.get('orbit_id', 'unknown'),
                    datetime=aoi_info.get('datetime', 'unknown'),
                    date=aoi_info.get('date', 'unknown'),
                    time=aoi_info.get('time', 'unknown'),
                    full_name=aoi_info.get('full_name', 'unknown')
                )
                self.aoi_list.append(aoi)
    
    def get_aoi_overlap_analysis(self) -> Dict[str, Any]:
        """
        Analyze AOI overlaps based on acquisition times and orbits.
        
        Returns:
            Dictionary containing overlap analysis results
        """
        # Group by satellite and date for overlap analysis
        date_groups = {}
        satellite_groups = {}
        
        for aoi in self.aoi_list:
            # Group by date
            if aoi.date not in date_groups:
                date_groups[aoi.date] = []
            date_groups[aoi.date].append(aoi)
            
            # Group by satellite
            if aoi.satellite not in satellite_groups:
                satellite_groups[aoi.satellite] = []
            satellite_groups[aoi.satellite].append(aoi)
        
        # Find potential overlaps (same date, different orbits)
        overlaps = []
        for date, aois in date_groups.items():
            if len(aois) > 1:
                for i in range(len(aois)):
                    for j in range(i + 1, len(aois)):
                        if aois[i].orbit_id != aois[j].orbit_id:
                            overlaps.append({
                                'date': date,
                                'aoi1': aois[i],
                                'aoi2': aois[j],
                                'overlap_type': 'temporal'
                            })
        
        return {
            'total_aois': len(self.aoi_list),
            'date_groups': {date: len(aois) for date, aois in date_groups.items()},
            'satellite_groups': {sat: len(aois) for sat, aois in satellite_groups.items()},
            'potential_overlaps': overlaps,
            'overlap_count': len(overlaps)
        }


def _load_and_analyze_pair(pair_data: Dict[str, Any], return_data: bool = False) -> Dict[str, Any]:
    """
    Load and analyze a single file pair (for multiprocessing).
    
    Args:
        pair_data: Dictionary containing file pair information
        return_data: Whether to return the loaded data arrays
        
    Returns:
        Dictionary containing analysis results and optionally data
    """
    try:
        result = {
            'pair_info': pair_data.copy(),
            'validation': {
                'files_exist': True,
                'readable': True,
                'same_dimensions': False,
                'same_crs': False,
                'valid_data_ranges': True,
                'no_invalid_values': True
            },
            'data_stats': {
                'intensity': {},
                'dsm': {}
            }
        }
        
        # Check if files exist
        if not (os.path.exists(pair_data['intensity']) and os.path.exists(pair_data['dsm'])):
            result['validation']['files_exist'] = False
            return result
        
        # Load intensity data
        with rasterio.open(pair_data['intensity']) as src_int:
            intensity_data = src_int.read(1)
            intensity_meta = src_int.meta.copy()
            intensity_meta.update({
                'transform': src_int.transform,
                'crs': src_int.crs,
                'file_path': pair_data['intensity']
            })
        
        # Load DSM data
        with rasterio.open(pair_data['dsm']) as src_dsm:
            dsm_data = src_dsm.read(1)
            dsm_meta = src_dsm.meta.copy()
            dsm_meta.update({
                'transform': src_dsm.transform,
                'crs': src_dsm.crs,
                'file_path': pair_data['dsm']
            })
        
        # Validation checks
        result['validation']['same_dimensions'] = intensity_data.shape == dsm_data.shape
        result['validation']['same_crs'] = intensity_meta['crs'] == dsm_meta['crs']
        
        # Check for invalid values
        intensity_finite = np.isfinite(intensity_data)
        dsm_finite = np.isfinite(dsm_data)
        result['validation']['no_invalid_values'] = intensity_finite.all() and dsm_finite.all()
        
        # Data range validation (reasonable ranges)
        intensity_in_range = np.all(intensity_data >= -150) and np.all(intensity_data <= 100)  # SAR dB range
        dsm_in_range = np.all(dsm_data >= -10000) and np.all(dsm_data <= 10000)  # Height range
        result['validation']['valid_data_ranges'] = intensity_in_range and dsm_in_range
        
        # Compute statistics
        valid_intensity = intensity_data[intensity_finite]
        if len(valid_intensity) > 0:
            result['data_stats']['intensity'] = {
                'min': np.min(valid_intensity),
                'max': np.max(valid_intensity),
                'mean': np.mean(valid_intensity),
                'std': np.std(valid_intensity),
                'count': len(valid_intensity),
                'shape': intensity_data.shape
            }
        
        valid_dsm = dsm_data[dsm_finite]
        if len(valid_dsm) > 0:
            result['data_stats']['dsm'] = {
                'min': np.min(valid_dsm),
                'max': np.max(valid_dsm),
                'mean': np.mean(valid_dsm),
                'std': np.std(valid_dsm),
                'count': len(valid_dsm),
                'shape': dsm_data.shape
            }
        
        # Store metadata
        result['metadata'] = {
            'intensity': intensity_meta,
            'dsm': dsm_meta
        }
        
        # Optionally return data
        if return_data:
            result['data'] = {
                'intensity': intensity_data,
                'dsm': dsm_data
            }
        
        logger.info(f"Processed pair: {pair_data['aoi']['full_name']}")
        logger.info(f"  Intensity shape: {intensity_data.shape}, dtype: {intensity_data.dtype}")
        logger.info(f"  DSM shape: {dsm_data.shape}, dtype: {dsm_data.dtype}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing pair {pair_data.get('base_name', 'unknown')}: {str(e)}")
        result['validation']['readable'] = False
        result['error'] = str(e)
        return result


class SARDataLoader:
    """
    Class to handle loading and management of SAR intensity and DSM data files.
    
    This class provides methods to:
    - Discover SAR and DSM file pairs
    - Load TIFF files with metadata
    - Validate file pairs
    - Extract AOI information from filenames
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the SAR data loader.
        
        Args:
            data_dir (str): Path to directory containing SAR and DSM TIFF files
        """
        self.data_dir = Path(data_dir)
        self.file_pairs = []
        self.intensity_files = []
        self.dsm_files = []
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self._discover_files()
    
    def _discover_files(self):
        """
        Discover all SAR intensity and DSM files in the data directory.
        Creates pairs of matching intensity and DSM files.
        """
        logger.info(f"Discovering files in {self.data_dir}")
        
        # Find all intensity files
        intensity_pattern = str(self.data_dir / "**/*Intensity_DB.tif")
        self.intensity_files = glob.glob(intensity_pattern, recursive=True)
        
        # Find all DSM files
        dsm_pattern = str(self.data_dir / "**/*DSM_RADAR.tif")
        self.dsm_files = glob.glob(dsm_pattern, recursive=True)
        
        logger.info(f"Found {len(self.intensity_files)} intensity files")
        logger.info(f"Found {len(self.dsm_files)} DSM files")
        
        # Create file pairs
        self._create_file_pairs()
    
    def _create_file_pairs(self):
        """
        Create pairs of intensity and DSM files based on filename matching.
        """
        self.file_pairs = []
        
        for intensity_file in self.intensity_files:
            # Extract base filename without the intensity suffix
            base_name = os.path.basename(intensity_file).replace('_Intensity_DB.tif', '')
            
            # Look for corresponding DSM file
            dsm_file = None
            for dsm_candidate in self.dsm_files:
                if base_name in os.path.basename(dsm_candidate):
                    dsm_file = dsm_candidate
                    break
            
            if dsm_file:
                self.file_pairs.append({
                    'intensity': intensity_file,
                    'dsm': dsm_file,
                    'base_name': base_name,
                    'aoi': self._extract_aoi_info(base_name)
                })
            else:
                logger.warning(f"No matching DSM file found for {intensity_file}")
        
        logger.info(f"Created {len(self.file_pairs)} intensity-DSM pairs")
    
    def _extract_aoi_info(self, filename: str) -> Dict[str, str]:
        """
        Extract Area of Interest (AOI) information from filename.
        
        Args:
            filename (str): Base filename without extension
            
        Returns:
            Dict[str, str]: Dictionary containing AOI information
        """
        # Parse ICEYE filename format: ICEYE_X#_SLC_SLH_#####_YYYYMMDDTHHMMSS
        parts = filename.split('_')
        
        if len(parts) >= 6:
            return {
                'satellite': parts[1],  # X2, X4, X7
                'orbit_id': parts[4],   # Orbit identifier
                'datetime': parts[5],   # Acquisition datetime
                'date': parts[5][:8] if len(parts[5]) >= 8 else parts[5],  # YYYYMMDD
                'time': parts[5][9:] if len(parts[5]) > 8 else '',  # HHMMSS
                'full_name': filename
            }
        else:
            return {
                'satellite': 'unknown',
                'orbit_id': 'unknown',
                'datetime': 'unknown',
                'date': 'unknown',
                'time': 'unknown',
                'full_name': filename
            }
    
    def load_file_pair(self, pair_index: int, return_metadata: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[Dict], Optional[Dict]]:
        """
        Load a specific intensity-DSM file pair.
        
        Args:
            pair_index (int): Index of the file pair to load
            return_metadata (bool): Whether to return metadata dictionaries
            
        Returns:
            Tuple containing:
            - intensity_data: SAR intensity data array
            - dsm_data: DSM height data array
            - intensity_meta: Intensity file metadata (if return_metadata=True)
            - dsm_meta: DSM file metadata (if return_metadata=True)
        """
        if pair_index >= len(self.file_pairs):
            raise IndexError(f"Pair index {pair_index} out of range")
        
        pair = self.file_pairs[pair_index]
        result = _load_and_analyze_pair(pair, return_data=True)
        
        if not result['validation']['readable']:
            raise RuntimeError(f"Failed to load pair {pair_index}: {result.get('error', 'Unknown error')}")
        
        intensity_data = result['data']['intensity']
        dsm_data = result['data']['dsm']
        
        if return_metadata:
            return intensity_data, dsm_data, result['metadata']['intensity'], result['metadata']['dsm']
        else:
            return intensity_data, dsm_data, None, None
    
    def get_file_pairs_info(self) -> List[Dict]:
        """
        Get information about all discovered file pairs.
        
        Returns:
            List[Dict]: List of dictionaries containing file pair information
        """
        return self.file_pairs.copy()


def _extract_spatial_aoi_info(file_path: str) -> Dict[str, Any]:
    """
    Extract spatial AOI information from a TIFF file.
    
    Args:
        file_path (str): Path to the TIFF file
        
    Returns:
        Dict containing spatial AOI information
    """
    try:
        with rasterio.open(file_path) as src:
            # Get bounds in original CRS
            bounds = src.bounds  # (left, bottom, right, top)
            original_crs = src.crs
            
            # Transform bounds to WGS84 for consistent comparison
            if original_crs and original_crs != CRS.from_epsg(4326):
                bounds_wgs84 = transform_bounds(
                    original_crs, CRS.from_epsg(4326), 
                    bounds.left, bounds.bottom, bounds.right, bounds.top
                )
            else:
                bounds_wgs84 = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            
            # Create polygon in WGS84
            polygon_wgs84 = box(*bounds_wgs84)
            
            # Calculate area in square kilometers
            # Use equal-area projection for accurate area calculation
            transformer = pyproj.Transformer.from_crs(
                'EPSG:4326', 'EPSG:3857', always_xy=True  # Web Mercator for rough area calculation
            )
            
            # Transform polygon to equal-area projection
            polygon_mercator = transform(transformer.transform, polygon_wgs84)
            area_m2 = polygon_mercator.area
            area_km2 = area_m2 / 1e6  # Convert to km²
            
            return {
                'bounds': (bounds.left, bounds.bottom, bounds.right, bounds.top),
                'bounds_wgs84': bounds_wgs84,
                'polygon_wgs84': polygon_wgs84,
                'crs': str(original_crs) if original_crs else None,
                'area_km2': area_km2
            }
            
    except Exception as e:
        logger.warning(f"Error extracting spatial AOI from {file_path}: {str(e)}")
        return {
            'bounds': None,
            'bounds_wgs84': None,
            'polygon_wgs84': None,
            'crs': None,
            'area_km2': None
        }


def get_data_statistics_with_spatial_aoi(data_dir: str, max_workers: Optional[int] = None) -> Dict:
    """
    Get comprehensive statistics about the SAR dataset including spatial AOI analysis.
    
    Args:
        data_dir (str): Path to directory containing SAR and DSM files
        max_workers (int, optional): Maximum number of worker processes
        
    Returns:
        Dict containing dataset statistics with spatial AOI information
    """
    loader = SARDataLoader(data_dir)
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(loader.file_pairs))
    
    # Initialize results structure
    stats = {
        'total_pairs': len(loader.file_pairs),
        'pair_details': [],
        'spatial_aois': [],  # New: spatial AOI information
        'summary': {
            'satellites': [],
            'orbit_ids': [],
            'dates': [],
            'acquisition_times': []
        },
        'validation_summary': {
            'total_pairs': len(loader.file_pairs),
            'valid_pairs': 0,
            'failed_pairs': [],
            'validation_details': []
        },
        'data_ranges': {
            'intensity': {'min': [], 'max': [], 'mean': [], 'std': [], 'shapes': []},
            'dsm': {'min': [], 'max': [], 'mean': [], 'std': [], 'shapes': []}
        }
    }
    
    logger.info(f"Processing {len(loader.file_pairs)} pairs with spatial AOI extraction...")
    
    # Process pairs in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_load_and_analyze_pair, pair, False): i 
            for i, pair in enumerate(loader.file_pairs)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            pair_index = future_to_index[future]
            try:
                result = future.result()
                pair_info = loader.file_pairs[pair_index]
                
                # Extract spatial AOI information
                intensity_spatial = _extract_spatial_aoi_info(pair_info['intensity'])
                
                # Create enhanced AOI info
                enhanced_aoi = pair_info['aoi'].copy()
                enhanced_aoi.update(intensity_spatial)
                
                # Store detailed pair information
                pair_detail = {
                    'index': pair_index,
                    'intensity_path': result['pair_info']['intensity'],
                    'dsm_path': result['pair_info']['dsm'],
                    'base_name': result['pair_info']['base_name'],
                    'aoi_info': enhanced_aoi,  # Now includes spatial information
                    'validation': result['validation'],
                    'data_stats': result['data_stats']
                }
                stats['pair_details'].append(pair_detail)
                
                # Store spatial AOI separately for analysis
                if enhanced_aoi['polygon_wgs84'] is not None:
                    spatial_aoi = {
                        'index': pair_index,
                        'base_name': result['pair_info']['base_name'],
                        'satellite': enhanced_aoi['satellite'],
                        'orbit_id': enhanced_aoi['orbit_id'],
                        'datetime': enhanced_aoi['datetime'],
                        'polygon_wgs84': enhanced_aoi['polygon_wgs84'],
                        'bounds_wgs84': enhanced_aoi['bounds_wgs84'],
                        'area_km2': enhanced_aoi['area_km2'],
                        'crs': enhanced_aoi['crs']
                    }
                    stats['spatial_aois'].append(spatial_aoi)
                
                # Update validation summary
                if result['validation']['readable'] and result['validation']['files_exist']:
                    stats['validation_summary']['valid_pairs'] += 1
                    
                    # Add to summary lists
                    aoi = result['pair_info']['aoi']
                    stats['summary']['satellites'].append(aoi['satellite'])
                    stats['summary']['orbit_ids'].append(aoi['orbit_id'])
                    stats['summary']['dates'].append(aoi['date'])
                    stats['summary']['acquisition_times'].append(aoi['datetime'])
                    
                    # ...existing code for data statistics...
                    if 'intensity' in result['data_stats'] and result['data_stats']['intensity']:
                        int_stats = result['data_stats']['intensity']
                        stats['data_ranges']['intensity']['min'].append(int_stats['min'])
                        stats['data_ranges']['intensity']['max'].append(int_stats['max'])
                        stats['data_ranges']['intensity']['mean'].append(int_stats['mean'])
                        stats['data_ranges']['intensity']['std'].append(int_stats['std'])
                        stats['data_ranges']['intensity']['shapes'].append(int_stats['shape'])
                    
                    if 'dsm' in result['data_stats'] and result['data_stats']['dsm']:
                        dsm_stats = result['data_stats']['dsm']
                        stats['data_ranges']['dsm']['min'].append(dsm_stats['min'])
                        stats['data_ranges']['dsm']['max'].append(dsm_stats['max'])
                        stats['data_ranges']['dsm']['mean'].append(dsm_stats['mean'])
                        stats['data_ranges']['dsm']['std'].append(dsm_stats['std'])
                        stats['data_ranges']['dsm']['shapes'].append(dsm_stats['shape'])
                else:
                    stats['validation_summary']['failed_pairs'].append({
                        'index': pair_index,
                        'base_name': result['pair_info']['base_name'],
                        'error': result.get('error', 'Validation failed')
                    })
                
                # Store validation details
                stats['validation_summary']['validation_details'].append(result['validation'])
                
            except Exception as e:
                logger.error(f"Error processing pair {pair_index}: {str(e)}")
                stats['validation_summary']['failed_pairs'].append({
                    'index': pair_index,
                    'base_name': f"pair_{pair_index}",
                    'error': str(e)
                })
    
    # Sort by index to maintain order
    stats['pair_details'].sort(key=lambda x: x['index'])
    stats['spatial_aois'].sort(key=lambda x: x['index'])
    
    # Analyze spatial intersections
    stats['spatial_intersection_analysis'] = analyze_spatial_intersections(stats['spatial_aois'])
    
    # ...existing code for aggregated stats and unique values...
    if stats['data_ranges']['intensity']['min']:
        stats['aggregated_stats'] = {
            'intensity': {
                'global_min': np.min(stats['data_ranges']['intensity']['min']),
                'global_max': np.max(stats['data_ranges']['intensity']['max']),
                'mean_of_means': np.mean(stats['data_ranges']['intensity']['mean']),
                'mean_of_stds': np.mean(stats['data_ranges']['intensity']['std']),
                'unique_shapes': list(set(map(tuple, stats['data_ranges']['intensity']['shapes'])))
            },
            'dsm': {
                'global_min': np.min(stats['data_ranges']['dsm']['min']),
                'global_max': np.max(stats['data_ranges']['dsm']['max']),
                'mean_of_means': np.mean(stats['data_ranges']['dsm']['mean']),
                'mean_of_stds': np.mean(stats['data_ranges']['dsm']['std']),
                'unique_shapes': list(set(map(tuple, stats['data_ranges']['dsm']['shapes'])))
            }
        }
    
    stats['unique_values'] = {
        'satellites': sorted(list(set(stats['summary']['satellites']))),
        'orbit_ids': sorted(list(set(stats['summary']['orbit_ids']))),
        'dates': sorted(list(set(stats['summary']['dates'])))
    }
    
    logger.info(f"Dataset analysis complete: {stats['validation_summary']['valid_pairs']}/{stats['total_pairs']} valid pairs")
    logger.info(f"Spatial AOI analysis: {len(stats['spatial_aois'])} AOIs with spatial information")
    
    return stats


def analyze_spatial_intersections(spatial_aois: List[Dict]) -> Dict[str, Any]:
    """
    Analyze spatial intersections between AOI polygons.
    
    Args:
        spatial_aois (List[Dict]): List of spatial AOI information
        
    Returns:
        Dict containing intersection analysis results
    """
    if len(spatial_aois) < 2:
        return {
            'total_aois': len(spatial_aois),
            'intersections': [],
            'intersection_summary': {
                'total_intersections': 0,
                'total_intersected_area_km2': 0,
                'max_intersection_percentage': 0,
                'intersections_by_satellite': {}
            }
        }
    
    intersections = []
    total_intersected_area = 0
    max_intersection_percentage = 0
    intersections_by_satellite = {}
    
    # Compare each pair of AOIs
    for i in range(len(spatial_aois)):
        for j in range(i + 1, len(spatial_aois)):
            aoi1 = spatial_aois[i]
            aoi2 = spatial_aois[j]
            
            # Calculate intersection
            try:
                intersection = aoi1['polygon_wgs84'].intersection(aoi2['polygon_wgs84'])
                
                if not intersection.is_empty and intersection.area > 0:
                    # Calculate intersection area in km²
                    transformer = pyproj.Transformer.from_crs(
                        'EPSG:4326', 'EPSG:3857', always_xy=True
                    )
                    intersection_mercator = transform(transformer.transform, intersection)
                    intersection_area_km2 = intersection_mercator.area / 1e6
                    
                    # Calculate intersection percentages
                    area1 = aoi1['area_km2']
                    area2 = aoi2['area_km2']
                    
                    percentage1 = (intersection_area_km2 / area1 * 100) if area1 > 0 else 0
                    percentage2 = (intersection_area_km2 / area2 * 100) if area2 > 0 else 0
                    
                    intersection_info = {
                        'aoi1_index': i,
                        'aoi2_index': j,
                        'aoi1_name': aoi1['base_name'],
                        'aoi2_name': aoi2['base_name'],
                        'aoi1_satellite': aoi1['satellite'],
                        'aoi2_satellite': aoi2['satellite'],
                        'aoi1_datetime': aoi1['datetime'],
                        'aoi2_datetime': aoi2['datetime'],
                        'intersection_area_km2': intersection_area_km2,
                        'aoi1_area_km2': area1,
                        'aoi2_area_km2': area2,
                        'intersection_percentage_aoi1': percentage1,
                        'intersection_percentage_aoi2': percentage2,
                        'max_intersection_percentage': max(percentage1, percentage2),
                        'intersection_polygon': intersection
                    }
                    
                    intersections.append(intersection_info)
                    total_intersected_area += intersection_area_km2
                    max_intersection_percentage = max(max_intersection_percentage, 
                                                    intersection_info['max_intersection_percentage'])
                    
                    # Count by satellite pairs
                    sat_pair = f"{aoi1['satellite']}-{aoi2['satellite']}"
                    if sat_pair not in intersections_by_satellite:
                        intersections_by_satellite[sat_pair] = 0
                    intersections_by_satellite[sat_pair] += 1
                    
            except Exception as e:
                logger.warning(f"Error calculating intersection between {aoi1['base_name']} and {aoi2['base_name']}: {str(e)}")
    
    # Sort intersections by area (largest first)
    intersections.sort(key=lambda x: x['intersection_area_km2'], reverse=True)
    
    return {
        'total_aois': len(spatial_aois),
        'intersections': intersections,
        'intersection_summary': {
            'total_intersections': len(intersections),
            'total_intersected_area_km2': total_intersected_area,
            'max_intersection_percentage': max_intersection_percentage,
            'intersections_by_satellite': intersections_by_satellite
        }
    }


# Update the main function to use the new spatial analysis
def get_data_statistics(data_dir: str, max_workers: Optional[int] = None) -> Dict:
    """
    Get comprehensive statistics about the SAR dataset including spatial AOI analysis.
    This is now an alias for get_data_statistics_with_spatial_aoi.
    """
    return get_data_statistics_with_spatial_aoi(data_dir, max_workers)


if __name__ == "__main__":
    # Example usage
    data_dir = "/home/younes.abid/git/physicsnemo/data/sar2height"
    
    # Create data loader
    loader = SARDataLoader(data_dir)
    
    # Print summary
    print(f"Found {len(loader.file_pairs)} file pairs")
    print("AOI Summary:", loader.get_file_pairs_info())
    
    # Load first pair as example
    if len(loader.file_pairs) > 0:
        intensity, dsm, int_meta, dsm_meta = loader.load_file_pair(0)
        print(f"First pair loaded: {intensity.shape}, {dsm.shape}")