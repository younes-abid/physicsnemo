"""
Main Processing Pipeline for SAR-to-Height Data Processing

This module orchestrates the complete SAR-to-height data preparation pipeline
for the cordiff training system. It integrates all individual processing modules
to convert raw SAR and DSM TIFF files into cordiff-compatible NetCDF patches.

Author: AI Assistant
Date: 2026-01-06
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import argparse

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from data_loader import SARDataLoader
from coregistration import coregister_file_pair
from feature_extraction import extract_sar_features
from patch_generator import generate_patches_from_data
from aoi_management import AOIManager
from netcdf_writer import save_patches_to_netcdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SARToHeightProcessor:
    """
    Main processor class for SAR-to-height data preparation pipeline.
    
    This class coordinates all processing steps:
    1. Load SAR intensity and DSM TIFF files
    2. Coregister data to common spatial grid
    3. Extract multi-channel features from SAR intensity
    4. Generate quality-filtered patches
    5. Check for spatial overlap with existing AOIs
    6. Save valid patches to cordiff-compatible NetCDF files
    """
    
    def __init__(self, 
                 data_dir: str,
                 patches_output_dir: str,
                 skipped_patches_dir: str,
                 patch_size: int = 432,
                 overlap_threshold: float = 0.1,
                 min_patch_quality: float = 0.7):
        """
        Initialize the SAR-to-height processor.
        
        Args:
            data_dir (str): Directory containing SAR and DSM TIFF files
            patches_output_dir (str): Directory to save processed NetCDF patches
            skipped_patches_dir (str): Directory to save skipped patches
            patch_size (int): Size of square patches (default: 432 for cordiff)
            overlap_threshold (float): AOI overlap threshold for filtering
            min_patch_quality (float): Minimum patch quality threshold
        """
        self.data_dir = Path(data_dir)
        self.patches_output_dir = Path(patches_output_dir)
        self.skipped_patches_dir = Path(skipped_patches_dir)
        
        # Processing parameters
        self.patch_size = patch_size
        self.overlap_threshold = overlap_threshold
        self.min_patch_quality = min_patch_quality
        
        # Create output directories
        self.patches_output_dir.mkdir(parents=True, exist_ok=True)
        self.skipped_patches_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = SARDataLoader(str(self.data_dir))
        self.aoi_manager = AOIManager(
            str(self.patches_output_dir), 
            str(self.skipped_patches_dir)
        )
        
        logger.info(f"SAR-to-Height Processor initialized")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Output directory: {self.patches_output_dir}")
        logger.info(f"  Patch size: {self.patch_size}")
        logger.info(f"  Found {len(self.data_loader.file_pairs)} file pairs to process")
    
    def process_single_file_pair(self, pair_index: int) -> Dict:
        """
        Process a single SAR intensity and DSM file pair.
        
        Args:
            pair_index (int): Index of the file pair to process
            
        Returns:
            Dict: Processing results and statistics
        """
        if pair_index >= len(self.data_loader.file_pairs):
            raise IndexError(f"File pair index {pair_index} out of range")
        
        pair_info = self.data_loader.file_pairs[pair_index]
        base_name = pair_info['base_name']
        aoi_info = pair_info['aoi']
        
        logger.info(f"Processing file pair {pair_index}: {base_name}")
        
        result = {
            'pair_index': pair_index,
            'base_name': base_name,
            'aoi_info': aoi_info,
            'success': False,
            'error': None,
            'n_patches_generated': 0,
            'n_patches_valid': 0,
            'n_patches_skipped': 0,
            'output_files': {}
        }
        
        try:
            # Step 1: Load data
            logger.info(f"Step 1: Loading data for {base_name}")
            intensity_data, dsm_data, intensity_meta, dsm_meta = self.data_loader.load_file_pair(pair_index)
            
            # Step 2: Coregister data
            logger.info(f"Step 2: Coregistering data for {base_name}")
            intensity_coreg, dsm_coreg, common_meta = coregister_file_pair(
                intensity_data, dsm_data, intensity_meta, dsm_meta
            )
            
            # Step 3: Extract features
            logger.info(f"Step 3: Extracting features for {base_name}")
            features, feature_names = extract_sar_features(
                intensity_coreg, 
                metadata=common_meta
            )
            
            # Step 4: Generate patches
            logger.info(f"Step 4: Generating patches for {base_name}")
            all_patches = generate_patches_from_data(
                features, dsm_coreg, common_meta,
                patch_size=self.patch_size,
                min_quality=self.min_patch_quality
            )
            
            result['n_patches_generated'] = len(all_patches)
            
            if not all_patches:
                logger.warning(f"No valid patches generated for {base_name}")
                return result
            
            # Step 5: Check for spatial overlaps
            logger.info(f"Step 5: Checking spatial overlaps for {base_name}")
            valid_patches, overlapping_patches = self.aoi_manager.filter_non_overlapping_patches(
                all_patches, self.overlap_threshold
            )
            
            result['n_patches_valid'] = len(valid_patches)
            result['n_patches_skipped'] = len(overlapping_patches)
            
            # Step 6: Save patches to NetCDF files
            logger.info(f"Step 6: Saving patches for {base_name}")
            
            created_files = {}
            
            if valid_patches:
                # Create output filename
                output_filename = f"sar2height_{base_name}_{len(valid_patches)}patches.nc"
                output_path = self.patches_output_dir / output_filename
                
                # Save valid patches
                created_files = save_patches_to_netcdf(
                    patches=valid_patches,
                    output_path=str(output_path),
                    feature_names=feature_names,
                    base_name=base_name,
                    aoi_info=aoi_info,
                    skipped_patches=overlapping_patches if overlapping_patches else None,
                    skipped_output_path=str(self.skipped_patches_dir / f"skipped_{output_filename}") if overlapping_patches else None
                )
                
                # Register new AOI
                if 'main' in created_files:
                    self.aoi_manager.register_new_aoi(
                        valid_patches, 
                        created_files['main'],
                        base_name,
                        satellite=aoi_info.get('satellite', 'unknown'),
                        orbit_id=aoi_info.get('orbit_id', 'unknown'),
                        datetime=aoi_info.get('datetime', 'unknown')
                    )
            
            result['output_files'] = created_files
            result['success'] = True
            
            logger.info(f"Successfully processed {base_name}: "
                       f"{result['n_patches_valid']} valid patches, "
                       f"{result['n_patches_skipped']} skipped patches")
        
        except Exception as e:
            logger.error(f"Error processing {base_name}: {str(e)}")
            result['error'] = str(e)
            result['success'] = False
        
        return result
    
    def process_all_file_pairs(self, 
                              start_index: Optional[int] = None,
                              end_index: Optional[int] = None) -> List[Dict]:
        """
        Process all available file pairs.
        
        Args:
            start_index (int, optional): Starting index for processing
            end_index (int, optional): Ending index for processing
            
        Returns:
            List[Dict]: List of processing results for each file pair
        """
        n_pairs = len(self.data_loader.file_pairs)
        start_idx = start_index or 0
        end_idx = end_index or n_pairs
        
        logger.info(f"Processing file pairs {start_idx} to {end_idx-1} ({end_idx-start_idx} pairs)")
        
        results = []
        successful_pairs = 0
        failed_pairs = 0
        
        for i in range(start_idx, min(end_idx, n_pairs)):
            logger.info(f"Processing pair {i+1}/{end_idx-start_idx}: {self.data_loader.file_pairs[i]['base_name']}")
            
            try:
                result = self.process_single_file_pair(i)
                results.append(result)
                
                if result['success']:
                    successful_pairs += 1
                else:
                    failed_pairs += 1
                    
            except Exception as e:
                logger.error(f"Critical error processing pair {i}: {str(e)}")
                failed_pairs += 1
                results.append({
                    'pair_index': i,
                    'base_name': self.data_loader.file_pairs[i]['base_name'],
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"Processing complete: {successful_pairs} successful, {failed_pairs} failed")
        
        return results
    
    def get_processing_summary(self, results: List[Dict]) -> Dict:
        """
        Generate summary statistics from processing results.
        
        Args:
            results (List[Dict]): List of processing results
            
        Returns:
            Dict: Summary statistics
        """
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        total_patches_generated = sum(r.get('n_patches_generated', 0) for r in successful_results)
        total_patches_valid = sum(r.get('n_patches_valid', 0) for r in successful_results)
        total_patches_skipped = sum(r.get('n_patches_skipped', 0) for r in successful_results)
        
        # Get AOI summary
        aoi_summary = self.aoi_manager.get_aoi_summary()
        
        summary = {
            'total_file_pairs': len(results),
            'successful_pairs': len(successful_results),
            'failed_pairs': len(failed_results),
            'total_patches_generated': total_patches_generated,
            'total_patches_valid': total_patches_valid,
            'total_patches_skipped': total_patches_skipped,
            'patch_acceptance_rate': (total_patches_valid / total_patches_generated * 100) if total_patches_generated > 0 else 0,
            'aoi_summary': aoi_summary,
            'failed_files': [r['base_name'] for r in failed_results]
        }
        
        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='SAR-to-Height Data Processing Pipeline')
    parser.add_argument('data_dir', help='Directory containing SAR and DSM TIFF files')
    parser.add_argument('output_dir', help='Directory to save processed NetCDF patches')
    parser.add_argument('--patch-size', type=int, default=432, help='Patch size (default: 432)')
    parser.add_argument('--overlap-threshold', type=float, default=0.1, help='AOI overlap threshold (default: 0.1)')
    parser.add_argument('--min-quality', type=float, default=0.7, help='Minimum patch quality (default: 0.7)')
    parser.add_argument('--start-index', type=int, help='Starting file pair index')
    parser.add_argument('--end-index', type=int, help='Ending file pair index')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create output directories
    patches_dir = Path(args.output_dir) / "patches"
    skipped_dir = Path(args.output_dir) / "patches_skipped"
    
    # Initialize processor
    processor = SARToHeightProcessor(
        data_dir=args.data_dir,
        patches_output_dir=str(patches_dir),
        skipped_patches_dir=str(skipped_dir),
        patch_size=args.patch_size,
        overlap_threshold=args.overlap_threshold,
        min_patch_quality=args.min_quality
    )
    
    # Process files
    results = processor.process_all_file_pairs(
        start_index=args.start_index,
        end_index=args.end_index
    )
    
    # Generate and print summary
    summary = processor.get_processing_summary(results)
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total file pairs processed: {summary['total_file_pairs']}")
    print(f"Successful: {summary['successful_pairs']}")
    print(f"Failed: {summary['failed_pairs']}")
    print(f"Total patches generated: {summary['total_patches_generated']}")
    print(f"Valid patches saved: {summary['total_patches_valid']}")
    print(f"Patches skipped (overlap): {summary['total_patches_skipped']}")
    print(f"Patch acceptance rate: {summary['patch_acceptance_rate']:.1f}%")
    
    aoi_summary = summary['aoi_summary']
    print(f"\nAOI Database:")
    print(f"  Total AOIs: {aoi_summary['total_aois']}")
    print(f"  Total patches in database: {aoi_summary['total_patches']}")
    print(f"  Satellites: {aoi_summary['satellites']}")
    
    if summary['failed_files']:
        print(f"\nFailed files:")
        for failed_file in summary['failed_files']:
            print(f"  - {failed_file}")
    
    print("="*60)
    
    # Save AOI database
    processor.aoi_manager.save_aoi_database()
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Exit with error code if any processing failed
    failed_count = len([r for r in results if not r.get('success', False)])
    if failed_count > 0:
        sys.exit(1)