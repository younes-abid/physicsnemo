"""
SAR2Height Statistics Computation Script

This script computes dataset statistics for SAR2Height data with the following structure:
- input group: Multi-channel SAR features 
- output group: DSM height data
- patch_metadata group: Spatial and quality metadata (optional for stats)

Adapted from the cordiff merge.py utility for SAR2Height specific data format.
"""

import os
import json
import numpy as np
import xarray as xr
from multiprocessing import Pool
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)

def process_sar2height_file_stats(fpath):
    """
    Process a single SAR2Height NetCDF file to compute statistics for input and output groups.
    
    Args:
        fpath (str): Path to the NetCDF file.
        
    Returns:
        dict: A dictionary containing the stats for the file.
    """
    print(f"Processing SAR2Height file: {os.path.basename(fpath)}")
    start_time = time.time()
    stats = {"input": {}, "output": {}}
    
    try:
        # Process input group (SAR features)
        input_start = time.time()
        ds_input = xr.open_dataset(fpath, group="input")
        for var in ds_input.data_vars:
            print(f"  Processing SAR variable '{var}' in input group...")
            data = ds_input[var].values.astype(np.float32)
            
            # Handle potential NaN/inf values in SAR data
            valid_mask = np.isfinite(data)
            if np.any(valid_mask):
                stats["input"][var] = {
                    "sum": float(np.sum(data[valid_mask])),
                    "sumsq": float(np.sum(data[valid_mask] ** 2)),
                    "count": int(np.sum(valid_mask))
                }
            else:
                print(f"    Warning: No valid values for variable '{var}'")
                stats["input"][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
                
        ds_input.close()
        print(f"  Processed input group in {time.time() - input_start:.2f} seconds.")

        # Process output group (DSM height)
        output_start = time.time()
        ds_output = xr.open_dataset(fpath, group="output")
        for var in ds_output.data_vars:
            print(f"  Processing height variable '{var}' in output group...")
            data = ds_output[var].values.astype(np.float32)
            
            # Handle potential NaN/inf values in height data
            valid_mask = np.isfinite(data)
            if np.any(valid_mask):
                stats["output"][var] = {
                    "sum": float(np.sum(data[valid_mask])),
                    "sumsq": float(np.sum(data[valid_mask] ** 2)),
                    "count": int(np.sum(valid_mask))
                }
            else:
                print(f"    Warning: No valid values for variable '{var}'")
                stats["output"][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
                
        ds_output.close()
        print(f"  Processed output group in {time.time() - output_start:.2f} seconds.")
        
    except Exception as e:
        print(f"  Error processing file '{fpath}': {e}")
        return {"input": {}, "output": {}}
    
    print(f"Finished processing {os.path.basename(fpath)} in {time.time() - start_time:.2f} seconds.")
    return stats


def compute_sar2height_dataset_stats(input_dir, json_out_path, num_workers=None):
    """
    Compute combined statistics for all SAR2Height NetCDF files in a directory using multiprocessing.
    
    Args:
        input_dir (str): Directory containing the NetCDF files
        json_out_path (str): Path to save the JSON statistics file
        num_workers (int, optional): Number of parallel workers. Defaults to CPU count.
    """
    start_time = time.time()
    
    if num_workers is None:
        num_workers = os.cpu_count()

    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' not found.")
        return

    # Find all NetCDF files
    file_list = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".nc"):
            fpath = os.path.join(input_dir, fname)
            file_list.append(fpath)
    
    if not file_list:
        print(f"❌ No NetCDF files found in '{input_dir}' to compute statistics.")
        return

    print(f"📊 Found {len(file_list)} SAR2Height NetCDF files in '{input_dir}'")
    print(f"🚀 Starting computation at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚡ Using {num_workers} parallel workers")

    # Process files in parallel
    print("🔄 Starting parallel processing of files...")
    parallel_start = time.time()
    
    with Pool(num_workers) as pool:
        file_stats_list = pool.map(process_sar2height_file_stats, file_list)
    
    print(f"✅ Parallel processing completed in {time.time() - parallel_start:.2f} seconds.")

    # Merge statistics from all files
    print("🔄 Merging statistics from all files...")
    merge_start = time.time()
    
    global_stats = {"input": {}, "output": {}}
    
    for file_stats in file_stats_list:
        if not file_stats:  # Skip empty results from failed files
            continue
            
        for category in ["input", "output"]:
            for var, v in file_stats[category].items():
                if var not in global_stats[category]:
                    global_stats[category][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
                
                global_stats[category][var]["sum"] += float(v["sum"])
                global_stats[category][var]["sumsq"] += float(v["sumsq"])
                global_stats[category][var]["count"] += int(v["count"])

    print(f"✅ Merging completed in {time.time() - merge_start:.2f} seconds.")

    # Compute mean and standard deviation
    print("🔄 Computing mean and standard deviation...")
    compute_start = time.time()
    
    final_stats = {"input": {}, "output": {}}
    
    for category in ["input", "output"]:
        for var, v in global_stats[category].items():
            if v["count"] == 0:
                print(f"⚠️  Warning: Variable '{var}' in category '{category}' has zero valid values. Using defaults.")
                final_stats[category][var] = {"mean": 0.0, "std": 1.0}
                continue
                
            mean = v["sum"] / v["count"]
            variance = max(0, v["sumsq"] / v["count"] - mean ** 2)
            std = np.sqrt(variance)
            
            final_stats[category][var] = {
                "mean": float(mean), 
                "std": float(std),
                "count": int(v["count"])
            }
            
            print(f"  {category}.{var}: mean={mean:.4f}, std={std:.4f}, count={v['count']}")

    print(f"✅ Mean and standard deviation computation completed in {time.time() - compute_start:.2f} seconds.")

    # Create output directory if needed
    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)

    # Save statistics to JSON
    print("💾 Saving statistics to JSON...")
    save_start = time.time()
    with open(json_out_path, "w") as f:
        json.dump(final_stats, f, indent=2)
    print(f"✅ Statistics saved to {json_out_path} in {time.time() - save_start:.2f} seconds.")

    print(f"🎉 Total computation time: {time.time() - start_time:.2f} seconds.")
    print(f"📄 Statistics file created: {json_out_path}")
    
    return final_stats


def compute_single_sar2height_stats(data_path, json_out_path):
    """
    Compute statistics for a single SAR2Height NetCDF file.
    
    Args:
        data_path (str): Path to the NetCDF file.
        json_out_path (str): Path to save the JSON file with computed statistics.
    """
    if not os.path.exists(data_path):
        print(f"❌ NetCDF file '{data_path}' not found.")
        return

    print(f"📊 Computing statistics for single file: {os.path.basename(data_path)}")
    
    stats = {"input": {}, "output": {}}

    try:
        # Process input group
        ds_input = xr.open_dataset(data_path, group="input")
        for var in ds_input.data_vars:
            data = ds_input[var].values.astype(np.float32)
            valid_mask = np.isfinite(data)
            if np.any(valid_mask):
                valid_data = data[valid_mask]
                stats["input"][var] = {
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "count": int(len(valid_data))
                }
            else:
                stats["input"][var] = {"mean": 0.0, "std": 1.0, "count": 0}
        ds_input.close()

        # Process output group
        ds_output = xr.open_dataset(data_path, group="output")
        for var in ds_output.data_vars:
            data = ds_output[var].values.astype(np.float32)
            valid_mask = np.isfinite(data)
            if np.any(valid_mask):
                valid_data = data[valid_mask]
                stats["output"][var] = {
                    "mean": float(np.mean(valid_data)),
                    "std": float(np.std(valid_data)),
                    "count": int(len(valid_data))
                }
            else:
                stats["output"][var] = {"mean": 0.0, "std": 1.0, "count": 0}
        ds_output.close()

    except Exception as e:
        print(f"❌ Error processing file '{data_path}': {e}")
        return

    # Create output directory if needed
    os.makedirs(os.path.dirname(json_out_path), exist_ok=True)
    
    # Save statistics to JSON
    with open(json_out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Saved statistics to {json_out_path}")
    
    return stats


if __name__ == "__main__":
    # Example usage
    print("SAR2Height Statistics Computation Script")
    print("Usage: python compute_sar2height_stats.py")
    print("Or import this module and call the functions directly.")