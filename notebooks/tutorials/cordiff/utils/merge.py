import os
import glob
import json
import numpy as np
import xarray as xr
from netCDF4 import Dataset
from multiprocessing import Pool, Manager
import warnings
import time
warnings.simplefilter(action='ignore', category=FutureWarning)

# CONFIG
# ERA5_DIR = "./ERA5_combined/"
# WRF_DIR = "./WRF/"
# OUT_DIR = "./ERA5_wrf_combined/"
# JSON_OUT = os.path.join(OUT_DIR, "uae_stats.json")
# INVARIANT_FILE = os.path.join(WRF_DIR, "invariants_wrf.nc")

# os.makedirs(OUT_DIR, exist_ok=True)

# # Load invariant dataset
# invariant_ds = xr.open_dataset(INVARIANT_FILE)[["XLAT", "XLONG", "elev_mean", "lsm_mean", "land_use"]]

def compute_rain_rate(ds):
    rainc = ds["RAINC"]
    rainnc = ds["RAINNC"]
    rain_total = rainc + rainnc
    rain_rate = rain_total.diff(dim="Time", label="upper")
    rain_rate = rain_rate.pad(Time=(1, 0), constant_values=0)
    return rain_rate

def write_grouped_netcdf(path, ds_input, ds_output, invariant_ds=None):
    for var in ds_input.data_vars:
        if ds_input[var].ndim == 3:
            sample, y_lr, x_lr = ds_input[var].shape
            break

    y_hr = ds_output.sizes["south_north"]
    x_hr = ds_output.sizes["west_east"]

    with Dataset(path, "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("sample", sample)
        ncfile.createDimension("y_lr", y_lr)
        ncfile.createDimension("x_lr", x_lr)
        ncfile.createDimension("y_hr", y_hr)
        ncfile.createDimension("x_hr", x_hr)

        # Group: input
        grp_in = ncfile.createGroup("input")
        for var in ds_input.data_vars:
            data = ds_input[var].values
            grp_in.createVariable(var, "f4", ("sample", "y_lr", "x_lr"))[:] = data

        # Group: output
        grp_out = ncfile.createGroup("output")
        for var in ds_output.data_vars:
            data = ds_output[var].values
            grp_out.createVariable(var, "f4", ("sample", "y_hr", "x_hr"))[:] = data

        # Group: invariant
        if invariant_ds is not None:
            grp_inv = ncfile.createGroup("invariant")
            for var in invariant_ds.data_vars:
                arr = invariant_ds[var]
                if "Time" in arr.dims:
                    arr = arr.isel(Time=0)
                data = arr.values.squeeze()
                if data.ndim == 2:
                    y_grid, x_grid = data.shape
                elif data.ndim == 3 and data.shape[0] == 1:
                    y_grid, x_grid = data.shape[1:]
                    data = data[0]
                else:
                    raise ValueError(f"Invariant variable {var} has unsupported shape {data.shape}")
                if "y_grid" not in ncfile.dimensions:
                    ncfile.createDimension("y_grid", y_grid)
                    ncfile.createDimension("x_grid", x_grid)
                grp_inv.createVariable(var, "f4", ("y_grid", "x_grid"))[:] = data

def compute_single_file_stats(data_path, json_out_path, invariant_ds=None):
    """
    Compute statistics for a single NetCDF file with input, output, and optional invariant groups.

    Args:
        data_path (str): Path to the NetCDF file.
        json_out_path (str): Path to save the JSON file with computed statistics.
        invariant_ds (xarray.Dataset, optional): Invariant dataset to include in the statistics. Defaults to None.
    """
    if not os.path.exists(data_path):
        print(f"NetCDF file '{data_path}' not found.")
        return

    stats = {"input": {}, "output": {}, "invariant": {}}

    try:
        # Process input group
        ds_input = xr.open_dataset(data_path, group="input")
        for var in ds_input.data_vars:
            data = ds_input[var].values.astype(np.float32)
            stats["input"][var] = {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data))
            }
        ds_input.close()

        # Process output group
        ds_output = xr.open_dataset(data_path, group="output")
        for var in ds_output.data_vars:
            data = ds_output[var].values.astype(np.float32)
            stats["output"][var] = {
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data))
            }
        ds_output.close()

        # Process invariant group if provided
        if invariant_ds is not None:
            for var in invariant_ds.data_vars:
                arr = invariant_ds[var]
                if "Time" in arr.dims:
                    arr = arr.isel(Time=0)
                data = arr.values.astype(np.float32).squeeze()
                stats["invariant"][var] = {
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data))
                }

    except Exception as e:
        print(f"Error processing file '{data_path}': {e}")
        return

    # Save statistics to JSON
    with open(json_out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {json_out_path}")

####
def process_file_stats(fpath):
    """
    Process a single file to compute statistics for input and output groups.

    Args:
        fpath (str): Path to the NetCDF file.

    Returns:
        dict: A dictionary containing the stats for the file.
    """
    print(f"Processing file: {fpath}\n")
    start_time = time.time()
    stats = {"input": {}, "output": {}}
    try:
        # Process input group
        input_start = time.time()
        ds_input = xr.open_dataset(fpath, group="input")
        for var in ds_input.data_vars:
            print(f"Processing variable '{var}' in input group...")
            data = ds_input[var].values.astype(np.float32)
            stats["input"][var] = {
                "sum": np.nansum(data),
                "sumsq": np.nansum(data ** 2),
                "count": np.count_nonzero(~np.isnan(data))
            }
        ds_input.close()
        print(f"Processed input group in {time.time() - input_start:.2f} seconds.")

        # Process output group
        output_start = time.time()
        ds_output = xr.open_dataset(fpath, group="output")
        for var in ds_output.data_vars:
            print(f"Processing variable '{var}' in output group...")
            data = ds_output[var].values.astype(np.float32)
            stats["output"][var] = {
                "sum": np.nansum(data),
                "sumsq": np.nansum(data ** 2),
                "count": np.count_nonzero(~np.isnan(data))
            }
        ds_output.close()
        print(f"Processed output group in {time.time() - output_start:.2f} seconds.")
    except Exception as e:
        print(f"Error processing file '{fpath}': {e}")
    print(f"Finished processing file {fpath} in {time.time() - start_time:.2f} seconds.")
    print(stats,"\n")
    return stats


def merge_stats(global_stats, file_stats):
    """
    Merge the stats from a single file into the global stats.

    Args:
        global_stats (dict): The global stats dictionary.
        file_stats (dict): The stats dictionary for a single file.
    """
    for category in ["input", "output"]:
        for var, v in file_stats[category].items():
            if var not in global_stats[category]:
                global_stats[category][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
            global_stats[category][var]["sum"] += v["sum"]
            global_stats[category][var]["sumsq"] += v["sumsq"]
            global_stats[category][var]["count"] += v["count"]


def compute_dataset_stats(input_dir, json_out_path, invariant_ds=None, num_workers=12):
    """
    Compute combined statistics for all NetCDF files in a directory using multiprocessing.

    Args:
        input_dir (str): Path to the directory containing NetCDF files.
        json_out_path (str): Path to save the JSON file with computed statistics.
        invariant_ds (xarray.Dataset, optional): Invariant dataset to include in the statistics. Defaults to None.
        num_workers (int): Number of parallel workers to use.
    """
    start_time = time.time()

    if not os.path.exists(input_dir):
        print(f"Output directory '{input_dir}' not found.")
        return

    file_list = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nc")])
    if not file_list:
        print(f"No NetCDF files found in '{input_dir}' to compute statistics.")
        return

    print(f"Found {len(file_list)} NetCDF files in '{input_dir}' to compute statistics.")
    print(f"Starting computation at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Use multiprocessing to process files in parallel
    with Manager() as manager:
        global_stats = manager.dict({"input": {}, "output": {}, "invariant": {}})
        with Pool(num_workers) as pool:
            print("Starting parallel processing of files...")
            parallel_start = time.time()
            file_stats_list = pool.map(process_file_stats, file_list)
            print(f"Parallel processing completed in {time.time() - parallel_start:.2f} seconds.")

        # Merge all file stats into global stats
        print("Merging statistics from all files...")
        merge_start = time.time()
        for file_stats in file_stats_list:
            merge_stats(global_stats, file_stats)
        print(f"Merging completed in {time.time() - merge_start:.2f} seconds.")

        # Compute mean and standard deviation for input and output groups
        print("Computing mean and standard deviation...")
        compute_start = time.time()
        for category in ["input", "output"]:
            for var, v in global_stats[category].items():
                mean = v["sum"] / v["count"]
                std = np.sqrt(v["sumsq"] / v["count"] - mean ** 2)
                global_stats[category][var] = {"mean": float(mean), "std": float(std)}
        print(f"Mean and standard deviation computation completed in {time.time() - compute_start:.2f} seconds.")

        # Process invariant dataset if provided
        if invariant_ds is not None:
            print("Processing invariant dataset...")
            invariant_start = time.time()
            for var in invariant_ds.data_vars:
                arr = invariant_ds[var]
                if "Time" in arr.dims:
                    arr = arr.isel(Time=0)
                data = arr.values.astype(np.float32).squeeze()
                global_stats["invariant"][var] = {
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data))
                }
            print(f"Invariant dataset processing completed in {time.time() - invariant_start:.2f} seconds.")

        # Save statistics to JSON
        print("Saving statistics to JSON...")
        save_start = time.time()
        with open(json_out_path, "w") as f:
            json.dump(dict(global_stats), f, indent=2)
        print(f"Statistics saved to {json_out_path} in {time.time() - save_start:.2f} seconds.")

    print(f"Total computation time: {time.time() - start_time:.2f} seconds.")

    
# def compute_dataset_stats(input_dir, json_out_path, invariant_ds=None):
#     if not os.path.exists(input_dir):
#         print(f"Output directory '{input_dir}' not found.")
#         return

#     file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".nc")])
#     if not file_list:
#         print(f"No NetCDF files found in '{input_dir}' to compute statistics.")
#         return
#     else:
#         print(f"Found {len(file_list)} NetCDF files in '{input_dir}' to compute statistics.")

#     stats = {"input": {}, "output": {}, "invariant": {}}

#     for fname in file_list:
#         fpath = os.path.join(input_dir, fname)
#         print(f"Processing file: {fname}")
#         try:
#             ds_input = xr.open_dataset(fpath, group="input")
#             for var in ds_input.data_vars:
#                 data = ds_input[var].values.astype(np.float32)
#                 if var not in stats["input"]:
#                     stats["input"][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
#                 stats["input"][var]["sum"] += np.nansum(data)
#                 stats["input"][var]["sumsq"] += np.nansum(data ** 2)
#                 stats["input"][var]["count"] += np.count_nonzero(~np.isnan(data))
#             ds_input.close()

#             ds_output = xr.open_dataset(fpath, group="output")
#             for var in ds_output.data_vars:
#                 data = ds_output[var].values.astype(np.float32)
#                 if var not in stats["output"]:
#                     stats["output"][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
#                 stats["output"][var]["sum"] += np.nansum(data)
#                 stats["output"][var]["sumsq"] += np.nansum(data ** 2)
#                 stats["output"][var]["count"] += np.count_nonzero(~np.isnan(data))
#             ds_output.close()
#         except Exception as e:
#             print(f"Skipped file {fname} due to error: {e}")
#             continue

#     for category in ["input", "output"]:
#         for var, v in stats[category].items():
#             mean = v["sum"] / v["count"]
#             std = np.sqrt(v["sumsq"] / v["count"] - mean ** 2)
#             stats[category][var] = {"mean": float(mean), "std": float(std)}

#     if invariant_ds is not None:
#         for var in invariant_ds.data_vars:
#             arr = invariant_ds[var]
#             if "Time" in arr.dims:
#                 arr = arr.isel(Time=0)
#             data = arr.values.astype(np.float32).squeeze()
#             stats["invariant"][var] = {
#                 "mean": float(np.nanmean(data)),
#                 "std": float(np.nanstd(data))
#             }

#     with open(json_out_path, "w") as f:
#         json.dump(stats, f, indent=2)
#     print(f"Saved full variable stats (input, output, invariant) to {json_out_path}")

def merge_files(era_path, wrf_path, out_path, invariant_ds=None):
    print(f"Merging files from {era_path} and {wrf_path} into {out_path}")
    era_files = sorted(glob.glob(os.path.join(era_path, "*.nc")))
    count = 0

    for era_file in era_files:
        date_str = os.path.basename(era_file).split("_")[-1].replace(".nc", "")
        wrf_file = os.path.join(wrf_path, f"wrfout_final_{date_str}.nc")

        if not os.path.exists(wrf_file):
            print(f"Skipping {date_str} — WRF file not found: {wrf_file}")
            continue

        print(f"Merging {os.path.basename(era_file)} + {os.path.basename(wrf_file)}")

        ds_era = xr.open_dataset(era_file)
        ds_wrf = xr.open_dataset(wrf_file)

        ds_wrf["rain_rate"] = compute_rain_rate(ds_wrf)
        
        print(f"Processing {date_str}...")
        # TODO print all the variables in ds_era and ds_wrf
        print(f"ERA5 variables: {list(ds_era.data_vars)}")
        print(f"WRF variables: {list(ds_wrf.data_vars)}")
        ds_input = ds_era[[
            "t_850", "t_500", "z_850", "z_500",
            "u_850", "u_500", "v_850", "v_500",
            "u10", "v10", "t2m", "d2m", "skt", "sst",
            "sp", "tcwv", "tp"
        ]]

        ds_output = ds_wrf[[
            "T2", "U10", "V10", "rain_rate", "SST", "TSK", "Q2", "PSFC"
        ]]

        out_file = os.path.join(out_path, f"inp_out_combined_{date_str}.nc")
        write_grouped_netcdf(out_file, ds_input, ds_output, invariant_ds)
        print(f"Saved: {out_file}")
        count += 1

    if count == 0:
        print("No files merged.")
    else:
        print(f"Successfully merged {count} file(s).")

def merge_file(date_str, era_path, wrf_path, out_path, input_vars, output_vars, invariant_ds=None, remove_era=False):
    """
    Merges a single ERA5 and WRF file for a given date.

    Args:
        date_str (str): Date string to process (e.g., "2023-01-01").
        era_path (str): Path to the interpolated ERA5 files.
        wrf_path (str): Path to the WRF files.
        out_path (str): Path to save the combined output files.
        input_vars (list): List of input variable names to extract from ERA5 files.
        output_vars (list): List of output variable names to extract from WRF files.
        invariant_ds (xarray.Dataset, optional): Invariant dataset to include in the output. Defaults to None.
        remove_era (bool): If True, removes the ERA5 file after successful merging to free disk space.
    """
    era_file = os.path.join(era_path, f"interpolated_era5_{date_str}.nc")
    wrf_file = os.path.join(wrf_path, f"wrfout_final_{date_str}.nc")
    out_file = os.path.join(out_path, f"inp_out_combined_{date_str}.nc")

    if not os.path.exists(wrf_file):
        print(f"Skipping {date_str} — WRF file not found: {wrf_file}")
        return

    try:
        # Open ERA5 and WRF datasets
        ds_era = xr.open_dataset(era_file)
        ds_wrf = xr.open_dataset(wrf_file)

        # Compute rain rate and add it to the WRF dataset
        ds_wrf["rain_rate"] = compute_rain_rate(ds_wrf)

        print(f"Processing {date_str}...")

        # Dynamically select input and output variables
        ds_input = ds_era[input_vars]
        ds_output = ds_wrf[output_vars]

        # Write the combined dataset to a NetCDF file
        write_grouped_netcdf(out_file, ds_input, ds_output, invariant_ds)
        print(f"Saved: {out_file}")

        # Remove the ERA5 file if the merge was successful and remove_era is True
        if remove_era:
            os.remove(era_file)
            print(f"Removed ERA5 file: {era_file}")

    except Exception as e:
        print(f"Failed to merge {date_str}: {e}")
        
# Run
# if __name__ == "__main__":
#     merge_files(ERA5_DIR, WRF_DIR, OUT_DIR)
#     compute_dataset_stats(OUT_DIR, JSON_OUT, invariant_ds=invariant_ds)

