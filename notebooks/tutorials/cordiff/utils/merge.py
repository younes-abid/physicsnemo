import os
import glob
import json
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import warnings
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

def compute_dataset_stats(input_dir, json_out_path, invariant_ds=None):
    if not os.path.exists(input_dir):
        print(f"Output directory '{input_dir}' not found.")
        return

    file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".nc")])
    if not file_list:
        print(f"No NetCDF files found in '{input_dir}' to compute statistics.")
        return

    stats = {"input": {}, "output": {}, "invariant": {}}

    for fname in file_list:
        fpath = os.path.join(input_dir, fname)
        try:
            ds_input = xr.open_dataset(fpath, group="input")
            for var in ds_input.data_vars:
                data = ds_input[var].values.astype(np.float32)
                if var not in stats["input"]:
                    stats["input"][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
                stats["input"][var]["sum"] += np.nansum(data)
                stats["input"][var]["sumsq"] += np.nansum(data ** 2)
                stats["input"][var]["count"] += np.count_nonzero(~np.isnan(data))
            ds_input.close()

            ds_output = xr.open_dataset(fpath, group="output")
            for var in ds_output.data_vars:
                data = ds_output[var].values.astype(np.float32)
                if var not in stats["output"]:
                    stats["output"][var] = {"sum": 0.0, "sumsq": 0.0, "count": 0}
                stats["output"][var]["sum"] += np.nansum(data)
                stats["output"][var]["sumsq"] += np.nansum(data ** 2)
                stats["output"][var]["count"] += np.count_nonzero(~np.isnan(data))
            ds_output.close()
        except Exception as e:
            print(f"Skipped file {fname} due to error: {e}")
            continue

    for category in ["input", "output"]:
        for var, v in stats[category].items():
            mean = v["sum"] / v["count"]
            std = np.sqrt(v["sumsq"] / v["count"] - mean ** 2)
            stats[category][var] = {"mean": float(mean), "std": float(std)}

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

    with open(json_out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved full variable stats (input, output, invariant) to {json_out_path}")

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

# Run
# if __name__ == "__main__":
#     merge_files(ERA5_DIR, WRF_DIR, OUT_DIR)
#     compute_dataset_stats(OUT_DIR, JSON_OUT, invariant_ds=invariant_ds)

