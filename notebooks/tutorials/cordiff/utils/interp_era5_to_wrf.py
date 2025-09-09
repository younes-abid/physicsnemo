import os
import xarray as xr
import numpy as np
from glob import glob
from scipy.interpolate import griddata
from scipy.interpolate import RectBivariateSpline
# ----------------------------
# User settings
# ----------------------------
# ERA5_DIR = "./ERA5/"  # Folder with both pressure and single level files
# WRF_DIR = "./WRF/"
# OUT_DIR = "./ERA5_combined/"
# LEVELS = [500, 850]  # hPa
# VARIABLES = ["t", "z", "u", "v"]  # From pressure level file
# SINGLE_VARS = ["u10", "v10", "d2m", "t2m", "sst", "skt", "sp", "tcwv", "tp", "q"]       # From single level file
# os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
def get_wrf_grid(wrf_path):
    ds = xr.open_dataset(wrf_path)
    lats = ds["XLAT"].values[0, :, :]
    lons = ds["XLONG"].values[0, :, :]
    ds.close()
    return lats, lons

# ----------------------------

def extract_and_interp_var(ds, varname, wrf_lats, wrf_lons, level=None):
    if "valid_time" in ds.dims:
        ds = ds.rename({"valid_time": "time"})

    interp_list = []
    times = ds["time"].values

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    if lat.ndim != 1 or lon.ndim != 1:
        raise ValueError(f"{varname}: Expected 1D lat/lon, got shapes {lat.shape}, {lon.shape}")

    for i, t in enumerate(times):
        try:
            if level is not None:
                da = ds[varname].sel(pressure_level=level).isel(time=i)
                name = f"{varname}_{level}"
            else:
                da = ds[varname].isel(time=i)
                name = varname

            data = np.squeeze(da.values)
            if lat[0] > lat[-1]:
                lat = lat[::-1]
                data = data[::-1, :]
            if lon[0] > lon[-1]:
                lon = lon[::-1]
                data = data[:, ::-1]

            if data.shape != (len(lat), len(lon)):
                raise ValueError(f"{varname}: data shape {data.shape} does not match lat/lon grid {len(lat)}x{len(lon)}")

            # Bicubic interpolation
            spline = RectBivariateSpline(lat, lon, data, kx=3, ky=3)
            interp = spline(wrf_lats[:, 0], wrf_lons[0, :])

            interp_list.append(interp)
        except Exception as e:
            print(f"  Failed to interpolate {varname} at time index {i}: {e}")
            continue

    if not interp_list:
        raise ValueError(f"No valid interpolations found for {varname}.")

    stacked = np.stack(interp_list, axis=0)

    return xr.DataArray(
        stacked,
        dims=("time", "y", "x"),
        coords={
            "time": times[:len(interp_list)],
            "y": np.arange(wrf_lats.shape[0]),
            "x": np.arange(wrf_lats.shape[1])
        },
        name=name
    )




# ----------------------------
def process_date(date_str, ERA5_DIR, WRF_DIR, OUT_DIR, VARIABLES, LEVELS, SINGLE_VARS):
    print(f"\n Processing date: {date_str}")

    pres_file = os.path.join(ERA5_DIR, f"era5_pressure_levels_cropped_{date_str}.nc")
    single_file = os.path.join(ERA5_DIR, f"era5_single_levels_cropped_{date_str}.nc")
    wrf_file = os.path.join(WRF_DIR, f"wrfout_final_{date_str}.nc")
    out_file = os.path.join(OUT_DIR, f"interpolated_era5_{date_str}.nc")

    if not (os.path.exists(pres_file) and os.path.exists(single_file) and os.path.exists(wrf_file)):
        print(" Skipping.")
        return
    if os.path.exists(out_file):
        print(" Already done.")
        return

    wrf_lats, wrf_lons = get_wrf_grid(wrf_file)

    ds_pres = xr.open_dataset(pres_file)
    ds_single = xr.open_dataset(single_file)

    vars_interp = []

    for var in VARIABLES:
        for lvl in LEVELS:
            print(f" Interpolating {var}_{lvl}")
            da_interp = extract_and_interp_var(ds_pres, var, wrf_lats, wrf_lons, level=lvl)
            vars_interp.append(da_interp)

    for var in SINGLE_VARS:
        print(f" Interpolating {var}")
        try:
            da_interp = extract_and_interp_var(ds_single, var, wrf_lats, wrf_lons, level=None)
            vars_interp.append(da_interp)
        except Exception as e:
            print(f" Failed: {e}")

    xr.merge(vars_interp).to_netcdf(out_file)
    print(f"Saved: {out_file}")

# ----------------------------
# Batch process
# ----------------------------
# era5_files = sorted(glob(os.path.join(ERA5_DIR, "era5_pressure_levels_cropped_*.nc")))
# date_list = [os.path.basename(f).replace("era5_pressure_levels_cropped_", "").replace(".nc", "") for f in era5_files]

# for date in date_list:
#     process_date(date)

