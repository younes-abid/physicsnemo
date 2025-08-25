import os
import xarray as xr
from netCDF4 import Dataset
import numpy as np
from datetime import timedelta, datetime


# Function to pad dimensions to the nearest multiple
def pad_to_multiple_of(ds, dim_y, dim_x, multiple_of, pad_value):
    new_y = ((ds.dims[dim_y] + (multiple_of - 1)) // multiple_of) * multiple_of
    new_x = ((ds.dims[dim_x] + (multiple_of - 1)) // multiple_of) * multiple_of
    pad_y = new_y - ds.dims[dim_y]
    pad_x = new_x - ds.dims[dim_x]
    return ds.pad({dim_y: (0, pad_y), dim_x: (0, pad_x)}, constant_values=pad_value)

# Function to process a single file
def process_file(file, multiple_of, pad_value):
    print(f"Processing file: {file}")
    input_ds = xr.open_dataset(file, group="input")
    output_ds = xr.open_dataset(file, group="output")
    
    # Pad input and output datasets
    input_ds = pad_to_multiple_of(input_ds, "y_lr", "x_lr", multiple_of, pad_value)
    output_ds = pad_to_multiple_of(output_ds, "y_hr", "x_hr", multiple_of, pad_value)
    
    # Check for invariant group
    try:
        invariant_ds = xr.open_dataset(file, group="invariant")
    except Exception:
        print(f"No invariant group found in {file}, skipping invariant.")
        invariant_ds = None
    
    # Extract the date from the filename
    date_str = os.path.basename(file).split("_")[-1].replace(".nc", "")
    base_date = np.datetime64(date_str)
    num_samples = input_ds.dims["sample"]
    
    # Generate time values for the file
    time_step = timedelta(hours=24 / num_samples)  # Time step based on num_samples
    time_values = [base_date + np.timedelta64(int(i * time_step.total_seconds()), 's') for i in range(num_samples)]
    
    return input_ds, output_ds, invariant_ds, time_values

# Function to write the combined dataset to a NetCDF file
def write_combined_dataset(output_file, combined_input, combined_output, combined_invariant, time_array):
    print(f"Writing combined dataset to {output_file}...")
    with Dataset(output_file, "w", format="NETCDF4") as ncfile:
        # Create dimensions
        ncfile.createDimension("sample", size=combined_input.dims["sample"])
        ncfile.createDimension("coord", size=2)  # Assuming coord has 2 dimensions (latitude, longitude)

        # Add time variable
        time_var = ncfile.createVariable("time", "f8", ("sample",))
        time_var[:] = time_array.astype("datetime64[s]").astype(float)  # Convert to seconds since epoch
        time_var.units = "seconds since 1970-01-01 00:00:00"
        time_var.calendar = "standard"

        # Add coord variable (dummy values for now)
        coord_var = ncfile.createVariable("coord", "f4", ("sample", "coord"))
        coord_var[:, :] = np.full((combined_input.dims["sample"], 2), 1)  # Replace with actual coordinates if available

        # Create groups
        input_group = ncfile.createGroup("input")
        output_group = ncfile.createGroup("output")
        invariant_group = ncfile.createGroup("invariant") if combined_invariant is not None else None

        # Add input variables
        for var in combined_input.data_vars:
            data = combined_input[var].values
            dims = combined_input[var].dims
            for dim in dims:
                if dim not in input_group.dimensions:
                    input_group.createDimension(dim, size=combined_input.dims[dim])
            input_group.createVariable(var, "f4", dims)[:] = data

        # Add output variables
        for var in combined_output.data_vars:
            data = combined_output[var].values
            dims = combined_output[var].dims
            for dim in dims:
                if dim not in output_group.dimensions:
                    output_group.createDimension(dim, size=combined_output.dims[dim])
            output_group.createVariable(var, "f4", dims)[:] = data

        # Add invariant variables if they exist
        if combined_invariant is not None:
            for var in combined_invariant.data_vars:
                data = combined_invariant[var].values
                dims = combined_invariant[var].dims
                for dim in dims:
                    if dim not in invariant_group.dimensions:
                        invariant_group.createDimension(dim, size=combined_invariant.dims[dim])
                invariant_group.createVariable(var, "f4", dims)[:] = data

    print(f"Combined dataset saved to {output_file}")

# Main function to process and combine all files
def concat(IN_DIR, OUTPUT_FILE, MULTIPLE_OF, PAD_VALUE, end=-1 ):
    # Get a list of all NetCDF files in the output directory except the output file
    file_list = sorted([os.path.join(IN_DIR, f) for f in os.listdir(IN_DIR) if (f.endswith(".nc")
                                                                                 and f != os.path.basename(OUTPUT_FILE))])
    print(f"Found {len(file_list)} files to concatenate in {IN_DIR}.")

    # Initialize lists to store datasets for each group
    input_datasets = []
    output_datasets = []
    invariant_datasets = []
    time_values = []

    # Process each file
    for file in file_list[:end]:  # Debug with only 5 files
        try:
            input_ds, output_ds, invariant_ds, file_time_values = process_file(file, MULTIPLE_OF, PAD_VALUE)
            input_datasets.append(input_ds)
            output_datasets.append(output_ds)
            if invariant_ds is not None:
                invariant_datasets.append(invariant_ds)
            time_values.extend(file_time_values)
        except Exception as e:
            print(f"Skipped file {file} due to error: {e}")

    # Concatenate datasets for each group
    print("Concatenating input datasets...")
    combined_input = xr.concat(input_datasets, dim="sample")
    print("Concatenating output datasets...")
    combined_output = xr.concat(output_datasets, dim="sample")

    # Combine invariant datasets if they exist
    if invariant_datasets:
        print("Combining invariant datasets...")
        combined_invariant = xr.concat(invariant_datasets, dim="sample")
    else:
        print("No invariant datasets found.")
        combined_invariant = None

    # Convert time values to a NumPy array
    time_array = np.array(time_values)

    # Write the combined dataset
    write_combined_dataset(OUTPUT_FILE, combined_input, combined_output, combined_invariant, time_array)