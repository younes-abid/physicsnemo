import streamlit as st
import os
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Initialize session state variables
if "base_path" not in st.session_state:
    st.session_state.base_path = "/app/data/custom_data_2/ERA5_WRF_combined_concatenated"
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None
if "selected_data_file" not in st.session_state:
    st.session_state.selected_data_file = None
if "selected_date" not in st.session_state:
    st.session_state.selected_date = None
if 'selected_date_index' not in st.session_state:
    st.session_state.selected_date_index = None
if 'selected_variable' not in st.session_state:
    st.session_state.selected_variable = None
if 'file_loaded' not in st.session_state:
    st.session_state.file_loaded = False
if 'nc_files' not in st.session_state:
    st.session_state.nc_files = []

# Function to list .nc files in a directory
def list_nc_files(base_path):
    if not os.path.exists(base_path):
        st.sidebar.error("The provided path does not exist.")
        return []
    return [f for f in os.listdir(base_path) if f.endswith(".nc")]

# Function to explore a NetCDF file
def explore_nc_file(file_path):
    try:
        # Open the NetCDF file
        dataset = nc.Dataset(file_path, "r")
        st.subheader("📅 Available Dates")
        
        if "time" in dataset.variables:
            time_var = dataset.variables["time"]
            dates = pd.to_datetime(time_var[:], unit="s")  # Convert to datetime
            st.write(f"**Number of Available Dates:** {len(dates)}")
            st.write(f"**First Date:** {dates.min()}")
            st.write(f"**Last Date:** {dates.max()}")
            
            # Create date selection - update session state when changed
            selected_date = st.selectbox(
                "Select a date for analysis: Regression, Diffusion and Visualization", 
                dates,
                index=st.session_state.selected_date_index if st.session_state.selected_date_index is not None else 0,
                key="date_selector"
            )
            
            # Only update session state if a new date is selected
            if selected_date is not None:
                selected_date_index = dates.get_loc(selected_date)
                if st.session_state.selected_date_index != selected_date_index:
                    st.session_state.selected_date_index = selected_date_index
                    st.session_state.selected_variable = None  # Reset variable selection
                    st.session_state.selected_date = selected_date
                
                st.write(f"**Selected Date:** `{selected_date}` of index {selected_date_index} from 0 to {len(dates)-1}")
            else:
                st.session_state.selected_date_index = None
                st.session_state.selected_variable = None
                st.session_state.selected_date = None
                
        else:
            st.warning("No `time` variable found in the root dataset.")
            st.session_state.selected_date_index = None
            st.session_state.selected_variable = None
            st.session_state.selected_date = None
            
        dataset.close()

    except Exception as e:
        st.error(f"An error occurred while exploring the file: {e}")
        st.session_state.selected_date_index = None
        st.session_state.selected_variable = None
        st.session_state.selected_date = None

# Function to explore groups and display dates
def explore_groups_and_dates(file_path):
    try:
        groups = ["Input", "Output"]
        
        for group in groups:
            group_data = xr.open_dataset(file_path, group=group.lower())
            st.subheader(f"📂 Explore data in {group} Group")
            
            # Collapsible section for variable preview - ONLY SHOW IF DATE IS SELECTED
            if st.session_state.selected_date_index is not None:
                with st.expander("🔍 Preview Variable"):
                    # Variable selection - update session state when changed
                    variable_options = list(group_data.data_vars.keys())
                    
                    # Set default index if variable was previously selected
                    default_index = 0
                    if st.session_state.selected_variable in variable_options:
                        default_index = variable_options.index(st.session_state.selected_variable)
                    
                    selected_var = st.selectbox(
                        f"Select a variable to preview in {group}:", 
                        variable_options,
                        index=default_index,
                        key=f"var_selector_{group}"
                    )
                    
                    # Update session state when variable is selected
                    if selected_var:
                        st.session_state.selected_variable = selected_var
                        
                        # Only compute statistics if both date and variable are selected
                        if st.session_state.selected_date_index is not None and st.session_state.selected_variable:
                            st.write(f"**Variable:** `{selected_var}`")
                            
                            # Extract the selected variable for the selected date
                            var_data = group_data[selected_var].isel(sample=st.session_state.selected_date_index)
                            
                            # Display basic statistics
                            st.markdown("### 📊 Statistics")
                            st.write(f"**Mean:** {var_data.mean().item():.4f}")
                            st.write(f"**Min:** {var_data.min().item():.4f}")
                            st.write(f"**Max:** {var_data.max().item():.4f}")
                            st.write(f"**Std Dev:** {var_data.std().item():.4f}")
                            
                            # Display histogram
                            st.markdown("### 📈 Histogram")
                            flattened_data = var_data.values.flatten()
                            st.bar_chart(pd.Series(flattened_data).value_counts(bins=30).sort_index())
                            
                            # Display raster plot
                            st.markdown("### 🗺️ Raster Plot")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Use xarray's built-in plot function
                            plot = var_data.plot(
                                ax=ax,
                                cmap='viridis',
                                robust=True,  # Better color scaling
                                add_colorbar=True,
                                cbar_kwargs={'label': selected_var}
                            )
                            
                            ax.set_title(f'{selected_var} - Index {st.session_state.selected_date_index}')
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        st.info("Please select a variable to see statistics.")
            else:
                with st.expander("🔍 Preview Variable"):
                    st.info("Please select a date first to preview variables.")
            
            group_data.close()
            st.success(f"Successfully explored the {group} group.")
            
        st.success("All groups have been successfully explored.")
        
    except Exception as e:
        st.error(f"An error occurred while exploring groups and dates: {e}")

# Function to handle base path changes
def handle_base_path_change():
    new_base_path = st.session_state.base_path_input
    if new_base_path != st.session_state.base_path:
        st.session_state.base_path = new_base_path
        st.session_state.nc_files = list_nc_files(new_base_path)
        st.session_state.selected_file = None
        st.session_state.selected_data_file = None
        st.session_state.selected_date = None
        st.session_state.selected_date_index = None
        st.session_state.selected_variable = None
        st.session_state.file_loaded = False

# Function to handle file selection changes
def handle_file_selection():
    if st.session_state.file_selector != st.session_state.selected_file:
        st.session_state.selected_file = st.session_state.file_selector
        st.session_state.selected_data_file = os.path.join(st.session_state.base_path, st.session_state.selected_file)
        st.session_state.selected_date = None
        st.session_state.selected_date_index = None
        st.session_state.selected_variable = None
        st.session_state.file_loaded = True

# Main logic
st.title("🔎 Raw Data")

# Sidebar for navigation
st.sidebar.header("Navigation")

# Base path input with callback
base_path_input = st.sidebar.text_input(
    "Enter the base directory path:",
    value=st.session_state.base_path,
    key="base_path_input",
    on_change=handle_base_path_change
)

# Get NC files (cached in session state)
if not st.session_state.nc_files:
    st.session_state.nc_files = list_nc_files(st.session_state.base_path)

if not st.session_state.nc_files:
    st.sidebar.warning("No `.nc` files found in the selected directory.")
else:
    # File selector with callback
    selected_file = st.sidebar.selectbox(
        "Select a `.nc` file:",
        st.session_state.nc_files,
        key="file_selector",
        on_change=handle_file_selection
    )

    # Auto-load if file is selected or already loaded
    if st.session_state.selected_file or st.session_state.file_loaded:
        if st.session_state.selected_data_file:
            st.write(f"### Selected File: `{st.session_state.selected_data_file}`")
            
            explore_nc_file(st.session_state.selected_data_file)
            
            # Only show groups if a date is selected
            if st.session_state.selected_date_index is not None:
                explore_groups_and_dates(st.session_state.selected_data_file)
            else:
                st.info("Select a date to explore the data groups and variables.")
        else:
            st.info("Please select a file to begin.")

# Add a reset button
if st.sidebar.button("Reset All Selections"):
    st.session_state.base_path = "/app/data/custom_data_2/ERA5_WRF_combined_concatenated"
    st.session_state.selected_file = None
    st.session_state.selected_data_file = None
    st.session_state.selected_date = None
    st.session_state.selected_date_index = None
    st.session_state.selected_variable = None
    st.session_state.file_loaded = False
    st.session_state.nc_files = []
    st.rerun()

# Display current state for debugging (optional)
with st.sidebar.expander("Debug Info"):
    st.write("Current State:")
    st.write(f"Base Path: {st.session_state.base_path}")
    st.write(f"Selected File: {st.session_state.selected_file}")
    st.write(f"Selected Date: {st.session_state.selected_date}")
    st.write(f"Date Index: {st.session_state.selected_date_index}")
    #st.write(f"Variable: {st.session_state.selected_variable}")
    st.write(f"File Loaded: {st.session_state.file_loaded}")