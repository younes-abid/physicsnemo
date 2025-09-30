import streamlit as st
import os
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from utils.states import init_state_variables, reset_state_variables, debug_state
from  utils.files import list_nc_files_raw_data
# Initialize session state variables
init_state_variables()

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
                index=st.session_state.raw_data["date_index"] if st.session_state.raw_data["date_index"] is not None else None,
                key="date_selector_raw_data"
            )
            
            if selected_date is not None:
                st.session_state.raw_data["date"] = selected_date
                st.session_state.raw_data["date_index"] = dates.get_loc(selected_date)
                st.write(f"**Selected Date Index:** {st.session_state.raw_data['date_index']}")
        else:
            st.warning("No `time` variable found in the root dataset.")
            st.session_state.raw_data["date_index"] = None
            st.session_state.raw_data["date"] = None

        dataset.close()

    except Exception as e:
        st.error(f"An error occurred while exploring the file: {e}")
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["date"] = None

# Function to explore groups and display dates
def explore_groups_and_dates(file_path):
    try:
        groups = ["Input", "Output"]

        for group in groups:
            group_data = xr.open_dataset(file_path, group=group.lower())
            st.subheader(f"📂 Explore data in {group} Group")

            # Collapsible section for variable preview - ONLY SHOW IF DATE IS SELECTED
            if st.session_state.raw_data["date_index"] is not None:
                with st.expander("🔍 Preview Variable"):
                    # Variable selection - update session state when changed
                    variable_options = list(group_data.data_vars.keys())

                    selected_var = st.selectbox(
                        f"Select a variable to preview in {group}:",
                        variable_options,
                        key=f"var_selector_{group}"
                    )

                    if selected_var:
                        st.write(f"**Variable:** `{selected_var}`")

                        # Extract the selected variable for the selected date
                        var_data = group_data[selected_var].isel(sample=st.session_state.raw_data["date_index"])

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
                        var_data.plot(
                            ax=ax,
                            cmap="viridis",
                            robust=True,
                            add_colorbar=True,
                            cbar_kwargs={"label": selected_var}
                        )
                        ax.set_title(f"{selected_var} - Index {st.session_state.raw_data['date_index']}")
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
def handle_base_path_raw_data_change():
    new_base_path_raw_data = st.session_state.base_path_raw_data_input
    if new_base_path_raw_data != st.session_state.raw_data["base_path"]:
        st.session_state.raw_data["base_path"] = new_base_path_raw_data
        st.session_state.raw_data["nc_files"] = list_nc_files_raw_data(new_base_path_raw_data)
        st.session_state.raw_data["file"] = None
        st.session_state.raw_data["file_path"] = None
        st.session_state.raw_data["date"] = None
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["file_loaded"] = False

# Function to handle file selection changes
def handle_file_selection():
    # The selectbox value is stored in the main session_state, not in raw_data
    if st.session_state.file_selector_raw_data != st.session_state.raw_data["file"]:
        st.session_state.raw_data["file"] = st.session_state.file_selector_raw_data
        st.session_state.raw_data["file_path"] = os.path.join(st.session_state.raw_data["base_path"], st.session_state.raw_data["file"])
        st.session_state.raw_data["date"] = None
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["file_loaded"] = True

# Main logic
st.title("🔎 Raw Data")

# Sidebar for navigation
st.sidebar.header("Navigation")

# Base path input with callback
st.sidebar.text_input(
    "Enter the base directory path:",
    value=st.session_state.raw_data["base_path"],
    key="base_path_raw_data_input",
    on_change=handle_base_path_raw_data_change
)

# Get NC files (cached in session state)
if not st.session_state.raw_data["nc_files"]:
    st.session_state.raw_data["nc_files"] = list_nc_files_raw_data(st.session_state.raw_data["base_path"])

if not st.session_state.raw_data["nc_files"]:
    st.sidebar.warning("No `.nc` files found in the selected directory.")
else:
    # File selector with callback - FIXED KEY
    st.sidebar.selectbox(
        "Select a `.nc` file:",
        st.session_state.raw_data["nc_files"],
        key="file_selector_raw_data", 
        on_change=handle_file_selection
    )

    # Auto-load if file is selected or already loaded
    if st.session_state.raw_data["file"] or st.session_state.raw_data["file_loaded"]:
        if st.session_state.raw_data["file_path"]:
            st.write(f"### Selected File: `{st.session_state.raw_data['file_path']}`")

            explore_nc_file(st.session_state.raw_data["file_path"])

            # Only show groups if a date is selected
            if st.session_state.raw_data["date_index"] is not None:
                explore_groups_and_dates(st.session_state.raw_data["file_path"])
            else:
                st.info("Select a date to explore the data groups and variables.")
        else:
            st.info("Please select a file to begin.")

# Add a reset button
if st.sidebar.button("Reset Raw Data Selections"):
    reset_state_variables(page="raw_data")
    st.sidebar.success("Selections have been reset.")
    st.rerun()

# Display current state for debugging
with st.sidebar.expander("Debug Info"):
    debug_state()