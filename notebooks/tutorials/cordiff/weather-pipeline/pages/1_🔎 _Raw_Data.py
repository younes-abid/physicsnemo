import streamlit as st
import os
import json
import netCDF4 as nc
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from utils.states import init_state_variables, reset_state_variables, debug_state, create_selectbox_with_default
from utils.files import list_nc_files_raw_data, list_json_files, load_stats_file, validate_nc_file

# Initialize session state variables
init_state_variables()

# Page configuration
st.set_page_config(
    page_title="Weather Pipeline - Raw Data",
    page_icon="🔎",
    layout="wide"
)

def display_stats_overview(stats_data):
    """Display a beautiful overview of the stats data."""
    if not stats_data:
        return
    
    st.markdown("### 📊 Statistics Overview")
    
    # Handle both possible structures
    input_vars = stats_data.get("input_variables", stats_data.get("input", {}))
    output_vars = stats_data.get("output_variables", stats_data.get("output", {}))
    invariant_vars = stats_data.get("invariant", {})
    
    # Create metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📥 Input Variables", len(input_vars))
    with col2:
        st.metric("📤 Output Variables", len(output_vars))
    with col3:
        st.metric("🔧 Invariant Variables", len(invariant_vars))
    with col4:
        total_vars = len(input_vars) + len(output_vars) + len(invariant_vars)
        st.metric("🔢 Total Variables", total_vars)

def display_stats_details(stats_data):
    """Display detailed statistics with beautiful formatting."""
    if not stats_data:
        return
    
    st.markdown("### 📋 Detailed Statistics")
    
    # Handle both possible structures
    input_vars = stats_data.get("input_variables", stats_data.get("input", {}))
    output_vars = stats_data.get("output_variables", stats_data.get("output", {}))
    invariant_vars = stats_data.get("invariant", {})
    
    if input_vars or output_vars or invariant_vars:
        tabs = []
        tab_names = []
        
        if input_vars:
            tabs.append("📥 Input Variables")
            tab_names.append("input")
        if output_vars:
            tabs.append("📤 Output Variables") 
            tab_names.append("output")
        if invariant_vars:
            tabs.append("🔧 Invariant Variables")
            tab_names.append("invariant")
        
        if len(tabs) == 1:
            # Single tab case
            if input_vars:
                display_variable_stats(input_vars, "Input")
            elif output_vars:
                display_variable_stats(output_vars, "Output")
            elif invariant_vars:
                display_variable_stats(invariant_vars, "Invariant")
        else:
            # Multiple tabs case
            tab_objects = st.tabs(tabs)
            
            for i, (tab_obj, tab_name) in enumerate(zip(tab_objects, tab_names)):
                with tab_obj:
                    if tab_name == "input" and input_vars:
                        display_variable_stats(input_vars, "Input")
                    elif tab_name == "output" and output_vars:
                        display_variable_stats(output_vars, "Output")
                    elif tab_name == "invariant" and invariant_vars:
                        display_variable_stats(invariant_vars, "Invariant")
                    else:
                        st.info(f"No {tab_name} variables found in stats file.")
    else:
        st.warning("No variable statistics found in the stats file.")

def display_variable_stats(variables_dict, var_type):
    """Display statistics for a set of variables."""
    if not variables_dict:
        return
    
    # Create a DataFrame for easy display
    stats_list = []
    for var_name, var_stats in variables_dict.items():
        stats_list.append({
            "Variable": var_name,
            "Mean": f"{var_stats.get('mean', 0):.4f}",
            "Std": f"{var_stats.get('std', 0):.4f}",
        })
    
    df = pd.DataFrame(stats_list)
    
    # Display the table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
    
def handle_data_dir_change():
    """Handle data directory path changes."""
    new_data_dir = st.session_state.data_dir_input
    if new_data_dir != st.session_state.raw_data["data_dir_path"]:
        st.session_state.raw_data["data_dir_path"] = new_data_dir
        st.session_state.raw_data["data_nc_files"] = list_nc_files_raw_data(new_data_dir)
        st.session_state.raw_data["data_file"] = None
        st.session_state.raw_data["data_file_path"] = None
        st.session_state.raw_data["date"] = None
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["data_file_loaded"] = False

def handle_stats_dir_change():
    """Handle stats directory path changes."""
    new_stats_dir = st.session_state.stats_dir_input
    if new_stats_dir != st.session_state.raw_data["stats_dir_path"]:
        st.session_state.raw_data["stats_dir_path"] = new_stats_dir
        st.session_state.raw_data["stats_json_files"] = list_json_files(new_stats_dir)
        st.session_state.raw_data["stats_file"] = None
        st.session_state.raw_data["stats_file_path"] = None
        st.session_state.raw_data["stats_data"] = None
        st.session_state.raw_data["stats_file_loaded"] = False

def handle_data_file_selection():
    """Handle data file selection changes."""
    # Use dynamic key that changes with reset counter
    reset_counter = st.session_state.get("reset_counter", 0)
    
    selected_file, selected_index, is_default = create_selectbox_with_default(
        "📄 Select Data File (.nc):",
        st.session_state.raw_data["data_nc_files"],
        st.session_state.raw_data,
        "data_file_selectbox_index",
        "Select data file...",
        key=f"data_file_selector_{reset_counter}",
        help_text="Choose the NetCDF file containing your weather data"
    )
    
    # Update session variables based on selection
    if is_default:
        # Reset to None when default option is selected
        st.session_state.raw_data["data_file"] = None
        st.session_state.raw_data["data_file_path"] = None
        st.session_state.raw_data["date"] = None
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["data_file_loaded"] = False
        # Reset date selectbox index to force showing "Select date..." message
        st.session_state.raw_data["date_selectbox_index"] = 0
    elif selected_file != st.session_state.raw_data["data_file"]:
        # Update when a new file is selected
        st.session_state.raw_data["data_file"] = selected_file
        st.session_state.raw_data["data_file_path"] = os.path.join(
            st.session_state.raw_data["data_dir_path"], 
            selected_file
        )
        st.session_state.raw_data["date"] = None
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["data_file_loaded"] = True
        # Reset date selectbox index to force showing "Select date..." message
        st.session_state.raw_data["date_selectbox_index"] = 0

def handle_stats_file_selection():
    """Handle stats file selection changes."""
    # Use dynamic key that changes with reset counter
    reset_counter = st.session_state.get("reset_counter", 0)
    
    selected_file, selected_index, is_default = create_selectbox_with_default(
        "📄 Select Statistics File (.json):",
        st.session_state.raw_data["stats_json_files"],
        st.session_state.raw_data,
        "stats_file_selectbox_index",
        "Select statistics file...",
        key=f"stats_file_selector_{reset_counter}",
        help_text="Choose the JSON file containing normalization statistics"
    )
    
    # Update session variables based on selection
    if is_default:
        # Reset to None when default option is selected
        st.session_state.raw_data["stats_file"] = None
        st.session_state.raw_data["stats_file_path"] = None
        st.session_state.raw_data["stats_data"] = None
        st.session_state.raw_data["stats_file_loaded"] = False
    elif selected_file != st.session_state.raw_data["stats_file"]:
        # Update when a new file is selected
        st.session_state.raw_data["stats_file"] = selected_file
        st.session_state.raw_data["stats_file_path"] = os.path.join(
            st.session_state.raw_data["stats_dir_path"], 
            selected_file
        )
        
        # Load the stats file immediately
        if st.session_state.raw_data["stats_file_path"]:
            stats_data, error = load_stats_file(st.session_state.raw_data["stats_file_path"])
            if error:
                st.error(f"Error loading stats file: {error}")
                st.session_state.raw_data["stats_data"] = None
                st.session_state.raw_data["stats_file_loaded"] = False
            else:
                st.session_state.raw_data["stats_data"] = stats_data
                st.session_state.raw_data["stats_file_loaded"] = True
                
        # Force a rerun to update the UI
        st.rerun()

def explore_nc_file(file_path):
    """Explore NetCDF file and handle date selection."""
    try:
        with nc.Dataset(file_path, "r") as dataset:
            st.markdown("### 📅 Available Dates")

            if "time" in dataset.variables:
                time_var = dataset.variables["time"]
                dates = pd.to_datetime(time_var[:], unit="s")
                
                # Display date information in a nice format
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📊 Total Dates", len(dates))
                with col2:
                    st.metric("🚀 First Date", dates.min().strftime("%Y-%m-%d"))
                with col3:
                    st.metric("🏁 Last Date", dates.max().strftime("%Y-%m-%d"))

                # Date selection with improved UI
                st.markdown("#### Select Date for Analysis")
                
                # Use the new selectbox function
                date_strings = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]
                reset_counter = st.session_state.get("reset_counter", 0)
                
                selected_date_str, selected_date_index, is_default = create_selectbox_with_default(
                    "Choose a date for regression, diffusion and visualization:",
                    date_strings,
                    st.session_state.raw_data,
                    "date_selectbox_index",
                    "Select date...",
                    key=f"date_selector_raw_data_{reset_counter}"
                )
                
                # Update session variables based on selection
                if is_default:
                    st.session_state.raw_data["date"] = None
                    st.session_state.raw_data["date_index"] = None
                    # Show warning when no date is selected
                    st.warning("⚠️ select a date from available dates.")
                else:
                    selected_date = dates[selected_date_index]
                    st.session_state.raw_data["date"] = selected_date
                    st.session_state.raw_data["date_index"] = selected_date_index
                    
                    # Display selected date info nicely
                    st.success(f"✅ **Selected Date:** {selected_date.strftime('%Y-%m-%d %H:%M:%S')} (Index: {st.session_state.raw_data['date_index']})")
            else:
                st.warning("⚠️ No `time` variable found in the dataset.")
                st.session_state.raw_data["date_index"] = None
                st.session_state.raw_data["date"] = None

    except Exception as e:
        st.error(f"❌ Error exploring file: {e}")
        st.session_state.raw_data["date_index"] = None
        st.session_state.raw_data["date"] = None

def explore_data_groups(file_path):
    """Explore data groups with enhanced visualization."""
    try:
        groups = ["Input", "Output"]

        for group in groups:
            with st.expander(f"📂 Explore {group} Data Group", expanded=False):
                if st.session_state.raw_data["date_index"] is not None:
                    group_data = xr.open_dataset(file_path, group=group.lower())
                    
                    # Display group overview
                    st.markdown(f"#### {group} Group Overview")
                    variable_options = list(group_data.data_vars.keys())
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🔢 Variables Count", len(variable_options))
                    with col2:
                        if variable_options:
                            sample_var = group_data[variable_options[0]]
                            shape = sample_var.shape
                            st.metric("📏 Data Shape", f"{shape}")

                    # Variable selection and preview using new function
                    reset_counter = st.session_state.get("reset_counter", 0)
                    selected_var, selected_var_index, is_var_default = create_selectbox_with_default(
                        "Select variable to preview:",
                        variable_options,
                        st.session_state.raw_data,
                        f"{group.lower()}_var_selectbox_index",
                        "Select variable...",
                        key=f"var_selector_{group}_{reset_counter}"
                    )
                    
                    # Update session variable
                    if is_var_default:
                        st.session_state.raw_data[f"{group.lower()}_var_index"] = 0
                    else:
                        st.session_state.raw_data[f"{group.lower()}_var_index"] = selected_var_index

                    if not is_var_default and selected_var:
                        # Extract data for selected date
                        var_data = group_data[selected_var].isel(sample=st.session_state.raw_data["date_index"])

                        # Create tabs for different views
                        tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📈 Distribution", "🗺️ Spatial View"])
                        
                        with tab1:
                            # Statistics in a nice format
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("📊 Mean", f"{var_data.mean().item():.4f}")
                            with col2:
                                st.metric("📈 Std Dev", f"{var_data.std().item():.4f}")
                            with col3:
                                st.metric("⬇️ Min", f"{var_data.min().item():.4f}")
                            with col4:
                                st.metric("⬆️ Max", f"{var_data.max().item():.4f}")
                        
                        with tab2:
                            # Enhanced histogram
                            flattened_data = var_data.values.flatten()
                            
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # Histogram
                            ax1.hist(flattened_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                            ax1.set_title(f'{selected_var} - Distribution')
                            ax1.set_xlabel('Value')
                            ax1.set_ylabel('Frequency')
                            ax1.grid(True, alpha=0.3)
                            
                            # Box plot
                            ax2.boxplot(flattened_data, vert=True)
                            ax2.set_title(f'{selected_var} - Box Plot')
                            ax2.set_ylabel('Value')
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with tab3:
                            # Enhanced spatial plot
                            fig, ax = plt.subplots(figsize=(10, 8))
                            im = var_data.plot(
                                ax=ax,
                                cmap="viridis",
                                robust=True,
                                add_colorbar=True,
                                cbar_kwargs={"label": f"{selected_var}", "shrink": 0.8}
                            )
                            ax.set_title(f"{selected_var} - Spatial Distribution (Index: {st.session_state.raw_data['date_index']})")
                            ax.set_aspect('equal')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                    group_data.close()
                else:
                    st.info("🔍 Please select a date first to explore data groups.")

    except Exception as e:
        st.error(f"❌ Error exploring data groups: {e}")

# Main UI
st.title("🔎 Raw Data Configuration")
st.markdown("Configure your data files, statistics, and select analysis parameters.")

st.markdown("---")

# Main content in two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 Data File Configuration")
    
    # Data directory input
    st.text_input(
        "📂 Data Directory Path:",
        value=st.session_state.raw_data["data_dir_path"],
        key="data_dir_input",
        on_change=handle_data_dir_change,
        help="Path to directory containing .nc data files"
    )
    
    # Update file list if needed
    if not st.session_state.raw_data["data_nc_files"]:
        st.session_state.raw_data["data_nc_files"] = list_nc_files_raw_data(
            st.session_state.raw_data["data_dir_path"]
        )
    
    # Data file selection
    handle_data_file_selection()
    
    # Show file info if selected
    if st.session_state.raw_data["data_file_path"]:
        st.success(f"✅ **Selected:** `{st.session_state.raw_data['data_file']}`")
        
        # File validation
        is_valid, error = validate_nc_file(st.session_state.raw_data["data_file_path"])
        if is_valid:
            st.success("🔍 File validation: ✅ Valid NetCDF structure")
        else:
            st.error(f"🔍 File validation: ❌ {error}")
    else:
        st.warning("⚠️ select .nc file from selected directory.")

with col2:
    st.markdown("### 📊 Statistics File Configuration")
    
    # Stats directory input
    st.text_input(
        "📂 Statistics Directory Path:",
        value=st.session_state.raw_data["stats_dir_path"],
        key="stats_dir_input",
        on_change=handle_stats_dir_change,
        help="Path to directory containing .json statistics files"
    )
    
    # Update file list if needed
    if not st.session_state.raw_data["stats_json_files"]:
        st.session_state.raw_data["stats_json_files"] = list_json_files(
            st.session_state.raw_data["stats_dir_path"]
        )
    
    # Stats file selection
    handle_stats_file_selection()
    
    # Show file info if selected
    if st.session_state.raw_data["stats_file_path"]:
        if st.session_state.raw_data["stats_file_loaded"]:
            st.success(f"✅ **Selected:** `{st.session_state.raw_data['stats_file']}`")
            st.success("🔍 File validation: ✅ Valid JSON format")
        else:
            st.error("❌ Failed to load statistics file")
    else:
        st.warning("⚠️ select .json file from selected directory.")

st.markdown("---")

# Data exploration section (only show if data file is selected)
if st.session_state.raw_data["data_file_path"]:
    st.markdown("## 🔍 Data Exploration")
    
    # Date selection
    explore_nc_file(st.session_state.raw_data["data_file_path"])
    
    # Data groups exploration (only show if date is selected)
    if st.session_state.raw_data["date_index"] is not None:
        st.markdown("---")
        explore_data_groups(st.session_state.raw_data["data_file_path"])

# Statistics display section (only show if stats file is loaded)
if st.session_state.raw_data["stats_data"]:
    st.markdown("---")
    st.markdown("## 📊 Statistics Analysis")
    display_stats_overview(st.session_state.raw_data["stats_data"])
    st.markdown("---")
    display_stats_details(st.session_state.raw_data["stats_data"])

# Sidebar controls
st.sidebar.markdown("### 🔧 Controls")

if st.sidebar.button("🔄 Reset All Selections", type="secondary"):
    reset_state_variables(page="raw_data")
    st.sidebar.success("✅ All selections reset!")
    st.rerun()

if st.sidebar.button("🔄 Refresh File Lists", type="secondary"):
    st.session_state.raw_data["data_nc_files"] = list_nc_files_raw_data(
        st.session_state.raw_data["data_dir_path"]
    )
    st.session_state.raw_data["stats_json_files"] = list_json_files(
        st.session_state.raw_data["stats_dir_path"]
    )
    st.sidebar.success("✅ File lists refreshed!")
    st.rerun()

# Debug section
with st.sidebar.expander("🔍 Debug Info"):
    debug_state()