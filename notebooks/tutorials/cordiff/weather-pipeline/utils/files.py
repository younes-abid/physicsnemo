import os 
import streamlit as st
import json

# Function to list .nc files in a directory
def list_nc_files_raw_data(data_dir_path):
    """List all .nc files in the data directory."""
    if not os.path.exists(data_dir_path):
        return []
    try:
        return [f for f in os.listdir(data_dir_path) if f.endswith(".nc")]
    except Exception:
        return []

# Function to list .json files in a directory
def list_json_files(stats_dir_path):
    """List all .json files in the stats directory."""
    if not os.path.exists(stats_dir_path):
        return []
    try:
        return [f for f in os.listdir(stats_dir_path) if f.endswith(".json")]
    except Exception:
        return []

# Function to list .pt/.mdlus files in a directory
def list_pt_files(base_path):
    """List all .pt and .mdlus files in the models directory."""
    if not os.path.exists(base_path):
        return []
    try:
        return [f for f in os.listdir(base_path) if f.endswith((".pt", ".mdlus"))]
    except Exception:
        return []

# Function to load and validate JSON stats file
def load_stats_file(stats_file_path):
    """Load and validate a JSON stats file."""
    try:
        with open(stats_file_path, 'r') as f:
            stats_data = json.load(f)
        return stats_data, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON format: {str(e)}"
    except FileNotFoundError:
        return None, "File not found"
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

# Function to validate NetCDF file structure
def validate_nc_file(nc_file_path):
    """Validate that NetCDF file has required structure."""
    try:
        import netCDF4 as nc
        with nc.Dataset(nc_file_path, "r") as dataset:
            # Check for required dimensions/variables
            has_time = "time" in dataset.variables
            has_groups = len(dataset.groups) > 0
            return True, None
    except Exception as e:
        return False, f"Invalid NetCDF file: {str(e)}"