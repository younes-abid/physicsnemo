import os 
import streamlit as st
# Function to list .nc files in a directory
def list_nc_files_raw_data(base_path_raw_data):
    if not os.path.exists(base_path_raw_data):
        st.sidebar.error("The provided path does not exist.")
        return []
    return [f for f in os.listdir(base_path_raw_data) if f.endswith(".nc")]

# Function to list .pt files in a directory
def list_pt_files(base_path):
    if not os.path.exists(base_path):
        st.sidebar.error("The provided path does not exist.")
        return []
    return [f for f in os.listdir(base_path) if f.endswith(".pt")]