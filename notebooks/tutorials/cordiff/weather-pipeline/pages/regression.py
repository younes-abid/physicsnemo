import streamlit as st
import os

# Initialize session state variables
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "selected_data_file" not in st.session_state:
    st.session_state.selected_data_file = None
if "selected_date_index" not in st.session_state:
    st.session_state.selected_date_index = None

# Function to list .pt files in a directory
def list_pt_files(base_path):
    if not os.path.exists(base_path):
        st.sidebar.error("The provided path does not exist.")
        return []
    return [f for f in os.listdir(base_path) if f.endswith(".pt")]

# Sidebar for navigation
st.sidebar.header("Navigation")
base_path = st.sidebar.text_input("Enter the base directory path:", "/app/checkpoints_regression/")

# List and select .pt files
pt_files = list_pt_files(base_path)
if not pt_files:
    st.sidebar.warning("No `.pt` files found in the selected directory.")
else:
    selected_model = st.sidebar.selectbox("Select a regression model:", pt_files)
    if st.sidebar.button("Load Model"):
        st.session_state.selected_model = os.path.join(base_path, selected_model)

# Reset selections button
if st.sidebar.button("Reset Selections"):
    st.session_state.selected_model = None
    st.session_state.selected_data_file = None
    st.session_state.selected_date_index = None
    st.experimental_rerun()

# Main content
st.title("📈 Regression")

# Display selected model
if st.session_state.selected_model:
    st.write(f"### Selected Model: `{st.session_state.selected_model}`")
else:
    st.warning("No model selected. Please select a regression model from the sidebar.")

# Display selected data file
if st.session_state.selected_data_file:
    st.write(f"### Selected Data File: `{st.session_state.selected_data_file}`")
else:
    st.warning("No data file selected. Please go to the **Raw Data** page and select a file.")

# Display selected date
if st.session_state.selected_date_index is not None:
    st.write(f"### Selected Date Index: `{st.session_state.selected_date_index}`")
else:
    st.warning("No date selected. Please go to the **Raw Data** page and select a date.")