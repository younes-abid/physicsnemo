import streamlit as st
import os
from utils.states import init_state_variables, reset_state_variables, debug_state
from utils.files import list_pt_files
# Initialize session state variables
init_state_variables()



# Sidebar for navigation
st.sidebar.header("Navigation")

# Base path input for regression models
base_path_regression = st.sidebar.text_input(
    "Enter the base directory path:",
    value="/app/checkpoints_regression/"
)

# List and select .pt files
pt_files = list_pt_files(base_path_regression)
if not pt_files:
    st.sidebar.warning("No `.pt` files found in the selected directory.")
else:
    selected_model = st.sidebar.selectbox(
        "Select a regression model:",
        pt_files
    )
    if selected_model:
        st.session_state.regression["selected_model"] = os.path.join(base_path_regression, selected_model)

# Add a reset button
if st.sidebar.button("Reset Regression Selections"):
    reset_state_variables(page="regression")
    st.sidebar.success("Selections have been reset.")
    st.rerun()

# Main content
st.title("📈 Regression")

# Display selected model
if st.session_state.regression["selected_model"]:
    st.write(f"### Selected Model: `{st.session_state.regression['selected_model']}`")
else:
    st.warning("No model selected. Please select a regression model from the sidebar.")

# Display selected data file
if st.session_state.raw_data["file_path"]:
    st.write(f"### Selected Data File: `{st.session_state.raw_data['file_path']}`")
else:
    st.warning("No data file selected. Please go to the **Raw Data** page and select a file.")

# Display selected date
if st.session_state.raw_data["date_index"] is not None:
    st.write(f"### Selected Date Index: `{st.session_state.raw_data['date_index']}`")
else:
    st.warning("No date selected. Please go to the **Raw Data** page and select a date.")

# Display current state for debugging
with st.sidebar.expander("Debug Info"):
    debug_state()