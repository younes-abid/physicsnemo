import streamlit as st
import os

def init_state_variables():
    # Raw Data state variables
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = {
            "base_path": "/app/data/custom_data_2/ERA5_WRF_combined_concatenated",
            "file": None,
            "file_path": None,
            "date": None,
            "date_index": None,
            "file_loaded": False,
            "nc_files": []
        }

    # Regression state variables
    if "regression" not in st.session_state:
        st.session_state.regression = {
            "selected_model": None,
            "done": False
        }
def reset_state_variables(page="all"):
    if page in ["all", "raw_data"]:
        st.session_state.raw_data = {
            "base_path": "/app/data/custom_data_2/ERA5_WRF_combined_concatenated",
            "file": None,
            "file_path": None,
            "date": None,
            "date_index": None,
            "file_loaded": False,
            "nc_files": []
        }
    if page in ["all", "regression"]:
        st.session_state.regression = {
            "selected_model": None,
            "done": False
        }
        
def debug_state():
    steps = [
        (
            "1️⃣ **Step 1:** Go to the **🔎 Raw Data** page and select a data file.",
            f"1️⃣ **Step 1:** ✅ Data file selected: `{st.session_state.raw_data['file_path']}`",
            st.session_state.raw_data["file_path"]
        ),
        (
            "2️⃣ **Step 2:** Go to the **🔎 Raw Data** page and select a date.",
            f"2️⃣ **Step 2:** ✅ Date selected: `{st.session_state.raw_data['date']}`",
            st.session_state.raw_data["date"]
        ),
        (
            "3️⃣ **Step 3:** Go to the **📈 Regression** page and select a regression model from the sidebar.",
            f"3️⃣ **Step 3:** ✅ Regression model selected: `{st.session_state.regression['selected_model']}`",
            st.session_state.regression["selected_model"]
        ),
        (
            "4️⃣ **Step 4:** Go to the **📈 Regression** page and run the regression model.",
            "4️⃣ **Step 4:** ✅ Regression completed.",
            st.session_state.regression["done"]
        ),
    ]

    for step_msg, done_msg, condition in steps:
        if condition:
            st.write(done_msg)
        else:
            st.write(step_msg)