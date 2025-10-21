import streamlit as st
import os

def init_state_variables():
    # Raw Data state variables
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = {
            # Data file configuration
            "data_dir_path": "/app/data/custom_data_2/ERA5_WRF_combined_concatenated_432",
            "data_file": None,
            "data_file_path": None,
            "data_nc_files": [],
            "data_file_loaded": False,
            
            # Statistics file configuration
            "stats_dir_path": "/app/data/custom_data_2/stats_432",
            "stats_file": None,
            "stats_file_path": None,
            "stats_json_files": [],
            "stats_data": None,
            "stats_file_loaded": False,
            
            # Date/time selection
            "date": None,
            "date_index": None,
            "date_selectbox_index": None,
            
            # Variable selection for data exploration
            "input_var_index": 0,
            "output_var_index": 0,
        }

    # Regression state variables
    if "regression" not in st.session_state:
        st.session_state.regression = {
            "selected_model": None,
            "model_index": None,
            "done": False,
            "viz_option_index": 0,
            "delete_experiment_index": 0,
        }
    
    # Diffusion state variables (placeholder for future implementation)
    if "diffusion" not in st.session_state:
        st.session_state.diffusion = {
            "selected_model": None,
            "model_index": None,
            "done": False,
            "noise_level": 0.1,
            "num_steps": 50,
        }
    
    # Statistics/Visualization state variables
    if "statistics" not in st.session_state:
        st.session_state.statistics = {
            "selected_experiment": None,
            "comparison_experiments": [],
            "chart_type": "line",
            "metric_type": "r2",
        }
    
    # Global experiment management
    if "experiment_manager" not in st.session_state:
        st.session_state.experiment_manager = None
    if "metrics_calculator" not in st.session_state:
        st.session_state.metrics_calculator = None
    if "visualizer" not in st.session_state:
        st.session_state.visualizer = None
    if "regression_pipeline" not in st.session_state:
        st.session_state.regression_pipeline = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "current_results" not in st.session_state:
        st.session_state.current_results = None

def reset_state_variables(page="all"):
    # Add a reset counter to force widget recreation
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    st.session_state.reset_counter += 1
    
    if page in ["all", "raw_data"]:
        st.session_state.raw_data = {
            # Data file configuration
            "data_dir_path": "/app/data/custom_data_2/ERA5_WRF_combined_concatenated_432",
            "data_file": None,
            "data_file_path": None,
            "data_nc_files": [],
            "data_file_loaded": False,
            
            # Statistics file configuration
            "stats_dir_path": "/app/data/custom_data_2/stats_432",
            "stats_file": None,
            "stats_file_path": None,
            "stats_json_files": [],
            "stats_data": None,
            "stats_file_loaded": False,
            
            # Date/time selection
            "date": None,
            "date_index": None,
            "date_selectbox_index": None,
            
            # Variable selection for data exploration
            "input_var_index": 0,
            "output_var_index": 0,
        }
        
        # Reset selectbox indices used by create_selectbox_with_default
        keys_to_delete = ["data_file_selectbox_index", "stats_file_selectbox_index", "date_selectbox_index"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
    
    if page in ["all", "regression"]:
        st.session_state.regression = {
            "selected_model": None,
            "model_index": None,
            "done": False,
            "viz_option_index": 0,
            "delete_experiment_index": 0,
        }
        # Reset regression-related objects but don't set managers to None
        if "regression_pipeline" in st.session_state:
            st.session_state.regression_pipeline = None
        if "model_loaded" in st.session_state:
            st.session_state.model_loaded = False
        if "current_results" in st.session_state:
            st.session_state.current_results = None
            
        # Reset selectbox indices used by create_selectbox_with_default
        keys_to_delete = ["model_selectbox_index", "viz_option_index", "delete_experiment_index"]
        for key in keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
    
    if page in ["all", "diffusion"]:
        st.session_state.diffusion = {
            "selected_model": None,
            "model_index": None,
            "done": False,
            "noise_level": 0.1,
            "num_steps": 50,
        }
    
    if page in ["all", "statistics"]:
        st.session_state.statistics = {
            "selected_experiment": None,
            "comparison_experiments": [],
            "chart_type": "line",
            "metric_type": "r2",
        }
    
    # Ensure managers are always initialized (don't reset to None)
    if "experiment_manager" not in st.session_state or st.session_state.experiment_manager is None:
        from prediction import ExperimentManager
        st.session_state.experiment_manager = ExperimentManager()
    if "metrics_calculator" not in st.session_state or st.session_state.metrics_calculator is None:
        from prediction import MetricsCalculator
        st.session_state.metrics_calculator = MetricsCalculator()
    if "visualizer" not in st.session_state or st.session_state.visualizer is None:
        from prediction import Visualizer
        st.session_state.visualizer = Visualizer()

def debug_state():
    """Display pipeline progress with colored Streamlit messages and progress tracking."""
    
    # Define steps with their conditions and message types
    steps = [
        {
            "step": "1️⃣ **Data File Selection**",
            "condition": st.session_state.raw_data["data_file_path"] is not None,
            "success_msg": f"✅ Data file: `{os.path.basename(st.session_state.raw_data['data_file_path']) if st.session_state.raw_data['data_file_path'] else 'None'}`",
            "pending_msg": "📂 Go to **🔎 Raw Data** page and select a data file",
            "type": "data"
        },
        {
            "step": "2️⃣ **Statistics File Selection**",
            "condition": st.session_state.raw_data["stats_file_path"] is not None,
            "success_msg": f"✅ Stats file: `{os.path.basename(st.session_state.raw_data['stats_file_path']) if st.session_state.raw_data['stats_file_path'] else 'None'}`",
            "pending_msg": "📊 Go to **🔎 Raw Data** page and select a statistics file",
            "type": "data"
        },
        {
            "step": "3️⃣ **Date/Sample Selection**",
            "condition": st.session_state.raw_data["date"] is not None,
            "success_msg": f"✅ Date selected: `{st.session_state.raw_data['date']}`",
            "pending_msg": "📅 Go to **🔎 Raw Data** page and select a date for analysis",
            "type": "data"
        },
        {
            "step": "4️⃣ **Regression Model Selection**",
            "condition": st.session_state.regression["selected_model"] is not None,
            "success_msg": f"✅ Regression model: `{os.path.basename(st.session_state.regression['selected_model']) if st.session_state.regression['selected_model'] else 'None'}`",
            "pending_msg": "📈 Go to **📈 Regression** page and select a model",
            "type": "regression"
        },
        {
            "step": "5️⃣ **Regression Execution**",
            "condition": st.session_state.regression["done"] or (hasattr(st.session_state, 'current_results') and st.session_state.current_results is not None),
            "success_msg": "✅ Regression completed successfully",
            "pending_msg": "🚀 Run regression on **📈 Regression** page",
            "type": "regression"
        },
        {
            "step": "6️⃣ **Diffusion Model Selection** (Optional)",
            "condition": st.session_state.diffusion["selected_model"] is not None,
            "success_msg": f"✅ Diffusion model: `{st.session_state.diffusion['selected_model']}`",
            "pending_msg": "🎛️ Go to **🎛️ Diffusion** page and select a model",
            "type": "diffusion"
        },
        {
            "step": "7️⃣ **Diffusion Execution** (Optional)",
            "condition": st.session_state.diffusion["done"],
            "success_msg": "✅ Diffusion completed successfully",
            "pending_msg": "🎛️ Run diffusion on **🎛️ Diffusion** page",
            "type": "diffusion"
        }
    ]
    
    # Count completed steps
    completed_steps = sum(1 for step in steps if step["condition"])
    total_steps = len(steps)
    required_steps = 5  # First 5 steps are required for basic functionality
    completed_required = sum(1 for step in steps[:required_steps] if step["condition"])
    
    # Display overall progress
    st.subheader("🎯 Pipeline Progress")
    progress_percentage = completed_required / required_steps
    st.progress(progress_percentage)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Completed Required", completed_required, f"/ {required_steps}")
    with col2:
        st.metric("🎯 Total Progress", completed_steps, f"/ {total_steps}")
    with col3:
        completion_rate = int(progress_percentage * 100)
        st.metric("📊 Completion Rate", f"{completion_rate}%")
    
    st.markdown("---")
    
    # Display step-by-step status
    st.subheader("📋 Step-by-Step Status")
    
    for i, step in enumerate(steps):
        with st.container():
            if step["condition"]:
                st.success(f"{step['step']}")
                st.caption(step["success_msg"])
            else:
                if i < required_steps:
                    st.error(f"{step['step']}")
                    st.caption(step["pending_msg"])
                else:
                    st.info(f"{step['step']}")
                    st.caption(step["pending_msg"])
    
    # Display summary message
    st.markdown("---")
    st.subheader("📝 Summary")
    
    if completed_required == required_steps:
        st.success("🎉 **Congratulations!** All required steps completed. Your weather pipeline is ready!")
        st.balloons()
        
        # Show optional steps status
        optional_completed = completed_steps - completed_required
        optional_total = total_steps - required_steps
        if optional_completed > 0:
            st.info(f"✨ **Bonus:** {optional_completed}/{optional_total} optional steps completed!")
        else:
            st.info("💡 **Tip:** You can also explore the optional Diffusion features!")
            
    elif completed_required >= 3:
        remaining = required_steps - completed_required
        st.warning(f"🔄 **Almost there!** {remaining} more step{'s' if remaining > 1 else ''} to complete the pipeline.")
        
    else:
        remaining = required_steps - completed_required
        st.error(f"🚀 **Getting started:** {remaining} step{'s' if remaining > 1 else ''} remaining to run your first prediction.")
    
    # Advanced debug information (collapsible)
    with st.expander("🔍 Advanced Debug Information"):
        st.subheader("📊 Raw Session State")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Raw Data State:**")
            st.json({
                "data_file": st.session_state.raw_data.get("data_file"),
                "stats_file": st.session_state.raw_data.get("stats_file"),
                "date_index": st.session_state.raw_data.get("date_index"),
                "files_loaded": {
                    "data": st.session_state.raw_data.get("data_file_loaded", False),
                    "stats": st.session_state.raw_data.get("stats_file_loaded", False)
                }
            })
        
        with col2:
            st.write("**Model States:**")
            st.json({
                "regression": {
                    "model_selected": st.session_state.regression.get("selected_model") is not None,
                    "model_loaded": st.session_state.get("model_loaded", False),
                    "results_available": st.session_state.get("current_results") is not None,
                    "done": st.session_state.regression.get("done", False)
                },
                "diffusion": {
                    "model_selected": st.session_state.diffusion.get("selected_model") is not None,
                    "done": st.session_state.diffusion.get("done", False)
                }
            })

def create_selectbox_with_default(
    label: str,
    options: list,
    session_key: str,
    default_message: str = "Select option...",
    key: str = None,
    on_change=None,
    help_text: str = None,
    disabled: bool = False
):
    """
    Create a selectbox with a default "Select..." option and proper state management.
    
    Args:
        label: The label for the selectbox
        options: List of actual options (can be empty)
        session_key: Key to store the selected index in session state
        default_message: Message to show as first option (default: "Select option...")
        key: Streamlit widget key
        on_change: Callback function for when selection changes
        help_text: Help text for the selectbox
        disabled: Whether the selectbox is disabled
    
    Returns:
        tuple: (selected_value, selected_index, is_default_selected)
        - selected_value: The actual selected value (None if default option selected)
        - selected_index: The index in the original options list (None if default selected)
        - is_default_selected: True if the default "Select..." option is selected
    """
    
    # Handle edge cases
    if not options:
        # Empty list case
        st.selectbox(
            label,
            ["No options available"],
            index=0,
            disabled=True,
            help=help_text or "No options found in the current directory"
        )
        return None, None, True
    
    if len(options) == 1:
        # Single option case - show selectbox but don't auto-select unless previously selected
        display_options = [default_message] + options
        
        # Get current index, default to 0 (Select...) instead of 1 (auto-select)
        current_index = st.session_state.get(session_key, 0)
        if current_index is None or current_index >= len(display_options):
            current_index = 0
        
        selected_index = st.selectbox(
            label,
            range(len(display_options)),
            format_func=lambda x: display_options[x],
            index=current_index,
            key=key,
            on_change=on_change,
            help=help_text,
            disabled=disabled
        )
        
        # Update session state
        st.session_state[session_key] = selected_index
        
        if selected_index == 0:
            return None, None, True
        else:
            return options[selected_index - 1], selected_index - 1, False
    
    # Multiple options case
    display_options = [default_message] + options
    
    # Get current index, default to 0 (Select...)
    current_index = st.session_state.get(session_key, 0)
    if current_index is None or current_index >= len(display_options):
        current_index = 0
    
    selected_index = st.selectbox(
        label,
        range(len(display_options)),
        format_func=lambda x: display_options[x],
        index=current_index,
        key=key,
        on_change=on_change,
        help=help_text,
        disabled=disabled
    )
    
    # Update session state
    st.session_state[session_key] = selected_index
    
    if selected_index == 0:  # Default option selected
        return None, None, True
    else:
        actual_index = selected_index - 1
        return options[actual_index], actual_index, False