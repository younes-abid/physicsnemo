import streamlit as st
import os
import time
import torch
import numpy as np
from utils.states import init_state_variables, reset_state_variables, debug_state
from utils.files import list_pt_files
from prediction import DataManager, RegressionPipeline, MetricsCalculator, Visualizer, ExperimentManager

# Initialize session state variables
init_state_variables()

# Page configuration
st.set_page_config(
    page_title="Weather Pipeline - Regression",
    page_icon="📈",
    layout="wide"
)

# Initialize managers
if 'experiment_manager' not in st.session_state:
    st.session_state.experiment_manager = ExperimentManager()
if 'metrics_calculator' not in st.session_state:
    st.session_state.metrics_calculator = MetricsCalculator()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = Visualizer()

# Sidebar for navigation and model selection
st.sidebar.header("🔧 Model Configuration")

# Base path input for regression models
base_path_regression = st.sidebar.text_input(
    "Regression Models Directory:",
    value="/app/checkpoints_regression/"
)

# Add a reset button
if st.sidebar.button("Reset Regression Selections"):
    reset_state_variables(page="regression")
    st.sidebar.success("Selections have been reset.")
    st.rerun()

# List and select .pt files
pt_files = list_pt_files(base_path_regression)
if not pt_files:
    st.sidebar.warning("No `.mdlus` files found in the selected directory.")
    st.session_state.regression["selected_model"] = None
    st.session_state.regression["model_index"] = None
else:
    # Prepare options with None as first option
    model_options = ["Select model..."] + pt_files
    
    # Get current index, default to 0 (Select...)
    current_model_index = st.session_state.regression["model_index"]
    if current_model_index is None:
        current_model_index = 0
    
    selected_model_index = st.sidebar.selectbox(
        "Select a regression model:",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        index=current_model_index,
        key="regression_model_selectbox"
    )
    
    # Update session state
    st.session_state.regression["model_index"] = selected_model_index
    
    if selected_model_index == 0:  # "Select model..." option
        st.session_state.regression["selected_model"] = None
    else:
        selected_model = model_options[selected_model_index]
        st.session_state.regression["selected_model"] = os.path.join(base_path_regression, selected_model)

# Main content
st.title("📈 Weather Regression Pipeline")
st.markdown("Run regression predictions on your weather data using the configured model and dataset.")

# Check if all required inputs are available using updated session variables
model_selected = st.session_state.regression["selected_model"] is not None
data_selected = st.session_state.raw_data["data_file_path"] is not None
stats_selected = st.session_state.raw_data["stats_file_path"] is not None
sample_selected = st.session_state.raw_data["date_index"] is not None

# Display current configuration
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Current Configuration")
    
    if model_selected:
        st.success(f"✓ Model: `{os.path.basename(st.session_state.regression['selected_model'])}`")
    else:
        st.error("✗ No model selected")
    
    if data_selected:
        st.success(f"✓ Data File: `{os.path.basename(st.session_state.raw_data['data_file_path'])}`")
    else:
        st.error("✗ No data file selected")
    
    if stats_selected:
        st.success(f"✓ Stats File: `{os.path.basename(st.session_state.raw_data['stats_file_path'])}`")
    else:
        st.error("✗ No stats file selected")
    
    if sample_selected:
        st.success(f"✓ Sample: Index `{st.session_state.raw_data['date_index']}` ({st.session_state.raw_data['date']})")
    else:
        st.error("✗ No sample selected")

with col2:
    st.subheader("🔍 Experiment Status")
    
    if model_selected and data_selected and stats_selected and sample_selected:
        # Generate experiment hash using updated session variables
        experiment_hash = st.session_state.experiment_manager.generate_experiment_hash(
            st.session_state.regression["selected_model"],
            st.session_state.raw_data["data_file_path"],
            st.session_state.raw_data["date_index"]
        )
        
        # Check if experiment already exists
        experiment_exists = st.session_state.experiment_manager.check_experiment_exists(experiment_hash)
        
        if experiment_exists:
            exp_info = st.session_state.experiment_manager.get_experiment_info(experiment_hash)
            st.warning(f"⚠️ Experiment already exists (Hash: `{experiment_hash}`)")
            if exp_info:
                st.info(f"Created: {exp_info['experiment_info']['created_at'][:19]}")
                st.info(f"Overall R²: {exp_info['metrics']['overall']['r2']:.4f}")
        else:
            st.info(f"🆕 New experiment (Hash: `{experiment_hash}`)")
    else:
        st.warning("Complete configuration to see experiment status")

st.markdown("---")

# Main regression section
if model_selected and data_selected and stats_selected and sample_selected:
    
    # Model configuration
    model_config = {
        "N_grid_channels": 4,
        "gridtype": "sinusoidal",
        "embedding_type": "zero",
        "model_channels": 128,
        "channel_mult": [1, 2, 2, 2, 2],
        "attn_resolutions": [28],
        "model_type": "SongUNetPosEmbd",
    }
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        load_model_btn = st.button("🔧 Load Regression Model", type="primary")
    
    with col2:
        run_prediction_btn = st.button("🚀 Run Prediction", disabled=True)
    
    with col3:
        force_rerun = st.checkbox("Force Re-run (ignore existing)", value=False)
    
    # Load model section
    if load_model_btn or 'regression_pipeline' not in st.session_state:
        with st.spinner("Loading regression model..."):
            try:
                # Create regression pipeline
                st.session_state.regression_pipeline = RegressionPipeline(
                    model_config=model_config,
                    checkpoint_path=st.session_state.regression["selected_model"]
                )
                
                # Load the model
                success = st.session_state.regression_pipeline.load_model()
                
                if success:
                    # Test model
                    test_success = st.session_state.regression_pipeline.test_model()
                    
                    if test_success:
                        st.success("✅ Model loaded and tested successfully!")
                        st.session_state.model_loaded = True
                        # Enable prediction button
                        run_prediction_btn = st.button("🚀 Run Prediction", disabled=False, key="run_pred_enabled")
                    else:
                        st.error("❌ Model test failed!")
                        st.session_state.model_loaded = False
                else:
                    st.error("❌ Failed to load model!")
                    st.session_state.model_loaded = False
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
                st.session_state.model_loaded = False
    
    # Display model information if loaded
    if hasattr(st.session_state, 'regression_pipeline') and hasattr(st.session_state, 'model_loaded') and st.session_state.model_loaded:
        model_info = st.session_state.regression_pipeline.get_model_info()
        
        st.subheader("🔧 Model Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Architecture", model_info['architecture'])
            st.metric("Parameters", f"{model_info['parameters']:,}")
        
        with col2:
            st.metric("Input Channels", model_info['input_channels'])
            st.metric("Output Channels", model_info['output_channels'])
        
        with col3:
            st.metric("Resolution", model_info['resolution'])
            st.metric("Device", model_info['device'])
        
        # Run prediction section
        if (run_prediction_btn or st.button("🚀 Run Prediction", disabled=False, key="run_pred_enabled2")) and (force_rerun or not experiment_exists):
            
            with st.spinner("Running regression prediction..."):
                try:
                    start_time = time.time()
                    
                    # Create data manager using updated session variables
                    data_manager = DataManager(
                        data_file_path=st.session_state.raw_data["data_file_path"],
                        stats_dir=st.session_state.raw_data["stats_dir_path"]
                    )
                    
                    # Load statistics and data
                    stats = data_manager.load_statistics()
                    input_data, output_data = data_manager.load_sample_data(
                        sample_idx=st.session_state.raw_data["date_index"]
                    )
                    
                    # Prepare input tensor
                    input_tensor = data_manager.prepare_input_tensor(apply_manual_normalization=True)
                    
                    # Run prediction
                    output_regression, u10_pred, v10_pred = st.session_state.regression_pipeline.predict(
                        input_tensor, data_manager
                    )
                    
                    # Get ground truth
                    u10_true, v10_true = data_manager.get_ground_truth()
                    
                    # Calculate metrics
                    metrics = st.session_state.metrics_calculator.compute_regression_metrics(
                        u10_pred, v10_pred, u10_true, v10_true, data_manager
                    )
                    
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    # Save experiment using updated session variables
                    experiment_dir = st.session_state.experiment_manager.save_experiment(
                        experiment_hash=experiment_hash,
                        model_path=st.session_state.regression["selected_model"],
                        data_path=st.session_state.raw_data["data_file_path"],
                        sample_idx=st.session_state.raw_data["date_index"],
                        u10_pred=u10_pred,
                        v10_pred=v10_pred,
                        u10_true=u10_true,
                        v10_true=v10_true,
                        metrics=metrics,
                        model_info=model_info,
                        execution_time=execution_time,
                        additional_info={
                            "stats_dir": st.session_state.raw_data["stats_dir_path"],
                            "stats_file": st.session_state.raw_data["stats_file_path"],
                            "manual_normalization": True
                        }
                    )
                    
                    st.success(f"✅ Prediction completed in {execution_time:.2f} seconds!")
                    st.success(f"💾 Results saved to: `{experiment_dir}`")
                    
                    # Store results in session state for display
                    st.session_state.current_results = {
                        'u10_pred': u10_pred,
                        'v10_pred': v10_pred,
                        'u10_true': u10_true,
                        'v10_true': v10_true,
                        'metrics': metrics,
                        'experiment_hash': experiment_hash
                    }
                    
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Load existing experiment if available
        elif experiment_exists and not force_rerun:
            if st.button("📂 Load Existing Results"):
                with st.spinner("Loading existing experiment..."):
                    try:
                        predictions, metadata = st.session_state.experiment_manager.load_experiment(experiment_hash)
                        
                        if predictions and metadata:
                            st.session_state.current_results = {
                                'u10_pred': predictions['u10_pred'],
                                'v10_pred': predictions['v10_pred'],
                                'u10_true': predictions['u10_true'],
                                'v10_true': predictions['v10_true'],
                                'metrics': metadata['metrics'],
                                'experiment_hash': experiment_hash
                            }
                            st.success("✅ Existing results loaded successfully!")
                        else:
                            st.error("❌ Failed to load existing results")
                    except Exception as e:
                        st.error(f"❌ Error loading results: {str(e)}")
    
    # Display results if available
    if hasattr(st.session_state, 'current_results'):
        st.markdown("---")
        st.subheader("📊 Regression Results")
        
        results = st.session_state.current_results
        
        # Display metrics
        st.subheader("📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall R²", f"{results['metrics']['overall_r2']:.4f}")
        with col2:
            st.metric("Overall RMSE", f"{results['metrics']['overall_rmse']:.4f}")
        with col3:
            st.metric("U10 R²", f"{results['metrics']['u10']['r2']:.4f}")
        with col4:
            st.metric("V10 R²", f"{results['metrics']['v10']['r2']:.4f}")
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        metrics_df = st.session_state.metrics_calculator.format_metrics_for_display(results['metrics'])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        # Visualization options with session state
        viz_options = ["Predictions vs Ground Truth", "Scatter Plots", "Distribution Comparison", "Complete Summary"]
        
        # Get current index, default to 0
        current_viz_index = st.session_state.regression["viz_option_index"]
        if current_viz_index is None:
            current_viz_index = 0
        
        selected_viz_index = st.selectbox(
            "Select visualization:",
            range(len(viz_options)),
            format_func=lambda x: viz_options[x],
            index=current_viz_index,
            key="regression_viz_selectbox"
        )
        
        # Update session state
        st.session_state.regression["viz_option_index"] = selected_viz_index
        viz_option = viz_options[selected_viz_index]
        
        if viz_option == "Predictions vs Ground Truth":
            fig = st.session_state.visualizer.plot_predictions_vs_truth(
                results['u10_pred'], results['v10_pred'],
                results['u10_true'], results['v10_true'],
                title=f"Regression Results - {results['experiment_hash']}"
            )
            st.pyplot(fig)
        
        elif viz_option == "Scatter Plots":
            fig = st.session_state.visualizer.plot_scatter_comparison(
                results['u10_pred'], results['v10_pred'],
                results['u10_true'], results['v10_true'],
                title=f"Scatter Comparison - {results['experiment_hash']}"
            )
            st.pyplot(fig)
        
        elif viz_option == "Distribution Comparison":
            fig = st.session_state.visualizer.plot_histograms(
                results['u10_pred'], results['v10_pred'],
                results['u10_true'], results['v10_true'],
                title=f"Distribution Comparison - {results['experiment_hash']}"
            )
            st.pyplot(fig)
        
        elif viz_option == "Complete Summary":
            fig = st.session_state.visualizer.create_summary_plot(
                results['u10_pred'], results['v10_pred'],
                results['u10_true'], results['v10_true'],
                results['metrics'],
                title=f"Complete Summary - {results['experiment_hash']}"
            )
            st.pyplot(fig)

else:
    st.warning("⚠️ Please complete the configuration:")
    if not model_selected:
        st.info("1. Select a regression model from the sidebar")
    if not data_selected:
        st.info("2. Go to **Raw Data** page and select a data file")
    if not stats_selected:
        st.info("3. Go to **Raw Data** page and select a stats file")
    if not sample_selected:
        st.info("4. Select a date/sample on the **Raw Data** page")

# Experiments history section
st.markdown("---")
st.subheader("📜 Experiments History")

experiments_df = st.session_state.experiment_manager.get_experiments_summary()
if not experiments_df.empty:
    st.dataframe(experiments_df, use_container_width=True)
    
    # Option to delete experiments
    if st.checkbox("Enable experiment management"):
        # Get current index, default to 0
        current_delete_index = st.session_state.regression["delete_experiment_index"]
        if current_delete_index is None:
            current_delete_index = 0
        
        # Prepare options with None as first option
        delete_options = ["Select experiment to delete..."] + experiments_df['Hash'].tolist()
        
        selected_delete_index = st.selectbox(
            "Select experiment to delete:",
            range(len(delete_options)),
            format_func=lambda x: delete_options[x],
            index=current_delete_index,
            key="regression_delete_experiment_selectbox"
        )
        
        # Update session state
        st.session_state.regression["delete_experiment_index"] = selected_delete_index
        
        if selected_delete_index > 0:  # Not "Select experiment..." option
            selected_hash = delete_options[selected_delete_index]
            if st.button("🗑️ Delete Selected Experiment"):
                if st.session_state.experiment_manager.delete_experiment(selected_hash):
                    st.success(f"✅ Experiment {selected_hash} deleted successfully!")
                    # Reset the selectbox index after deletion
                    st.session_state.regression["delete_experiment_index"] = 0
                    st.rerun()
                else:
                    st.error(f"❌ Failed to delete experiment {selected_hash}")
else:
    st.info("No experiments found. Run a prediction to create your first experiment!")

# Debug section
with st.sidebar.expander("🔍 Debug Info"):
    debug_state()