import streamlit as st
import os
import time
import torch
import numpy as np
from utils.states import init_state_variables, reset_state_variables, debug_state, create_selectbox_with_default
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

# Initialize managers with safety checks
if 'experiment_manager' not in st.session_state or st.session_state.experiment_manager is None:
    st.session_state.experiment_manager = ExperimentManager()
if 'metrics_calculator' not in st.session_state or st.session_state.metrics_calculator is None:
    st.session_state.metrics_calculator = MetricsCalculator()
if 'visualizer' not in st.session_state or st.session_state.visualizer is None:
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
# List and select .pt files
pt_files = list_pt_files(base_path_regression)

# Model selection using new function
reset_counter = st.session_state.get("reset_counter", 0)
selected_model_file, selected_model_index, is_model_default = create_selectbox_with_default(
    "Select a regression model:",
    pt_files,
    "model_selectbox_index",
    "Select model...",
    key=f"regression_model_selector_{reset_counter}",
    help_text="Choose a trained regression model file (.mdlus)"
)

# Update session variables based on selection
if is_model_default:
    # Reset to None when default option is selected
    st.session_state.regression["selected_model"] = None
    st.session_state.regression["model_index"] = None
elif selected_model_file:
    # Update when a new model is selected
    st.session_state.regression["selected_model"] = os.path.join(base_path_regression, selected_model_file)
    st.session_state.regression["model_index"] = selected_model_index
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
    if hasattr(st.session_state, 'current_results') and st.session_state.current_results is not None:
        st.markdown("---")
        st.subheader("📊 Regression Results")
        
        results = st.session_state.current_results
        
        # Validate that results is a valid dictionary
        if not isinstance(results, dict):
            st.error("❌ Invalid results format. Please run a new prediction.")
            st.info("💡 Try running a new prediction to generate valid results")
            #st.stop()
        
        # Normalize metrics structure to handle both new and loaded experiments
        def normalize_metrics(metrics):
            """Normalize metrics structure to ensure consistent access."""
            if not metrics:
                return None
                
            # If metrics has the nested structure (from loaded experiments)
            if 'overall' in metrics and isinstance(metrics['overall'], dict):
                return {
                    'overall_r2': metrics['overall'].get('r2', 0.0),
                    'overall_rmse': metrics['overall'].get('rmse', 0.0),
                    'overall_mae': metrics['overall'].get('mae', 0.0),
                    'u10': metrics.get('u10', {}),
                    'v10': metrics.get('v10', {})
                }
            # If metrics already has the flat structure (from new predictions)
            else:
                return metrics
        
        normalized_metrics = normalize_metrics(results.get('metrics'))
        
        if normalized_metrics:
            # Display metrics
            st.subheader("📈 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                r2_val = normalized_metrics.get('overall_r2', 0.0)
                st.metric("Overall R²", f"{r2_val:.4f}")
            with col2:
                rmse_val = normalized_metrics.get('overall_rmse', 0.0)
                st.metric("Overall RMSE", f"{rmse_val:.4f}")
            with col3:
                u10_r2 = normalized_metrics.get('u10', {}).get('r2', 0.0)
                st.metric("U10 R²", f"{u10_r2:.4f}")
            with col4:
                v10_r2 = normalized_metrics.get('v10', {}).get('r2', 0.0)
                st.metric("V10 R²", f"{v10_r2:.4f}")
            
            # Detailed metrics table
            st.subheader("📋 Detailed Metrics")
            try:
                metrics_df = st.session_state.metrics_calculator.format_metrics_for_display(normalized_metrics)
                st.dataframe(metrics_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying detailed metrics: {str(e)}")
                st.json(normalized_metrics)  # Fallback: show raw metrics
            
            # Visualizations
            st.subheader("📊 Visualizations")
            
            # Check if prediction arrays are available
            if all(key in results for key in ['u10_pred', 'v10_pred', 'u10_true', 'v10_true']):
                # Visualization options with session state
                viz_options = ["Predictions vs Ground Truth", "Scatter Plots", "Distribution Comparison", "Complete Summary"]
                
                # Use new selectbox function for visualization options
                selected_viz, selected_viz_index, is_viz_default = create_selectbox_with_default(
                    "Select visualization:",
                    viz_options,
                    "viz_option_index",
                    "Select visualization...",
                    key="regression_viz_selectbox"
                )
                
                if not is_viz_default:
                    st.session_state.regression["viz_option_index"] = selected_viz_index
                    
                    try:
                        if selected_viz == "Predictions vs Ground Truth":
                            fig = st.session_state.visualizer.plot_predictions_vs_truth(
                                results['u10_pred'], results['v10_pred'],
                                results['u10_true'], results['v10_true'],
                                title=f"Regression Results - {results['experiment_hash']}"
                            )
                            st.pyplot(fig)
                        
                        elif selected_viz == "Scatter Plots":
                            fig = st.session_state.visualizer.plot_scatter_comparison(
                                results['u10_pred'], results['v10_pred'],
                                results['u10_true'], results['v10_true'],
                                title=f"Scatter Comparison - {results['experiment_hash']}"
                            )
                            st.pyplot(fig)
                        
                        elif selected_viz == "Distribution Comparison":
                            fig = st.session_state.visualizer.plot_histograms(
                                results['u10_pred'], results['v10_pred'],
                                results['u10_true'], results['v10_true'],
                                title=f"Distribution Comparison - {results['experiment_hash']}"
                            )
                            st.pyplot(fig)
                        
                        elif selected_viz == "Complete Summary":
                            fig = st.session_state.visualizer.create_summary_plot(
                                results['u10_pred'], results['v10_pred'],
                                results['u10_true'], results['v10_true'],
                                normalized_metrics,
                                title=f"Complete Summary - {results['experiment_hash']}"
                            )
                            st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
            else:
                st.warning("⚠️ Prediction arrays not available for visualization")
        else:
            st.error("❌ No metrics available to display")
            st.info("💡 Try running a new prediction or check if the loaded experiment has valid metrics")

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
        # Use new selectbox function for experiment deletion
        selected_experiment, selected_exp_index, is_exp_default = create_selectbox_with_default(
            "Select experiment to delete:",
            experiments_df['Hash'].tolist(),
            "delete_experiment_index",
            "Select experiment to delete...",
            key="regression_delete_experiment_selectbox"
        )
        
        if not is_exp_default and selected_experiment:
            if st.button("🗑️ Delete Selected Experiment"):
                if st.session_state.experiment_manager.delete_experiment(selected_experiment):
                    st.success(f"✅ Experiment {selected_experiment} deleted successfully!")
                    # Reset the selectbox index after deletion
                    st.session_state.regression["delete_experiment_index"] = 0
                    st.rerun()
                else:
                    st.error(f"❌ Failed to delete experiment {selected_experiment}")
else:
    st.info("No experiments found. Run a prediction to create your first experiment!")

# Debug section
with st.sidebar.expander("🔍 Debug Info"):
    debug_state()