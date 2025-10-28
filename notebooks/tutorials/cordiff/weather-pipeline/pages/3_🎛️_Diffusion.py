import streamlit as st
import os
import time
import torch
import numpy as np
from utils.states import init_state_variables, reset_state_variables, debug_state, create_selectbox_with_default
from utils.files import list_mdlus_files
from prediction import DataManager, DiffusionPipeline, MetricsCalculator, Visualizer, ExperimentManager
import matplotlib.pyplot as plt

# Initialize session state variables
init_state_variables()

# Page configuration
st.set_page_config(
    page_title="Weather Pipeline - Diffusion",
    page_icon="🎛️",
    layout="wide"
)

def handle_model_dir_change():
    """Handle diffusion model directory path changes."""
    new_model_dir = st.session_state.model_dir_input
    if new_model_dir != st.session_state.diffusion["model_dir_path"]:
        st.session_state.diffusion["model_dir_path"] = new_model_dir
        st.session_state.diffusion["model_files"] = list_mdlus_files(new_model_dir)
        reset_diffusion_from_model_selection()

def handle_model_file_selection():
    """Handle diffusion model file selection changes."""
    reset_counter = st.session_state.get("reset_counter", 0)
    
    selected_file, selected_index, is_default = create_selectbox_with_default(
        "📄 Select Diffusion Model (.mdlus):",
        st.session_state.diffusion["model_files"],
        st.session_state.diffusion,
        "model_selectbox_index",
        "Select diffusion model file...",
        key=f"diffusion_model_file_selector_{reset_counter}",
        help_text="Choose a trained diffusion model file"
    )
    
    # Update session variables based on selection
    if is_default:
        reset_diffusion_from_model_selection()
    elif selected_file != st.session_state.diffusion["selected_model_file"]:
        # Update when a new file is selected
        st.session_state.diffusion["selected_model_file"] = selected_file
        st.session_state.diffusion["selected_model"] = os.path.join(
            st.session_state.diffusion["model_dir_path"], 
            selected_file
        )
        # Reset downstream states when model changes
        reset_diffusion_from_model_loading()
        # Force a rerun to update the hash display immediately
        st.rerun()

def reset_diffusion_from_model_selection():
    """Reset all diffusion states when model selection changes."""
    st.session_state.diffusion["selected_model"] = None
    st.session_state.diffusion["selected_model_file"] = None
    reset_diffusion_from_model_loading()

def reset_diffusion_from_model_loading():
    """Reset diffusion states from model loading onwards."""
    st.session_state.diffusion["model_loaded"] = False
    st.session_state.diffusion["model_info"] = None
    st.session_state.diffusion["model_validation_status"] = None
    st.session_state.diffusion["experiment_completed"] = False
    st.session_state.diffusion["current_results"] = None
    st.session_state.diffusion["show_results"] = False

def load_diffusion_model():
    """Load and validate the selected diffusion model."""
    if not st.session_state.diffusion["selected_model"]:
        return False
    
    try:
        with st.spinner("Loading diffusion model..."):
            # Use the EXACT same configuration structure as the working debug file
            OUTPUT_DIR = "/app/"
            
            # DIFFUSION CONFIGURATION (EXACT COPY from working debug file)
            diffusion_config = {
                "model_config": {
                    # Model architecture from diffusion_normal.yaml (EXACT MATCH)
                    "gridtype": "sinusoidal",
                    "N_grid_channels": 4,
                    "embedding_type": "zero",  # Changed from "positional" to "zero" like debug
                    "model_channels": 128,
                    "channel_mult": [1, 2, 2, 2, 2],
                    "attn_resolutions": [28],
                    "model_type": "SongUNetPosEmbd",
                    
                    # Model execution config (EXACT MATCH with training)
                    "use_fp16": False,
                    "checkpoint_level": 0,
                    "hr_mean_conditioning": True,  # This is the ONLY parameter passed to ResidualLoss
                },
                "execution_config": {
                    # Execution parameters from train.py validation (EXACT MATCH)
                    "use_apex_gn": False,
                    "enable_amp": False,
                    "amp_dtype": "float32",
                    "profile_mode": False,
                    "batch_size_per_gpu": 1,
                    "use_patch_grad_acc": None,
                    "patching": None,
                    "patch_nums_iter": [1],
                },
                "checkpoint_config": {
                    "checkpoint_dir": OUTPUT_DIR + "checkpoints_diffusion/",
                    "checkpoint_index": 265008,
                },
                # Add the regression path that ResidualLoss needs internally
                "training_config": {
                    "regression_checkpoint_path": OUTPUT_DIR + "checkpoints_regression/UNet.0.535008.mdlus",
                }
            }
            
            # Create diffusion pipeline with full config (like debug file)
            st.session_state.diffusion["diffusion_pipeline"] = DiffusionPipeline(diffusion_config)
            
            # Load the model
            success = st.session_state.diffusion["diffusion_pipeline"].load_model()
            
            if success:
                # Test model by creating loss function
                test_success = st.session_state.diffusion["diffusion_pipeline"].create_loss_function()
                
                if test_success:
                    # Get model info
                    st.session_state.diffusion["model_info"] = st.session_state.diffusion["diffusion_pipeline"].get_model_info()
                    st.session_state.diffusion["model_loaded"] = True
                    st.session_state.diffusion["model_validation_status"] = "success"
                    return True
                else:
                    st.session_state.diffusion["model_loaded"] = False
                    st.session_state.diffusion["model_validation_status"] = "loss_function_failed"
                    return False
            else:
                st.session_state.diffusion["model_loaded"] = False
                st.session_state.diffusion["model_validation_status"] = "load_failed"
                return False
                
    except Exception as e:
        st.session_state.diffusion["model_loaded"] = False
        st.session_state.diffusion["model_validation_status"] = "error"
        st.error(f"Error details: {str(e)}")
        return False

def run_diffusion_experiment(experiment_hash, is_rerun=False):
    """Run or re-run diffusion experiment."""
    try:
        st.session_state.diffusion["experiment_running"] = True
        
        with st.spinner("Running diffusion prediction..."):
            start_time = time.time()
            
            # DETERMINISTIC EXECUTION: Set seeds for reproducible results
            experiment_seed = hash(experiment_hash) % (2**31)  # Use experiment hash as seed
            np.random.seed(experiment_seed)
            torch.manual_seed(experiment_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(experiment_seed)
                torch.cuda.manual_seed_all(experiment_seed)
            
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            st.info(f"🎲 Using deterministic seed: {experiment_seed} (from experiment hash)")
            
            # Create data manager
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
            
            # STEP 1: First we need to run regression to get the baseline for diffusion
            # Check if regression model is loaded and available
            if not st.session_state.regression.get("selected_model"):
                st.error("❌ Diffusion requires a regression baseline. Please go to the Regression page and select a model first.")
                return False
            
            # Load regression pipeline if not already loaded
            if not st.session_state.regression.get("regression_pipeline"):
                # Create regression config for loading
                regression_model_config = {
                    "N_grid_channels": 4,
                    "gridtype": "sinusoidal", 
                    "embedding_type": "zero",
                    "model_channels": 128,
                    "channel_mult": [1, 2, 2, 2, 2],
                    "attn_resolutions": [28],
                    "model_type": "SongUNetPosEmbd",
                }
                
                # Import and create regression pipeline
                from prediction import RegressionPipeline
                st.session_state.regression["regression_pipeline"] = RegressionPipeline(
                    model_config=regression_model_config,
                    checkpoint_path=st.session_state.regression["selected_model"]
                )
                
                # Load regression model
                regression_success = st.session_state.regression["regression_pipeline"].load_model()
                if not regression_success:
                    st.error("❌ Failed to load regression model for diffusion baseline")
                    return False
            
            # Run regression to get baseline
            st.info("🔄 Running regression to get baseline for diffusion...")
            regression_output, u10_reg, v10_reg = st.session_state.regression["regression_pipeline"].predict(
                input_tensor, data_manager
            )
            
            # STEP 2: Now run diffusion with regression baseline
            st.info("🔄 Running diffusion enhancement...")
            output_diffusion, u10_pred, v10_pred = st.session_state.diffusion["diffusion_pipeline"].predict(
                input_tensor, regression_output, data_manager
            )
            
            # Get ground truth
            u10_true, v10_true = data_manager.get_ground_truth()
            
            # Calculate metrics
            metrics = st.session_state.diffusion["metrics_calculator"].compute_metrics(
                u10_pred, v10_pred, u10_true, v10_true, data_manager
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Save experiment WITHOUT regression baseline in metadata (to keep YAML files small)
            experiment_dir = st.session_state.diffusion["experiment_manager"].save_experiment(
                experiment_hash=experiment_hash,
                model_path=st.session_state.diffusion["selected_model"],
                data_path=st.session_state.raw_data["data_file_path"],
                sample_idx=st.session_state.raw_data["date_index"],
                u10_pred=u10_pred,
                v10_pred=v10_pred,
                u10_true=u10_true,
                v10_true=v10_true,
                metrics=metrics,
                model_info=st.session_state.diffusion["model_info"],
                execution_time=execution_time,
                additional_info={
                    "stats_dir": st.session_state.raw_data["stats_dir_path"],
                    "stats_file": st.session_state.raw_data["stats_file_path"],
                    "manual_normalization": True,
                    "regression_baseline_available": True,  # Flag to indicate baseline exists
                    "regression_model_path": st.session_state.regression["selected_model"],  # Reference to regression model used
                    "experiment_seed": experiment_seed  # Store seed for reproducibility
                }
            )
            
            # OPTIONAL: Save regression baseline as separate NetCDF file (efficient storage)
            try:
                import xarray as xr
                from datetime import datetime
                
                # Create regression baseline dataset
                baseline_ds = xr.Dataset({
                    'u10_regression': (['y', 'x'], u10_reg),
                    'v10_regression': (['y', 'x'], v10_reg),
                })
                
                # Add metadata (fix: use current timestamp instead of undefined metadata)
                baseline_ds.attrs['description'] = 'Regression baseline for diffusion experiment'
                baseline_ds.attrs['experiment_hash'] = experiment_hash
                baseline_ds.attrs['regression_model'] = os.path.basename(st.session_state.regression["selected_model"])
                baseline_ds.attrs['created_at'] = datetime.now().isoformat()
                baseline_ds.attrs['experiment_seed'] = experiment_seed
                
                # Save as NetCDF file in experiment directory
                baseline_file = os.path.join(experiment_dir, 'regression_baseline.nc')
                baseline_ds.to_netcdf(baseline_file)
                
                st.info(f"💾 Regression baseline saved to: {os.path.basename(baseline_file)}")
                
            except Exception as e:
                st.warning(f"⚠️ Could not save regression baseline as NetCDF: {str(e)}")
                # Continue without baseline file - not critical
            
            # Store results in session state
            st.session_state.diffusion["current_results"] = {
                'u10_pred': u10_pred,
                'v10_pred': v10_pred,
                'u10_true': u10_true,
                'v10_true': v10_true,
                'metrics': metrics,
                'experiment_hash': experiment_hash,
                'execution_time': execution_time,
                'experiment_seed': experiment_seed,  # Store seed for display
                'regression_baseline': {
                    'u10_reg': u10_reg,
                    'v10_reg': v10_reg
                }
            }
            
            st.session_state.diffusion["experiment_completed"] = True
            st.session_state.diffusion["show_results"] = True
            st.session_state.diffusion["done"] = True
            
            st.success(f"✅ Experiment completed with deterministic seed: {experiment_seed}")
            
            return True
            
    except Exception as e:
        st.error(f"❌ Experiment failed: {str(e)}")
        return False
    finally:
        st.session_state.diffusion["experiment_running"] = False

def normalize_metrics(metrics):
    """Normalize metrics structure to handle both nested and flat formats."""
    if not metrics:
        return None
    if 'overall' in metrics and isinstance(metrics['overall'], dict):
        return {
            'overall_r2': metrics['overall'].get('r2', 0.0),
            'overall_rmse': metrics['overall'].get('rmse', 0.0),
            'overall_mae': metrics['overall'].get('mae', 0.0),
            'u10': metrics.get('u10', {}),
            'v10': metrics.get('v10', {})
        }
    else:
        return metrics

def display_experiment_results(results):
    """Display experiment results with metrics."""
    if not results:
        st.warning("⚠️ No results to display")
        return
    
    normalized_metrics = normalize_metrics(results.get('metrics'))
    
    if normalized_metrics:
        # Performance Metrics Dashboard
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
        
        # Execution time if available
        if 'execution_time' in results:
            st.metric("⏱️ Execution Time", f"{results['execution_time']:.2f}s")
    else:
        st.error("❌ No metrics available to display")

def display_visualizations(results, key_suffix=""):
    """Display interactive visualizations."""
    if not results or not all(key in results for key in ['u10_pred', 'v10_pred', 'u10_true', 'v10_true']):
        st.info("Visualizations will appear here after running an experiment")
        return
    
    viz_options = ["Predictions vs Ground Truth", "Scatter Plots", "Distribution Comparison", "Complete Summary"]
    
    # Use key_suffix to make keys unique for different contexts
    selectbox_key = f"diffusion_viz_selectbox{key_suffix}"
    session_key = f"viz_option_selectbox_index{key_suffix}" if key_suffix else "viz_option_selectbox_index"
    
    selected_viz, selected_viz_index, is_viz_default = create_selectbox_with_default(
        "Select visualization:",
        viz_options,
        st.session_state.diffusion,
        session_key, 
        "Select visualization...",
        key=selectbox_key
    )
    
    if not is_viz_default:
        try:
            if selected_viz == "Predictions vs Ground Truth":
                fig = st.session_state.diffusion["visualizer"].plot_predictions_vs_truth(
                    results['u10_pred'], results['v10_pred'],
                    results['u10_true'], results['v10_true'],
                    title=f"Diffusion Results - {results['experiment_hash']}"
                )
                st.pyplot(fig)
            
            elif selected_viz == "Scatter Plots":
                fig = st.session_state.diffusion["visualizer"].plot_scatter_comparison(
                    results['u10_pred'], results['v10_pred'],
                    results['u10_true'], results['v10_true'],
                    title=f"Scatter Comparison - {results['experiment_hash']}"
                )
                st.pyplot(fig)
            
            elif selected_viz == "Distribution Comparison":
                fig = st.session_state.diffusion["visualizer"].plot_histograms(
                    results['u10_pred'], results['v10_pred'],
                    results['u10_true'], results['v10_true'],
                    title=f"Distribution Comparison - {results['experiment_hash']}"
                )
                st.pyplot(fig)
            
            elif selected_viz == "Complete Summary":
                normalized_metrics = normalize_metrics(results.get('metrics'))
                fig = st.session_state.diffusion["visualizer"].create_summary_plot(
                    results['u10_pred'], results['v10_pred'],
                    results['u10_true'], results['v10_true'],
                    normalized_metrics,
                    title=f"Complete Summary - {results['experiment_hash']}"
                )
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

def display_experiments_history():
    """Display experiments history with individual and aggregate views."""
    st.markdown("## 📜 Experiments History")
    
    experiments_df = st.session_state.diffusion["experiment_manager"].get_experiments_summary()
    
    if experiments_df.empty:
        st.info("No experiments found. Run a prediction to create your first experiment!")
        return
    
    # Display experiments table
    st.dataframe(experiments_df, use_container_width=True)
    
    # View mode selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        view_mode = st.radio(
            "View Mode:",
            ["Individual Experiment", "Aggregate Analysis"],
            index=0 if st.session_state.diffusion["history_view_mode"] == "individual" else 1,
            horizontal=True
        )
        st.session_state.diffusion["history_view_mode"] = "individual" if view_mode == "Individual Experiment" else "aggregate"
    
    with col2:
        # Management actions
        if st.checkbox("Enable experiment management"):
            selected_experiment, selected_exp_index, is_exp_default = create_selectbox_with_default(
                "Select experiment to delete:",
                experiments_df['Hash'].tolist(),
                st.session_state.diffusion,
                "delete_experiment_selectbox_index",
                "Select experiment to delete...",
                key="diffusion_delete_experiment_selectbox"
            )
            
            if not is_exp_default and selected_experiment:
                if st.button("🗑️ Delete Selected Experiment"):
                    if st.session_state.diffusion["experiment_manager"].delete_experiment(selected_experiment):
                        st.success(f"✅ Experiment {selected_experiment} deleted successfully!")
                        st.session_state.diffusion["delete_experiment_selectbox_index"] = 0
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to delete experiment {selected_experiment}")
    
    st.markdown("---")
    
    if st.session_state.diffusion["history_view_mode"] == "individual":
        # Individual experiment view
        st.subheader("🔍 Individual Experiment Results")
        
        selected_exp, selected_exp_index, is_exp_default = create_selectbox_with_default(
            "Select experiment to view results:",
            experiments_df['Hash'].tolist(),
            st.session_state.diffusion,
            "history_experiment_selectbox_index",
            "Select experiment to view...",
            key="diffusion_history_experiment_selectbox"
        )
        
        if not is_exp_default and selected_exp:
            try:
                predictions, metadata = st.session_state.diffusion["experiment_manager"].load_experiment(selected_exp)
                
                if predictions and metadata:
                    # Create results structure for display
                    historical_results = {
                        'u10_pred': predictions['u10_pred'],
                        'v10_pred': predictions['v10_pred'],
                        'u10_true': predictions['u10_true'],
                        'v10_true': predictions['v10_true'],
                        'metrics': metadata['metrics'],
                        'experiment_hash': selected_exp,
                        'execution_time': metadata['experiment_info']['execution_time_seconds']
                    }
                    
                    # Display experiment info
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Created:** {metadata['experiment_info']['created_at'][:19].replace('T', ' ')}")
                    with col2:
                        st.info(f"**Model:** {os.path.basename(metadata['input_data']['model_path'])}")
                    with col3:
                        st.info(f"**Sample:** Index {metadata['input_data']['sample_idx']}")
                    
                    # Display results
                    display_experiment_results(historical_results)
                    
                    # Add visualization section for historical experiments
                    st.markdown("#### 📊 Historical Experiment Visualizations")
                    display_visualizations(historical_results, key_suffix="_history")
                    
                else:
                    st.error("❌ Failed to load experiment data")
            except Exception as e:
                st.error(f"❌ Error loading experiment: {str(e)}")
    
    else:
        # Aggregate analysis view
        st.subheader("📊 Aggregate Analysis")
        
        try:
            # Load all experiments for aggregation
            all_experiments = []
            for exp_hash in experiments_df['Hash'].tolist():
                try:
                    predictions, metadata = st.session_state.diffusion["experiment_manager"].load_experiment(exp_hash)
                    if predictions and metadata:
                        all_experiments.append({
                            'hash': exp_hash,
                            'metrics': metadata['metrics'],
                            'created_at': metadata['experiment_info']['created_at'],
                            'execution_time': metadata['experiment_info']['execution_time_seconds']
                        })
                except:
                    continue
            
            if all_experiments:
                # Calculate aggregate statistics
                r2_values = []
                rmse_values = []
                exec_times = []
                
                for exp in all_experiments:
                    if 'overall_r2' in exp['metrics']:
                        r2_values.append(exp['metrics']['overall_r2'])
                    elif 'overall' in exp['metrics']:
                        r2_values.append(exp['metrics']['overall']['r2'])
                    
                    if 'overall_rmse' in exp['metrics']:
                        rmse_values.append(exp['metrics']['overall_rmse'])
                    elif 'overall' in exp['metrics']:
                        rmse_values.append(exp['metrics']['overall']['rmse'])
                        
                    exec_times.append(exp['execution_time'])
                
                # Display aggregate metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if r2_values:
                        st.metric("Mean R²", f"{np.mean(r2_values):.4f}", f"±{np.std(r2_values):.4f}")
                with col2:
                    if rmse_values:
                        st.metric("Mean RMSE", f"{np.mean(rmse_values):.4f}", f"±{np.std(rmse_values):.4f}")
                with col3:
                    if exec_times:
                        st.metric("Mean Exec Time", f"{np.mean(exec_times):.2f}s", f"±{np.std(exec_times):.2f}s")
                with col4:
                    st.metric("Total Experiments", len(all_experiments))
                
                # Aggregate plots
                if r2_values and rmse_values:
                    
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
                    
                    # R² distribution
                    ax1.hist(r2_values, bins=min(10, len(r2_values)), alpha=0.7, color='skyblue', edgecolor='black')
                    ax1.set_title('R² Distribution Across Experiments')
                    ax1.set_xlabel('R² Score')
                    ax1.set_ylabel('Frequency')
                    
                    # RMSE distribution
                    ax2.hist(rmse_values, bins=min(10, len(rmse_values)), alpha=0.7, color='lightcoral', edgecolor='black')
                    ax2.set_title('RMSE Distribution Across Experiments')
                    ax2.set_xlabel('RMSE')
                    ax2.set_ylabel('Frequency')
                    
                    # Performance over time
                    dates = [exp['created_at'] for exp in all_experiments]
                    ax3.plot(range(len(r2_values)), r2_values, 'o-', color='blue', label='R²')
                    ax3.set_title('R² Performance Over Experiments')
                    ax3.set_xlabel('Experiment Order')
                    ax3.set_ylabel('R² Score')
                    
                    # Execution time trend
                    ax4.plot(range(len(exec_times)), exec_times, 'o-', color='green', label='Execution Time')
                    ax4.set_title('Execution Time Trend')
                    ax4.set_xlabel('Experiment Order')
                    ax4.set_ylabel('Time (seconds)')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("⚠️ No valid experiments found for aggregation")
                
        except Exception as e:
            st.error(f"❌ Error in aggregate analysis: {str(e)}")

# Main content
st.title("🎛️ Weather Diffusion Pipeline")
st.markdown("Run diffusion predictions on your weather data using the configured diffusion model and dataset.")

st.markdown("---")

# Check if all required inputs are available
model_selected = st.session_state.diffusion["selected_model"] is not None
data_selected = st.session_state.raw_data["data_file_path"] is not None
stats_selected = st.session_state.raw_data["stats_file_path"] is not None
sample_selected = st.session_state.raw_data["date_index"] is not None

# Data configuration display (read-only from Raw Data page)
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 Data File Configuration")
    
    # Display data directory path (read-only from Raw Data page)
    st.text_input(
        "📂 Data Directory Path:",
        value=st.session_state.raw_data["data_dir_path"],
        disabled=True,
        help="Path configured in Raw Data page"
    )
    
    if data_selected:
        st.text_input(
            "📄 Selected Data File (.nc):", 
            value=st.session_state.raw_data["data_file"], 
            disabled=True,
            help="Data file selected in Raw Data page"
        )
        if sample_selected:
            st.success(f"✅ **Selected Date:** {st.session_state.raw_data['date']} (Index `{st.session_state.raw_data['date_index']}`)")
        else:
            st.warning("⚠️ Go to **🔎 Raw Data** page and select a date for analysis")
    else:
        st.warning("⚠️ Go to **🔎 Raw Data** page and select a data file")

with col2:
    st.markdown("### 📊 Statistics File Configuration")
    
    # Display statistics directory path (read-only from Raw Data page)
    st.text_input(
        "📂 Statistics Directory Path:",
        value=st.session_state.raw_data["stats_dir_path"],
        disabled=True,
        help="Path configured in Raw Data page"
    )
    
    if stats_selected:
        st.text_input(
            "📄 Selected Statistics File (.json):", 
            value=st.session_state.raw_data["stats_file"], 
            disabled=True,
            help="Statistics file selected in Raw Data page"
        )
    else:
        st.warning("⚠️ Go to **🔎 Raw Data** page and select a statistics file")

st.markdown("---")

# 1. 📁 Regression and Diffusion Model Configuration
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📁 Regression Model Configuration")
    
    # Display regression model directory path (read-only from Regression page)
    st.text_input(
        "📂 Regression Model Directory Path:",
        value=st.session_state.regression["model_dir_path"],
        disabled=True,
        help="Path configured in the Regression page"
    )
    
    # Display selected regression model or warning
    if st.session_state.regression["selected_model_file"]:
        # Show the selected regression model
        st.selectbox(
            "📄 Select Regression Model:",
            [st.session_state.regression["selected_model_file"]],
            index=0,
            disabled=True,
            help="Regression model selected in the dedicated Regression page"
        )
        
        # Show regression experiment hash if available
        if st.session_state.regression["selected_model"] and data_selected and stats_selected and sample_selected:
            regression_experiment_hash = st.session_state.regression["experiment_manager"].generate_experiment_hash(
                st.session_state.regression["selected_model"],
                st.session_state.raw_data["data_file_path"],
                st.session_state.raw_data["date_index"]
            )
            st.info(f"**Regression Experiment Hash:** `{regression_experiment_hash}`")
        else:
            st.info("**Regression Experiment Hash:** `-`")
    else:
        # Show warning when no regression model is selected
        st.selectbox(
            "📄 Select Regression Model:",
            ["No regression model selected"],
            index=0,
            disabled=True,
            help="Go to the Regression page to select a model"
        )
        st.info("**Regression Experiment Hash:** `-`")
        st.warning("⚠️ **Go to 📈 Regression page and select a regression model**")
    

with col2:
    st.markdown("## 📁 Diffusion Model Configuration")
    
    # Model directory and file selection - ACTIVE for diffusion page
    st.text_input(
        "📂 Model Directory Path:",
        value=st.session_state.diffusion["model_dir_path"],
        key="model_dir_input",
        on_change=handle_model_dir_change,
        help="Path to directory containing diffusion model files"
    )
    
    # Update file list if needed
    if not st.session_state.diffusion["model_files"]:
        st.session_state.diffusion["model_files"] = list_mdlus_files(st.session_state.diffusion["model_dir_path"])
    
    # Model file selection
    handle_model_file_selection()
    
    # Experiment hash and status (merged from experiment status section)
    if model_selected and data_selected and stats_selected and sample_selected:
        experiment_hash = st.session_state.diffusion["experiment_manager"].generate_experiment_hash(
            st.session_state.diffusion["selected_model"],
            st.session_state.raw_data["data_file_path"],
            st.session_state.raw_data["date_index"]
        )
        experiment_exists = st.session_state.diffusion["experiment_manager"].check_experiment_exists(experiment_hash)
        
        st.info(f"**Diffusion Experiment Hash:** `{experiment_hash}`")
        
        if experiment_exists:
            st.success("🔄 **Experiment already exists**")
        else:
            st.success("🆕 **New experiment ready to run**")
    else:
        st.info("**Experiment Hash:** `-`")
        st.warning("⚠️ Complete configuration to generate experiment hash")

st.markdown("---")

# 2. 🤖 Diffusion Model Management
st.markdown("## 🤖 Diffusion Model Management")

if not model_selected:
    st.warning("⚠️ Please select a diffusion model to be able to load it")
else:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("🔧 Load & Validate Diffusion Model"):
            if load_diffusion_model():
                st.rerun()
            else:
                st.error("❌ Failed to load and validate diffusion model")
    
    with col2:
        if st.session_state.diffusion["model_loaded"]:
            st.success("✅ **Diffusion model loaded and validated**")
        else:
            st.warning("⚠️ Click 'Load & Validate Diffusion Model' to proceed")

    # Model Information Panel
    st.markdown("#### 📋 Diffusion Model Information")
    
    if st.session_state.diffusion["model_loaded"] and st.session_state.diffusion["model_info"]:
        model_info = st.session_state.diffusion["model_info"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏗️ Architecture", model_info.get('architecture', 'N/A'))
        with col2:
            st.metric("🔢 Parameters", f"{model_info.get('parameters', 0):,}")
        with col3:
            st.metric("📥 Input Channels", model_info.get('input_channels', 'N/A'))
        with col4:
            st.metric("📤 Output Channels", model_info.get('output_channels', 'N/A'))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📏 Resolution", model_info.get('resolution', 'N/A'))
        with col2:
            st.metric("🖥️ Device", model_info.get('device', 'N/A'))
        with col3:
            st.metric("✅ Validation", st.session_state.diffusion["model_validation_status"])
    else:
        # Show placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏗️ Architecture", "-")
        with col2:
            st.metric("🔢 Parameters", "-")
        with col3:
            st.metric("📥 Input Channels", "-")
        with col4:
            st.metric("📤 Output Channels", "-")

st.markdown("---")

# 3. 🚀 Experiment Execution
st.markdown("## 🚀 Experiment Execution")

# Check prerequisites
model_loaded = st.session_state.diffusion["model_loaded"]
can_run_experiment = model_loaded and model_selected and data_selected and stats_selected and sample_selected

if not can_run_experiment:
    st.warning("⚠️  Please load diffusion model to run experiment")

col1, col2, col3 = st.columns(3)

with col1:
    if can_run_experiment:
        experiment_hash = st.session_state.diffusion["experiment_manager"].generate_experiment_hash(
            st.session_state.diffusion["selected_model"],
            st.session_state.raw_data["data_file_path"],
            st.session_state.raw_data["date_index"]
        )
        experiment_exists = st.session_state.diffusion["experiment_manager"].check_experiment_exists(experiment_hash)
        
        if experiment_exists:
            if st.button("🔄 Re-run Experiment", type="primary"):
                if run_diffusion_experiment(experiment_hash, is_rerun=True):
                    st.rerun()
        else:
            if st.button("🆕 Run New Experiment", type="primary"):
                if run_diffusion_experiment(experiment_hash, is_rerun=False):
                    st.rerun()
    else:
        st.button("🚫 Run Experiment", disabled=True, help="Complete prerequisites first")

with col2:
    if can_run_experiment:
        experiment_hash = st.session_state.diffusion["experiment_manager"].generate_experiment_hash(
            st.session_state.diffusion["selected_model"],
            st.session_state.raw_data["data_file_path"],
            st.session_state.raw_data["date_index"]
        )
        experiment_exists = st.session_state.diffusion["experiment_manager"].check_experiment_exists(experiment_hash)
        
        if experiment_exists:
            st.info("💡 This will replace existing results")

with col3:
    if st.session_state.diffusion.get("experiment_running"):
        st.info("🔄 Experiment in progress...")

st.markdown("---")

# 4. 📊 Experiment Results
st.markdown("## 📊 Experiment Results")

# Get results to display
results_to_show = None

if model_selected and data_selected and stats_selected and sample_selected and st.session_state.diffusion["model_loaded"]:
    experiment_hash = st.session_state.diffusion["experiment_manager"].generate_experiment_hash(
        st.session_state.diffusion["selected_model"],
        st.session_state.raw_data["data_file_path"],
        st.session_state.raw_data["date_index"]
    )
    experiment_exists = st.session_state.diffusion["experiment_manager"].check_experiment_exists(experiment_hash)
    
    if experiment_exists:
        # Load existing experiment results
        try:
            predictions, metadata = st.session_state.diffusion["experiment_manager"].load_experiment(experiment_hash)
            if predictions and metadata:
                results_to_show = {
                    'u10_pred': predictions['u10_pred'],
                    'v10_pred': predictions['v10_pred'],
                    'u10_true': predictions['u10_true'],
                    'v10_true': predictions['v10_true'],
                    'metrics': metadata['metrics'],
                    'experiment_hash': experiment_hash,
                    'execution_time': metadata['experiment_info']['execution_time_seconds']
                }
                st.info("💡 **Results loaded from existing experiment**")
        except:
            pass
    elif st.session_state.diffusion.get("current_results"):
        # Show current session results
        results_to_show = st.session_state.diffusion["current_results"]

if results_to_show:
    display_experiment_results(results_to_show)
else:
    st.info("💡 **Run experiment to view results**")
    # Show placeholder metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall R²", "-")
    with col2:
        st.metric("Overall RMSE", "-")
    with col3:
        st.metric("U10 R²", "-")
    with col4:
        st.metric("V10 R²", "-")

st.markdown("---")

# 5. 📊 Interactive Visualizations
st.markdown("## 📊 Interactive Visualizations")

if results_to_show:
    display_visualizations(results_to_show)
else:
    st.info("💡 **Visualizations will appear here after running an experiment**")

st.markdown("---")

# 6. 📜 Experiments History
display_experiments_history()

# Sidebar controls
st.sidebar.markdown("### 🔧 Controls")

if st.sidebar.button("🔄 Reset All Selections", type="secondary"):
    reset_state_variables(page="diffusion")
    st.sidebar.success("✅ All selections reset!")
    st.rerun()

if st.sidebar.button("🔄 Refresh File Lists", type="secondary"):
    st.session_state.diffusion["model_files"] = list_mdlus_files(st.session_state.diffusion["model_dir_path"])
    st.sidebar.success("✅ File lists refreshed!")
    st.rerun()

# Debug section
with st.sidebar.expander("🔍 Debug Info"):
    debug_state()