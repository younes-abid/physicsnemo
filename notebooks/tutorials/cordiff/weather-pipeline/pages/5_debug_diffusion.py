import streamlit as st
import os
import traceback
from prediction import DiffusionPipeline

# Page configuration
st.set_page_config(
    page_title="Weather Pipeline - Diffusion Debug",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ Weather Diffusion Pipeline - DEBUG MODE")
st.markdown("Minimal debug page with hardcoded parameters to test diffusion loading")

st.markdown("---")

# Hardcoded configuration - EXACT COPY from notebook
OUTPUT_DIR = "/app/"

# DIFFUSION CONFIGURATION (EXACT COPY from train.py validation logic)
diffusion_config = {
    "model_config": {
        # Model architecture from diffusion_normal.yaml (EXACT MATCH)
        "gridtype": "sinusoidal",
        "N_grid_channels": 4,
        "embedding_type": "zero",
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
        # REMOVED: regression_checkpoint_path - ResidualLoss will load it internally!
    },
    # NEW: Add the regression path that ResidualLoss needs internally
    "training_config": {
        "regression_checkpoint_path": OUTPUT_DIR + "checkpoints_regression/UNet.0.535008.mdlus",
    }
}

st.markdown("### Hardcoded Configuration")
st.json(diffusion_config)

st.markdown("---")

# Single button to test diffusion loading
if st.button("🔄 Load Diffusion Model", type="primary"):
    try:
        with st.spinner("Loading diffusion model..."):
            st.info("Creating DiffusionPipeline with hardcoded config...")
            
            # EXACT COPY from notebook
            diffusion_pipeline = DiffusionPipeline(diffusion_config)
            
            st.info("Loading diffusion model...")
            # EXACT COPY of train.py - ResidualLoss handles regression internally!
            diffusion_loaded = diffusion_pipeline.load_model()
            
            if diffusion_loaded:
                st.success("✓ Diffusion model loaded successfully")
                
                st.info("Creating loss function...")
                loss_fn_created = diffusion_pipeline.create_loss_function()
                
                if loss_fn_created:
                    st.success("✓ All diffusion components loaded EXACTLY like train.py")
                    st.balloons()
                else:
                    st.error("❌ Failed to create loss function")
            else:
                st.error("❌ Failed to load diffusion model")
                
    except Exception as e:
        error_msg = f"Error loading diffusion model: {str(e)}"
        st.error(f"❌ {error_msg}")
        st.error(f"Full error traceback: {traceback.format_exc()}")

st.markdown("---")
st.markdown("**Debug Info:**")
st.code(f"""
Diffusion checkpoint dir: {diffusion_config['checkpoint_config']['checkpoint_dir']}
Diffusion checkpoint index: {diffusion_config['checkpoint_config']['checkpoint_index']}
Regression checkpoint path: {diffusion_config['training_config']['regression_checkpoint_path']}
""")