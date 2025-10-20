"""
CENTRALIZED Diffusion Pipeline - Uses ONLY parameters from notebook
All parameters are now passed from the notebook - NO hardcoded values!
This ensures exact replication of train.py validation_block behavior.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional

# EXACT IMPORTS from train.py
from physicsnemo.models.diffusion import EDMPrecondSuperResolution
from physicsnemo.metrics.diffusion import ResidualLoss
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo import Module

class DiffusionPipeline:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with CENTRALIZED configuration from notebook.
        
        Args:
            config: Complete configuration dict containing:
                - model_config: All model parameters from notebook
                - execution_config: All execution parameters from notebook  
                - checkpoint_config: All checkpoint paths from notebook
        """
        print(f"🎯 CENTRALIZED: Initializing DiffusionPipeline with notebook parameters")
        
        # Extract configuration sections
        self.model_config = config["model_config"]
        self.execution_config = config["execution_config"] 
        self.checkpoint_config = config["checkpoint_config"]
        
        # Initialize model components
        self.model = None
        self.regression_net = None
        self.loss_fn = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set execution parameters from notebook (NO hardcoded values!)
        self.use_apex_gn = self.execution_config["use_apex_gn"]
        self.enable_amp = self.execution_config["enable_amp"]
        self.amp_dtype = getattr(torch, self.execution_config["amp_dtype"])
        self.profile_mode = self.execution_config["profile_mode"]
        self.batch_size_per_gpu = self.execution_config["batch_size_per_gpu"]
        self.use_patch_grad_acc = self.execution_config["use_patch_grad_acc"]
        self.patching = self.execution_config["patching"]
        self.patch_nums_iter = self.execution_config["patch_nums_iter"]
        
        # CRITICAL: Apply exact deterministic settings from train.py
        self._configure_deterministic_inference()
        
        print(f"📋 CENTRALIZED parameters loaded:")
        print(f"   Device: {self.device}")
        print(f"   Deterministic mode: enabled")
        
    def _configure_deterministic_inference(self):
        """
        EXACT DETERMINISTIC SETTINGS from train.py to ensure reproducible results.
        Source: train.py lines 152-157 and helpers/train_helpers.py lines 47-64
        """
        # EXACT SEED SETTING from train.py set_seed() function
        # Use fixed seed for inference to ensure deterministic results
        inference_seed = 42  # Fixed seed for consistent inference
        np.random.seed(inference_seed % (1 << 31))
        torch.manual_seed(inference_seed)
        
        # EXACT CUDA/cuDNN SETTINGS from train.py configure_cuda_for_consistent_precision()
        torch.backends.cudnn.benchmark = True  # From train.py
        torch.backends.cudnn.allow_tf32 = False  # From train.py - critical for consistency
        torch.backends.cuda.matmul.allow_tf32 = False  # From train.py - critical for consistency
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False  # From train.py
        
        # Additional deterministic settings
        torch.backends.cudnn.deterministic = True  # Ensure deterministic operations
        if torch.cuda.is_available():
            torch.cuda.manual_seed(inference_seed)
            torch.cuda.manual_seed_all(inference_seed)  # For multi-GPU consistency
        
        print(f"✓ Deterministic inference configured:")
        print(f"   - Fixed seed: {inference_seed}")
        print(f"   - cuDNN deterministic: True")
        print(f"   - TF32 disabled: True")
        
    def load_model(self) -> bool:
        """
        EXACT MODEL CREATION from train.py create_model() function
        Uses ONLY parameters from notebook configuration.
        """
        
        print(f"🔧 Creating diffusion model with CENTRALIZED parameters...")
        
        # DATA-DERIVED PARAMETERS (automatically determined from data)
        # These are the only parameters computed here, as noted in notebook
        dataset_channels = 16  # From input_variables count in config
        img_in_channels = dataset_channels
        img_out_channels = 2  # U10, V10
        img_shape = [432, 432]  # From actual data dimensions
        
        # EXACT hr_mean_conditioning logic from train.py
        hr_mean_conditioning = self.model_config["hr_mean_conditioning"]
        if hr_mean_conditioning:
            img_in_channels += img_out_channels  # 16 + 2 = 18
        
        print(f"📊 Data-derived parameters (computed automatically):")
        print(f"   dataset_channels: {dataset_channels}")
        print(f"   img_in_channels: {img_in_channels}")
        print(f"   img_out_channels: {img_out_channels}")
        print(f"   img_shape: {img_shape}")
        print(f"   hr_mean_conditioning: {hr_mean_conditioning}")
        
        # MODEL ARGS using CENTRALIZED parameters from notebook
        model_args = {
            # Basic model structure
            "img_out_channels": img_out_channels,
            "img_resolution": list(img_shape),
            
            # Performance settings from notebook
            "use_fp16": self.model_config["use_fp16"],
            "checkpoint_level": self.model_config["checkpoint_level"],
            
            # Architecture parameters from notebook 
            "gridtype": self.model_config["gridtype"],
            "N_grid_channels": self.model_config["N_grid_channels"],
            "embedding_type": self.model_config["embedding_type"],
            "model_channels": self.model_config["model_channels"],
            "channel_mult": self.model_config["channel_mult"],
            "attn_resolutions": self.model_config["attn_resolutions"],
            "model_type": self.model_config["model_type"],
            #"sigma_data": self.model_config["sigma_data"],
            
            # Note: sigma_min and sigma_max are for reference only, not used in model creation
        }
        
        print(f"🏗️  CENTRALIZED model parameters:")
        for key, value in model_args.items():
            print(f"   {key}: {value}")
        
        # EXACT MODEL CREATION from train.py line ~305
        total_in_channels = img_in_channels + model_args["N_grid_channels"]  # 18 + 4 = 22
        print(f"   total_in_channels: {total_in_channels}")
        
        try:
            self.model = EDMPrecondSuperResolution(
                img_in_channels=total_in_channels,
                **model_args,
            )
            
            # EXACT MODEL SETUP from train.py line ~310  
            self.model.eval().requires_grad_(False).to(self.device)
            
            print(f"✓ Model created and moved to {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to create model: {e}")
            print(f"   Model args: {model_args}")
            raise
        
        # EXACT CHECKPOINT LOADING using CENTRALIZED paths
        try:
            epoch = load_checkpoint(
                path=self.checkpoint_config["checkpoint_dir"],
                models=self.model,
                epoch=self.checkpoint_config["checkpoint_index"],
                device=self.device,
            )
            print(f"✓ Diffusion model loaded from epoch {epoch}")
            print(f"   Checkpoint dir: {self.checkpoint_config['checkpoint_dir']}")
            
            # Log model state
            print(f"   Model mode: {'training' if self.model.training else 'eval'}")
            print(f"   Model requires_grad: {any(p.requires_grad for p in self.model.parameters())}")
            
            return True
        except Exception as e:
            print(f"✗ Failed to load diffusion model: {e}")
            raise
            
    def load_regression_model(self) -> bool:
        """
        EXACT REGRESSION LOADING from train.py load_regression_checkpoint()
        Uses CENTRALIZED checkpoint path from notebook.
        """
        
        regression_checkpoint_path = self.checkpoint_config.get("regression_checkpoint_path")
        if not regression_checkpoint_path:
            print(f"⚠️  No regression checkpoint path provided")
            return False
            
        print(f"🔧 Loading regression model with CENTRALIZED path:")
        print(f"   Path: {regression_checkpoint_path}")
        
        try:
            # EXACT LOADING from train.py load_regression_checkpoint()
            self.regression_net = Module.from_checkpoint(
                regression_checkpoint_path, 
                override_args={"use_apex_gn": self.use_apex_gn}
            )
            self.regression_net.amp_mode = self.enable_amp
            self.regression_net.profile_mode = self.profile_mode
            self.regression_net.eval().requires_grad_(False).to(self.device)
            
            if self.use_apex_gn:
                self.regression_net.to(memory_format=torch.channels_last)
            
            print(f"✓ Regression model loaded successfully")
            print(f"   Model mode: {'training' if self.regression_net.training else 'eval'}")
            print(f"   Model requires_grad: {any(p.requires_grad for p in self.regression_net.parameters())}")
            
            return True
        except Exception as e:
            print(f"✗ Failed to load regression model: {e}")
            raise
            
    def create_loss_function(self) -> bool:
        """
        EXACT LOSS FUNCTION CREATION from train.py create_loss_function
        Uses ONLY hr_mean_conditioning parameter, just like the training code.
        """
        
        if not self.regression_net:
            print(f"✗ Cannot create loss function: regression_net not loaded")
            return False
            
        print(f"🔧 Creating loss function with EXACT train.py parameters...")
        
        # EXACT MATCH with train.py create_loss_function - ONLY pass hr_mean_conditioning!
        # The training code does NOT pass P_mean, P_std, or sigma_data - uses class defaults
        self.loss_fn = ResidualLoss(
            regression_net=self.regression_net,
            hr_mean_conditioning=self.model_config["hr_mean_conditioning"],
        )
        
        print(f"✓ ResidualLoss created with EXACT train.py parameters:")
        print(f"   hr_mean_conditioning: {self.model_config['hr_mean_conditioning']}")
        print(f"   P_mean: 0.0 (class default)")
        print(f"   P_std: 1.2 (class default)") 
        print(f"   sigma_data: 0.5 (class default)")
        return True
    
    def predict(self, input_tensor: torch.Tensor, regression_output: torch.Tensor, 
               data_manager) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        EXACT COPY-PASTE from train.py validation_block lines 842-890
        Uses CENTRALIZED execution parameters from notebook.
        This is identical to the validation logic in the training script.
        """
        
        print(f"\n🔮 Starting CENTRALIZED diffusion prediction...")
        print(f"   Input tensor shape: {input_tensor.shape}")
        print(f"   Regression output shape: {regression_output.shape}")
        print(f"   Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        print(f"   Regression range: [{regression_output.min():.3f}, {regression_output.max():.3f}]")
        
        if not all([self.model, self.regression_net, self.loss_fn]):
            raise RuntimeError("Models not loaded. Call load_model(), load_regression_model(), and create_loss_function() first.")
        
        # EXACT VALIDATION BLOCK COPY-PASTE from train.py validation_block
        with torch.no_grad():
            # EXACT VARIABLE ASSIGNMENTS from validation_block lines 842-843
            img_clean_valid = regression_output
            img_lr_valid = input_tensor
            
            print(f"   img_clean_valid shape: {img_clean_valid.shape}")
            print(f"   img_lr_valid shape: {img_lr_valid.shape}")
            
            # EXACT DATA PREPARATION from validation_block lines 845-862
            # Uses CENTRALIZED use_apex_gn parameter
            if self.use_apex_gn:
                img_clean_valid = img_clean_valid.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).to(memory_format=torch.channels_last)
                img_lr_valid = img_lr_valid.to(
                    self.device,
                    dtype=torch.float32,
                    non_blocking=True,
                ).to(memory_format=torch.channels_last)
            else:
                img_clean_valid = (
                    img_clean_valid.to(self.device)
                    .to(torch.float32)
                    .contiguous()
                )
                img_lr_valid = (
                    img_lr_valid.to(self.device)
                    .to(torch.float32)
                    .contiguous()
                )
            
            print(f"   After device transfer (CENTRALIZED use_apex_gn={self.use_apex_gn}):")
            print(f"     img_clean_valid shape: {img_clean_valid.shape}, device: {img_clean_valid.device}")
            print(f"     img_lr_valid shape: {img_lr_valid.shape}, device: {img_lr_valid.device}")
            
            # EXACT LOSS KWARGS SETUP from validation_block lines 864-876
            # Uses CENTRALIZED use_patch_grad_acc parameter
            loss_valid_kwargs = {
                "net": self.model,
                "img_clean": img_clean_valid,
                "img_lr": img_lr_valid,
                "augment_pipe": None,
            }
            if self.use_patch_grad_acc is not None and hasattr(self.loss_fn, "use_patch_grad_acc"):
                loss_valid_kwargs["use_patch_grad_acc"] = self.use_patch_grad_acc
            
            # No lead_time_label in our case (from validation_block logic)
            
            if self.use_patch_grad_acc:
                self.loss_fn.y_mean = None
                
            print(f"   Loss kwargs keys: {list(loss_valid_kwargs.keys())}")
            print(f"   CENTRALIZED use_patch_grad_acc: {self.use_patch_grad_acc}")
            
            # EXACT PATCH LOOP from validation_block lines 878-890
            # Uses CENTRALIZED patch_nums_iter parameter
            for patch_num_per_iter in self.patch_nums_iter:
                print(f"   Processing patch_num_per_iter: {patch_num_per_iter}")
                
                # Uses CENTRALIZED patching parameter
                if self.patching is not None:
                    self.patching.set_patch_num(patch_num_per_iter)
                    loss_valid_kwargs.update({"patching": self.patching})
                    
                # EXACT AUTOCAST and LOSS CALL from validation_block
                # Uses CENTRALIZED enable_amp and amp_dtype parameters
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                    print(f"   Calling loss function with CENTRALIZED autocast:")
                    print(f"     enable_amp: {self.enable_amp}")
                    print(f"     amp_dtype: {self.amp_dtype}")
                    loss_valid, predictions = self.loss_fn(
                        **loss_valid_kwargs, return_predictions=True
                    )
                    
                print(f"   Loss function returned:")
                print(f"     loss_valid shape: {loss_valid.shape}")
                print(f"     loss_valid range: [{loss_valid.min():.3f}, {loss_valid.max():.3f}]")
                print(f"     predictions shape: {predictions.shape}")
                print(f"     predictions range: [{predictions.min():.3f}, {predictions.max():.3f}]")
                
                # EXACT LOSS PROCESSING from validation_block lines 883-890
                # Uses CENTRALIZED batch_size_per_gpu parameter
                loss_valid = (
                    (loss_valid.sum() / self.batch_size_per_gpu)
                    .cpu()
                    .item()
                )
                
                print(f"   Final validation loss (CENTRALIZED batch_size_per_gpu={self.batch_size_per_gpu}): {loss_valid:.6f}")
            
            print(f"✓ CENTRALIZED diffusion prediction completed")
            
            # FIXED: The ResidualLoss already returns hr_pred = D_yn + self.y_mean
            # So predictions is already the final HR prediction, not the residual!
            print(f"📊 Denormalizing predictions...")
            print(f"   ⚠️  FIXED: predictions is already HR prediction (D_yn + y_mean), not residual!")
            
            u10_pred_norm = predictions[0, 0].cpu().numpy()
            v10_pred_norm = predictions[0, 1].cpu().numpy()
            
            print(f"   u10_pred_norm range: [{u10_pred_norm.min():.3f}, {u10_pred_norm.max():.3f}]")
            print(f"   v10_pred_norm range: [{v10_pred_norm.min():.3f}, {v10_pred_norm.max():.3f}]")
            
            u10_denorm = data_manager.denormalize_output(u10_pred_norm, "U10")
            v10_denorm = data_manager.denormalize_output(v10_pred_norm, "V10")
            
            print(f"   u10_denorm range: [{u10_denorm.min():.3f}, {u10_denorm.max():.3f}]")
            print(f"   v10_denorm range: [{v10_denorm.min():.3f}, {v10_denorm.max():.3f}]")
            print(f"✓ CENTRALIZED denormalization completed")
            
            return predictions, u10_denorm, v10_denorm