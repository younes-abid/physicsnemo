"""
SAR2Height Diffusion Pipeline

EXACT COPY of train.py validation block adapted for SAR2Height
This replicates the exact validation logic from train.py using EDMPrecondSuperResolution and ResidualLoss
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

# EXACT IMPORTS from train.py
from physicsnemo.models.diffusion import EDMPrecondSuperResolution
from physicsnemo.metrics.diffusion import ResidualLoss
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo import Module
import logging

logger = logging.getLogger(__name__)


class SAR2HeightDiffusionPipeline:
    """
    SAR2Height Diffusion Pipeline using EXACT train.py approach.
    Uses EDMPrecondSuperResolution and ResidualLoss exactly like training.
    """
    
    def __init__(self, model_config: Dict[str, Any], checkpoint_path: str):
        """
        Initialize with EXACT train.py configuration for SAR2Height.
        
        Args:
            model_config: Dictionary containing model architecture parameters
            checkpoint_path: Full path to the diffusion checkpoint file
        """
        print(f"🎯 EXACT COPY: Initializing SAR2HeightDiffusionPipeline like train.py")
        
        # Store configuration
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
        
        # Initialize model components
        self.model = None
        self.loss_fn = None  # ResidualLoss will handle regression internally!
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SAR2Height specific parameters
        self.dataset_channels = None  # Will be set based on input variables
        self.img_out_channels = 1  # DSM height (single channel)
        self.img_shape = [432, 432]  # SAR2Height patch size
        
        # Set execution parameters (using defaults similar to weather pipeline)
        self.use_apex_gn = False
        self.enable_amp = False
        self.amp_dtype = torch.float32
        self.profile_mode = False
        self.batch_size_per_gpu = 1
        self.use_patch_grad_acc = None
        self.patching = None
        self.patch_nums_iter = [1]
        
        # CRITICAL: Apply exact deterministic settings from train.py
        self._configure_deterministic_inference()
    
    def _configure_deterministic_inference(self):
        """EXACT DETERMINISTIC SETTINGS from train.py"""
        inference_seed = 42
        np.random.seed(inference_seed % (1 << 31))
        torch.manual_seed(inference_seed)
        
        # EXACT CUDA/cuDNN SETTINGS from train.py
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.deterministic = True
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(inference_seed)
            torch.cuda.manual_seed_all(inference_seed)
        
        print(f"✓ Deterministic inference configured (seed: {inference_seed})")
    
    def load_model(self, n_input_channels: int) -> bool:
        """EXACT MODEL CREATION from train.py create_model() adapted for SAR2Height"""
        print(f"🔧 Creating SAR2Height diffusion model EXACTLY like train.py...")
        
        # DATA-DERIVED PARAMETERS for SAR2Height
        self.dataset_channels = n_input_channels  # input_variables count (e.g., 2 for intensity_db + intensity_percentile_rescaled)
        img_in_channels = self.dataset_channels
        img_out_channels = self.img_out_channels  # 1 for DSM height
        img_shape = self.img_shape
        
        # EXACT hr_mean_conditioning logic from train.py
        hr_mean_conditioning = self.model_config.get("hr_mean_conditioning", True)
        if hr_mean_conditioning:
            img_in_channels += img_out_channels  # e.g., 2 + 1 = 3
        
        # MODEL ARGS using EXACT parameters from train.py
        model_args = {
            "img_out_channels": img_out_channels,
            "img_resolution": list(img_shape),
            "use_fp16": self.model_config.get("use_fp16", False),
            "checkpoint_level": self.model_config.get("checkpoint_level", 0),
            "gridtype": self.model_config.get("gridtype", "sinusoidal"),
            "N_grid_channels": self.model_config.get("N_grid_channels", 4),
            "embedding_type": self.model_config.get("embedding_type", "zero"),
            "model_channels": self.model_config.get("model_channels", 128),
            "channel_mult": self.model_config.get("channel_mult", [1, 2, 2, 2, 2]),
            "attn_resolutions": self.model_config.get("attn_resolutions", [28]),
            "model_type": self.model_config.get("model_type", "SongUNetPosEmbd"),
        }
        
        # EXACT MODEL CREATION from train.py
        total_in_channels = img_in_channels + model_args["N_grid_channels"]  # e.g., 3 + 4 = 7
        
        try:
            self.model = EDMPrecondSuperResolution(
                img_in_channels=total_in_channels,
                **model_args,
            )
            
            # EXACT MODEL SETUP from train.py
            self.model.eval().requires_grad_(False).to(self.device)
            
            print(f"✓ SAR2Height Model created: total_in_channels={total_in_channels}")
            print(f"   Dataset channels: {self.dataset_channels}")
            print(f"   HR mean conditioning: {hr_mean_conditioning}")
            print(f"   Grid channels: {model_args['N_grid_channels']}")
            print(f"   Output channels: {img_out_channels}")
            
        except Exception as e:
            print(f"✗ Failed to create SAR2Height model: {e}")
            raise
        
        # EXACT CHECKPOINT LOADING
        try:
            # Extract directory and epoch from full path
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            checkpoint_filename = os.path.basename(self.checkpoint_path)
            
            # Extract epoch from filename (e.g., "EDMPrecondSuperResolution.0.45008.mdlus")
            parts = checkpoint_filename.split('.')
            if len(parts) >= 3:
                epoch = int(parts[-2])  # Second to last part before extension
            else:
                epoch = None
            
            loaded_epoch = load_checkpoint(
                path=checkpoint_dir,
                models=self.model,
                epoch=epoch,
                device=self.device,
            )
            print(f"✓ SAR2Height Diffusion model loaded from epoch {loaded_epoch}")
            return True
            
        except Exception as e:
            print(f"✗ Failed to load SAR2Height diffusion model: {e}")
            raise
    
    def create_loss_function(self, regression_checkpoint_path: str) -> bool:
        """
        EXACT LOSS FUNCTION CREATION from train.py.
        ResidualLoss will internally load the regression network!
        
        Args:
            regression_checkpoint_path: Path to the regression model checkpoint
        """
        print(f"🔧 Creating ResidualLoss for SAR2Height EXACTLY like train.py...")
        
        # STEP 1: Load regression network EXACTLY like train.py load_regression_checkpoint()
        print(f"   Loading SAR2Height regression network from: {regression_checkpoint_path}")
        
        try:
            regression_net = Module.from_checkpoint(
                regression_checkpoint_path, 
                override_args={"use_apex_gn": self.use_apex_gn}
            )
            regression_net.amp_mode = self.enable_amp
            regression_net.profile_mode = self.profile_mode
            regression_net.eval().requires_grad_(False).to(self.device)
            
            if self.use_apex_gn:
                regression_net.to(memory_format=torch.channels_last)
            
            print(f"✓ SAR2Height regression network loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load SAR2Height regression network: {e}")
            raise
        
        # STEP 2: Create ResidualLoss EXACTLY like train.py create_loss_function()
        hr_mean_conditioning = self.model_config.get("hr_mean_conditioning", True)
        
        self.loss_fn = ResidualLoss(
            regression_net=regression_net,
            hr_mean_conditioning=hr_mean_conditioning,
        )
        
        print(f"✓ SAR2Height ResidualLoss created with:")
        print(f"   hr_mean_conditioning: {hr_mean_conditioning}")
        print(f"   P_mean: 0.0 (class default)")
        print(f"   P_std: 1.2 (class default)") 
        print(f"   sigma_data: 0.5 (class default)")
        return True
    
    def predict(self, input_tensor: torch.Tensor, regression_output: torch.Tensor, 
               data_manager) -> Tuple[torch.Tensor, np.ndarray]:
        """
        EXACT COPY-PASTE from train.py validation_block adapted for SAR2Height
        
        Args:
            input_tensor: Normalized input tensor of shape (1, N_channels, H, W)
            regression_output: Regression output (not used for ground truth)
            data_manager: SAR2HeightDataManager instance for ground truth and denormalization
            
        Returns:
            Tuple of (predictions_tensor, denormalized_prediction)
        """
        
        print(f"\n🔮 Starting EXACT COPY validation from train.py for SAR2Height...")
        
        if not all([self.model, self.loss_fn]):
            raise RuntimeError("Models not loaded. Call load_model() and create_loss_function() first.")
        
        # CRITICAL FIX: Get ground truth using the correct DataManager method
        # In training validation, img_clean is the GROUND TRUTH TARGET!
        ground_truth = data_manager.get_ground_truth()
        
        # Normalize ground truth using the same method as the DataManager
        ground_truth_norm = data_manager.normalize_output(ground_truth)
        
        # Convert to tensor format for SAR2Height (single channel)
        img_clean_valid = torch.from_numpy(ground_truth_norm).float().unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 432, 432]
        
        print(f"   FIXED: Using ground truth as img_clean (not regression output)")
        print(f"   img_clean_valid (ground truth) range: [{img_clean_valid.min():.3f}, {img_clean_valid.max():.3f}]")
        
        # EXACT VALIDATION BLOCK COPY-PASTE from train.py validation_block
        with torch.no_grad():
            # EXACT VARIABLE ASSIGNMENTS from validation_block lines 842-843
            # img_clean_valid = already set to ground truth above
            img_lr_valid = input_tensor
            
            print(f"   img_clean_valid shape: {img_clean_valid.shape}")
            print(f"   img_lr_valid shape: {img_lr_valid.shape}")
            
            # EXACT DATA PREPARATION from validation_block lines 845-862
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
            
            # EXACT LOSS KWARGS SETUP from validation_block lines 864-876
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
                
            # EXACT PATCH LOOP from validation_block lines 878-890
            for patch_num_per_iter in self.patch_nums_iter:
                
                # Uses patching parameter
                if self.patching is not None:
                    self.patching.set_patch_num(patch_num_per_iter)
                    loss_valid_kwargs.update({"patching": self.patching})
                    
                # EXACT AUTOCAST and LOSS CALL from validation_block
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.enable_amp):
                    loss_valid, predictions = self.loss_fn(
                        **loss_valid_kwargs, return_predictions=True
                    )
                    
                # EXACT LOSS PROCESSING from validation_block lines 883-890
                loss_valid = (
                    (loss_valid.sum() / self.batch_size_per_gpu)
                    .cpu()
                    .item()
                )
                
                print(f"   Final validation loss: {loss_valid:.6f}")
            
            print(f"✓ EXACT COPY validation completed for SAR2Height")
            
            # Denormalize predictions (predictions is already HR prediction!)
            height_pred_norm = predictions[0, 0].cpu().numpy()
            height_denorm = data_manager.denormalize_output(height_pred_norm)
            
            return predictions, height_denorm
    
    def predict_ensemble(self, input_tensor: torch.Tensor, regression_output: torch.Tensor,
                        data_manager, seeds: List[int], 
                        sampling_steps: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Run ensemble diffusion prediction with multiple seeds for SAR2Height.
        
        Args:
            input_tensor: Normalized input tensor of shape (1, N_channels, H, W)
            regression_output: Regression output tensor
            data_manager: SAR2HeightDataManager instance for denormalization
            seeds: List of random seeds for ensemble generation
            sampling_steps: Number of diffusion sampling steps (not used in exact validation)
            
        Returns:
            Tuple of (normalized_predictions, denormalized_predictions) lists
        """
        if not all([self.model, self.loss_fn]):
            raise ValueError("No diffusion model loaded. Call load_model() and create_loss_function() first.")
        
        print(f"=== Running SAR2Height Diffusion Ensemble Prediction ===")
        print(f"  - Ensemble size: {len(seeds)}")
        print(f"  - Seeds: {seeds}")
        
        normalized_predictions = []
        denormalized_predictions = []
        
        try:
            for i, seed in enumerate(seeds):
                print(f"  → Generating prediction {i+1}/{len(seeds)} with seed {seed}")
                
                # Set seed for reproducibility
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                
                # Run single prediction using the exact validation approach
                predictions_tensor, height_denorm = self.predict(
                    input_tensor, regression_output, data_manager
                )
                
                # Store results
                height_pred_norm = predictions_tensor[0, 0].cpu().numpy()
                normalized_predictions.append(height_pred_norm)
                denormalized_predictions.append(height_denorm)
            
            print(f"✅ SAR2Height ensemble prediction completed successfully!")
            return normalized_predictions, denormalized_predictions
            
        except Exception as e:
            print(f"❌ SAR2Height ensemble prediction failed: {e}")
            raise
    
    def generate_ensemble_seeds(self, ensemble_size: int, base_seed: int = 42) -> List[int]:
        """Generate reproducible seeds for ensemble prediction."""
        np.random.seed(base_seed)
        seeds = [base_seed + i for i in range(ensemble_size)]
        print(f"Generated {ensemble_size} ensemble seeds starting from {base_seed}")
        return seeds
    
    def set_sampling_steps(self, steps: int):
        """Set sampling steps (for compatibility - not used in exact validation approach)."""
        print(f"Note: Using exact validation approach - sampling steps parameter ignored")
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded SAR2Height diffusion model."""
        if not self.model:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "loaded": True,
            "architecture": "EDMPrecondSuperResolution",
            "dataset_channels": self.dataset_channels,
            "input_channels": self.dataset_channels + self.img_out_channels + 4,  # lr + hr_conditioning + grid
            "output_channels": self.img_out_channels,
            "resolution": "432x432",
            "parameters": total_params,
            "device": str(self.device),
            "model_channels": self.model_config.get("model_channels", "unknown"),
            "channel_mult": self.model_config.get("channel_mult", "unknown"),
            "hr_mean_conditioning": self.model_config.get("hr_mean_conditioning", True),
            "loss_function_loaded": self.loss_fn is not None,
            "checkpoint_path": self.checkpoint_path
        }