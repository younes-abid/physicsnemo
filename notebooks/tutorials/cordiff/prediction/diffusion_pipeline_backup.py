"""
Diffusion Pipeline for CorrDiff Prediction

Handles diffusion model loading, testing, and inference operations.
"""

import os
import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from physicsnemo.models.diffusion.preconditioning import EDMPrecondSuperResolution
from physicsnemo.launch.utils.checkpoint import load_checkpoint
from physicsnemo.utils.diffusion import deterministic_sampler


class DiffusionPipeline:
    """Manages diffusion model operations for CorrDiff prediction pipeline."""
    
    def __init__(self, model_config: Dict[str, Any], checkpoint_config: Dict[str, Any]):
        """
        Initialize DiffusionPipeline.
        
        Args:
            model_config: Dictionary containing model architecture parameters
            checkpoint_config: Dictionary containing checkpoint paths and settings
        """
        self.model_config = model_config
        self.checkpoint_config = checkpoint_config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
        # EDM sampling parameters - THESE ARE CRITICAL!
        self.sampling_params = {
            "num_steps": 18,        # Number of denoising steps
            "sigma_min": 0.002,     # Minimum noise level  
            "sigma_max": 80.0,      # Maximum noise level
            "rho": 7.0,             # Time step exponent
            "solver": "heun",       # 2nd order solver
            "discretization": "edm", # EDM discretization
            "schedule": "linear",   # Linear noise schedule
            "scaling": "none"       # No signal scaling
        }
        
    def create_model(self) -> EDMPrecondSuperResolution:
        """Create the diffusion model."""
        print("=== Creating Diffusion Model ===")
        
        # Create the EDM Super-Resolution model
        self.model = EDMPrecondSuperResolution(
            img_resolution=432,
            img_in_channels=22,  # 16 + 2 + 4 grid channels (added internally)
            img_out_channels=2,  # U10, V10
            **self.model_config,
        )
        
        print(f"Diffusion model created with:")
        print(f"  - Input channels: 22 (16 variables + 2 conditioning + 4 grid)")
        print(f"  - Output channels: 2 (U10, V10)")
        print(f"  - Resolution: 432x432")
        print(f"  - Model channels: {self.model_config['model_channels']}")
        print(f"  - Channel multipliers: {self.model_config['channel_mult']}")
        
        return self.model
    
    def load_model(self) -> bool:
        """Load the diffusion model from checkpoint."""
        if self.model is None:
            self.create_model()
        
        print("=== Loading Diffusion Model ===")
        
        try:
            diffusion_epoch = load_checkpoint(
                path=self.checkpoint_config["checkpoint_dir"],
                models=self.model,
                epoch=self.checkpoint_config.get("checkpoint_index", None),
                device=self.device,
            )
            
            self.model.eval()
            self.model = self.model.to(self.device)
            self.model_loaded = True
            
            print(f"✓ Diffusion model loaded successfully from epoch {diffusion_epoch}")
            print(f"✓ Model set to evaluation mode")
            print(f"✓ Model moved to device: {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to load diffusion model: {e}")
            print(f"✗ Model will not be available for inference")
            self.model_loaded = False
            return False
        
        return True
    
    def inspect_model(self) -> None:
        """Inspect the diffusion model architecture and properties."""
        if not self.model_loaded:
            print("No diffusion model loaded to inspect")
            return
        
        print("=== Diffusion Model Architecture ===")
        print(f"Model type: {type(self.model).__name__}")
        print(f"Model device: {next(self.model.parameters()).device}")
        print(f"Model dtype: {next(self.model.parameters()).dtype}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Model memory usage
        if torch.cuda.is_available():
            model_memory = sum(p.numel() * p.element_size() for p in self.model.parameters()) / (1024**3)
            print(f"Model memory usage: {model_memory:.2f} GB")
    
    def test_on_dummy_data(self, batch_size: int = 1) -> bool:
        """Test the diffusion model on dummy data."""
        if not self.model_loaded:
            print("No diffusion model loaded to test")
            return False
        
        print("=== Testing Diffusion Model on Dummy Data ===")
        
        # Create test inputs with correct dimensions
        main_channels = 18  # 16 + 2
        hr_channels = 2     # U10/V10 output
        height = width = 432
        
        print(f"Creating dummy test tensors:")
        print(f"  - HR input: {batch_size} x {hr_channels} x {height} x {width}")
        print(f"  - LR input: {batch_size} x {main_channels} x {height} x {width}")
        print(f"  - Noise level: sigma = 1.0")
        
        # Create test tensors
        dummy_x = torch.randn(batch_size, hr_channels, height, width, device=self.device)
        dummy_img_lr = torch.randn(batch_size, main_channels, height, width, device=self.device)
        sigma = torch.tensor([1.0], device=self.device)
        
        # Test forward pass
        try:
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                # Forward pass
                output = self.model(x=dummy_x, img_lr=dummy_img_lr, sigma=sigma)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    print(f"✓ Forward pass successful! Inference time: {inference_time:.2f} ms")
                else:
                    print(f"✓ Forward pass successful!")
                
                print(f"✓ Output shape: {output.shape}")
                print(f"✓ Output dtype: {output.dtype}")
                print(f"✓ Output device: {output.device}")
                print(f"✓ Output statistics:")
                print(f"    - Mean: {output.mean().item():.6f}")
                print(f"    - Std: {output.std().item():.6f}")
                print(f"    - Min: {output.min().item():.6f}")
                print(f"    - Max: {output.max().item():.6f}")
                
                # Check for NaN or Inf
                if torch.isnan(output).any():
                    print(f"⚠ Warning: Output contains NaN values")
                    return False
                if torch.isinf(output).any():
                    print(f"⚠ Warning: Output contains infinite values")
                    return False
                
                print(f"✓ Dummy data test completed successfully!")
                return True
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def simulate_regression_output(self, ground_truth_data, noise_std: float = 0.1, 
                                 jitter_std: float = 0.05, kernel_size: int = 5) -> torch.Tensor:
        """
        Simulate regression output by degrading ground truth data.
        
        Args:
            ground_truth_data: xarray Dataset containing U10 and V10
            noise_std: Standard deviation of noise to add
            jitter_std: Standard deviation of jitter multiplier
            kernel_size: Size of smoothing kernel
            
        Returns:
            Simulated regression tensor of shape (1, 2, H, W)
        """
        print("=== Simulating Regression Output ===")
        
        # Extract data and convert to PyTorch tensors
        u10 = torch.tensor(ground_truth_data["U10"].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        v10 = torch.tensor(ground_truth_data["V10"].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Concatenate along the channel dimension
        data = torch.cat((u10, v10), dim=1)  # Shape: (1, 2, H, W)
        
        # Apply Gaussian smoothing
        kernel = torch.ones((2, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size ** 2)
        kernel = kernel.to(data.device)
        smoothed_data = F.conv2d(data, kernel, padding=kernel_size // 2, groups=2)
        
        # Add Gaussian noise
        noise = torch.randn_like(smoothed_data) * noise_std
        noisy_data = smoothed_data + noise
        
        # Apply jitter (pixel-wise random multiplier)
        jitter = 1 + torch.randn_like(noisy_data) * jitter_std
        simulated_data = noisy_data * jitter
        
        print(f"✓ Simulated regression output created")
        print(f"  - Shape: {simulated_data.shape}")
        print(f"  - Range: [{simulated_data.min():.3f}, {simulated_data.max():.3f}]")
        
        return simulated_data
    
    def predict(self, input_tensor: torch.Tensor, regression_output: torch.Tensor, 
               data_manager, num_iterations: int = 1) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Run CORRECTED diffusion prediction using proper EDM sampling.
        
        CRITICAL: This now uses the proper EDM sampler with multiple denoising steps,
        not just a single forward pass!
        
        Args:
            input_tensor: Normalized input tensor of shape (1, 16, H, W)
            regression_output: NORMALIZED output from regression model of shape (1, 2, H, W)
            data_manager: DataManager instance for statistics
            num_iterations: Number of diffusion iterations to run (typically 1 with proper sampling)
            
        Returns:
            Tuple of (final_hr_output_normalized, u10_denorm, v10_denorm)
        """
        if not self.model_loaded:
            raise ValueError("No diffusion model loaded. Call load_model() first.")
        
        print(f"=== Running CORRECTED Diffusion Prediction with Proper EDM Sampling ===")
        print(f"CRITICAL FIX: Using {self.sampling_params['num_steps']} denoising steps instead of 1!")
        print(f"Noise schedule: σ_min={self.sampling_params['sigma_min']}, σ_max={self.sampling_params['sigma_max']}")
        print(f"Solver: {self.sampling_params['solver']}, discretization: {self.sampling_params['discretization']}")
        
        try:
            with torch.no_grad():
                # Move inputs to device
                input_tensor = input_tensor.to(self.device)
                regression_output = regression_output.to(self.device)
                
                # Prepare conditioning (2 channels - means of U10/V10 in normalized space = 0)
                conditioning = torch.zeros(1, 2, 432, 432, device=self.device)
                
                # Combine inputs (16 vars + 2 conditioning = 18 channels)
                # Model will add 4 grid channels internally
                low_res_input = torch.cat([input_tensor, conditioning], dim=1)
                
                print(f"✓ Input preparation completed")
                print(f"  - LR input shape: {low_res_input.shape}")
                print(f"  - Regression output (normalized) shape: {regression_output.shape}")
                print(f"  - Regression output range: [{regression_output.min():.3f}, {regression_output.max():.3f}]")
                
                # CRITICAL FIX: Use proper EDM sampling instead of single forward pass
                # Initialize with regression output (not random noise)
                latents = regression_output.clone()
                
                # Timing
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                # Run proper EDM sampling - THIS IS THE KEY FIX!
                print(f"  - Running EDM sampling with {self.sampling_params['num_steps']} steps...")
                final_hr_output_normalized = deterministic_sampler(
                    net=self.model,
                    latents=latents,
                    img_lr=low_res_input,
                    class_labels=None,
                    **self.sampling_params
                )
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    print(f"✓ CORRECTED EDM sampling time: {inference_time:.2f} ms")
                
                print(f"✓ Final HR output shape: {final_hr_output_normalized.shape}")
                print(f"✓ Final HR output range (normalized): [{final_hr_output_normalized.min():.3f}, {final_hr_output_normalized.max():.3f}]")
                
                # Denormalize final predictions for visualization and metrics
                u10_pred_norm = final_hr_output_normalized[0, 0].cpu().numpy()
                v10_pred_norm = final_hr_output_normalized[0, 1].cpu().numpy()
                
                u10_denorm = data_manager.denormalize_output(u10_pred_norm, "U10")
                v10_denorm = data_manager.denormalize_output(v10_pred_norm, "V10")
                
                print(f"✓ Final predictions denormalized")
                print(f"  - U10 range (denormalized): [{u10_denorm.min():.3f}, {u10_denorm.max():.3f}]")
                print(f"  - V10 range (denormalized): [{v10_denorm.min():.3f}, {v10_denorm.max():.3f}]")
                
                print(f"🎉 CRITICAL FIX APPLIED: Using proper EDM sampling instead of single forward pass!")
                
                return final_hr_output_normalized, u10_denorm, v10_denorm
                
        except Exception as e:
            print(f"✗ Diffusion prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the diffusion model."""
        if not self.model_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "architecture": "EDMPrecondSuperResolution",
            "input_channels": 22,
            "output_channels": 2,
            "resolution": "432x432",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "model_channels": self.model_config.get("model_channels", "unknown"),
            "channel_mult": self.model_config.get("channel_mult", "unknown"),
            "sigma_min": self.model_config.get("sigma_min", "unknown"),
            "sigma_max": self.model_config.get("sigma_max", "unknown"),
        }