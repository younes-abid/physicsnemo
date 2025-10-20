"""
Regression Pipeline for CorrDiff Prediction

Handles regression model loading, testing, and inference operations.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from physicsnemo.models.diffusion.unet import UNet
from physicsnemo.launch.utils.checkpoint import load_checkpoint


class RegressionPipeline:
    """Manages regression model operations for CorrDiff prediction pipeline."""
    
    def __init__(self, model_config: Dict[str, Any], checkpoint_config: Dict[str, Any]):
        """
        Initialize RegressionPipeline.
        
        Args:
            model_config: Dictionary containing model architecture parameters
            checkpoint_config: Dictionary containing checkpoint paths and settings
        """
        self.model_config = model_config
        self.checkpoint_config = checkpoint_config
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        
    def create_model(self) -> UNet:
        """Create the regression UNet model."""
        print("=== Creating Regression Model ===")
        
        # Create regression model
        self.model = UNet(
            img_in_channels=20,  # 16 variables + 4 grid channels
            img_out_channels=2,  # U10, V10
            img_resolution=432,
            **self.model_config,
        )
        
        print(f"Regression model created with:")
        print(f"  - Input channels: 20 (16 variables + 4 grid)")
        print(f"  - Output channels: 2 (U10, V10)")
        print(f"  - Resolution: 432x432")
        print(f"  - Model channels: {self.model_config['model_channels']}")
        print(f"  - Channel multipliers: {self.model_config['channel_mult']}")
        
        return self.model
    
    def load_model(self) -> bool:
        """Load the regression model from checkpoint."""
        if self.model is None:
            self.create_model()
        
        print("=== Loading Regression Model ===")
        
        try:
            regression_epoch = load_checkpoint(
                path=self.checkpoint_config["checkpoint_dir"],
                models=self.model,
                epoch=self.checkpoint_config.get("checkpoint_index", None),
                device=self.device,
            )
            
            self.model.eval()
            self.model = self.model.to(self.device)
            self.model_loaded = True
            
            print(f"✓ Regression model loaded successfully from epoch {regression_epoch}")
            print(f"✓ Model set to evaluation mode")
            print(f"✓ Model moved to device: {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to load regression model: {e}")
            print(f"✗ Model will not be available for inference")
            self.model_loaded = False
            return False
        
        return True
    
    def inspect_model(self) -> None:
        """Inspect the regression model architecture and properties."""
        if not self.model_loaded:
            print("No regression model loaded to inspect")
            return
        
        print("=== Regression Model Architecture ===")
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
    
    def test_on_dummy_data(self, batch_size: int = 2) -> bool:
        """Test the regression model on dummy data."""
        if not self.model_loaded:
            print("No regression model loaded to test")
            return False
        
        print("=== Testing Regression Model on Dummy Data ===")
        
        # Create test inputs with correct dimensions
        height = width = 432
        
        print(f"Creating dummy test tensors:")
        print(f"  - HR input (zeros): {batch_size} x 2 x {height} x {width}")
        print(f"  - LR input (16 vars): {batch_size} x 16 x {height} x {width}")
        print(f"  - Grid channels (added internally): 4 channels")
        
        # Create test tensors
        hr_input = torch.zeros(batch_size, 2, height, width, device=self.device)
        lr_input = torch.randn(batch_size, 16, height, width, device=self.device)
        
        # Test forward pass
        try:
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                # Forward pass
                output = self.model(hr_input, lr_input)
                
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
    
    def predict(self, input_tensor: torch.Tensor, data_manager) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Run regression prediction on input data.
        
        Args:
            input_tensor: Normalized input tensor of shape (1, 16, H, W)
            data_manager: DataManager instance for denormalization
            
        Returns:
            Tuple of (raw_output_normalized, u10_denorm, v10_denorm)
        """
        if not self.model_loaded:
            raise ValueError("No regression model loaded. Call load_model() first.")
        
        print("=== Running Regression Prediction ===")
        
        try:
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                
                # Zero input for regression (as per training)
                zero_input = torch.zeros(1, 2, 432, 432, device=self.device)
                
                # Timing
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                
                if torch.cuda.is_available():
                    start_time.record()
                
                # Run regression - output is NORMALIZED
                output_regression = self.model(zero_input, input_tensor)
                
                if torch.cuda.is_available():
                    end_time.record()
                    torch.cuda.synchronize()
                    inference_time = start_time.elapsed_time(end_time)
                    print(f"✓ Real data inference time: {inference_time:.2f} ms")
                
                print(f"✓ Regression output shape: {output_regression.shape}")
                print(f"✓ Regression output range (NORMALIZED): [{output_regression.min():.3f}, {output_regression.max():.3f}]")
                
                # Denormalize predictions for visualization and metrics
                u10_pred_norm = output_regression[0, 0].cpu().numpy()
                v10_pred_norm = output_regression[0, 1].cpu().numpy()
                
                u10_denorm = data_manager.denormalize_output(u10_pred_norm, "U10")
                v10_denorm = data_manager.denormalize_output(v10_pred_norm, "V10")
                
                print(f"✓ Predictions denormalized")
                print(f"  - U10 range (DENORMALIZED): [{u10_denorm.min():.3f}, {u10_denorm.max():.3f}]")
                print(f"  - V10 range (DENORMALIZED): [{v10_denorm.min():.3f}, {v10_denorm.max():.3f}]")
                
                # Return normalized output for diffusion + denormalized for metrics
                return output_regression, u10_denorm, v10_denorm
                
        except Exception as e:
            print(f"✗ Regression prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the regression model."""
        if not self.model_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "architecture": "UNet",
            "input_channels": 20,
            "output_channels": 2,
            "resolution": "432x432",
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "model_channels": self.model_config.get("model_channels", "unknown"),
            "channel_mult": self.model_config.get("channel_mult", "unknown")
        }