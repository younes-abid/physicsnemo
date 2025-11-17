"""
Regression Pipeline for Weather Pipeline - Streamlit Demo

Handles regression model loading, testing, and inference operations.
Based on the successful prediction pipeline implementation.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from physicsnemo.models.diffusion.unet import UNet
from physicsnemo.launch.utils.checkpoint import load_checkpoint


class RegressionPipeline:
    """Manages regression model operations for Weather Pipeline."""
    
    def __init__(self, model_config: Dict[str, Any], checkpoint_path: str):
        """
        Initialize RegressionPipeline.
        
        Args:
            model_config: Dictionary containing model architecture parameters
            checkpoint_path: Full path to the checkpoint file
        """
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
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
        print(f"Loading from: {self.checkpoint_path}")
        
        try:
            # Extract directory and epoch from full path
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            checkpoint_filename = os.path.basename(self.checkpoint_path)
            
            # Extract epoch from filename (e.g., UNet.0.535008.mdlus -> 535008)
            parts = checkpoint_filename.split('.')
            if len(parts) >= 3:
                epoch = int(parts[-2])  # Second to last part before extension
            else:
                epoch = None
            
            regression_epoch = load_checkpoint(
                path=checkpoint_dir,
                models=self.model,
                epoch=epoch,
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
            self.model_loaded = False
            return False
        
        return True
    
    def test_model(self, batch_size: int = 2) -> bool:
        """Test the regression model on dummy data."""
        if not self.model_loaded:
            print("No regression model loaded to test")
            return False
        
        print("=== Testing Regression Model ===")
        
        # Create test inputs with correct dimensions
        height = width = 432
        
        # Create test tensors
        hr_input = torch.zeros(batch_size, 2, height, width, device=self.device)
        lr_input = torch.randn(batch_size, 16, height, width, device=self.device)
        
        # Test forward pass
        try:
            with torch.no_grad():
                output = self.model(hr_input, lr_input)
                
                print(f"✓ Forward pass successful!")
                print(f"✓ Output shape: {output.shape}")
                print(f"✓ Output statistics: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
                
                # Check for NaN or Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"⚠ Warning: Output contains NaN or infinite values")
                    return False
                
                return True
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
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
                
                # Run regression - output is NORMALIZED
                output_regression = self.model(zero_input, input_tensor)
                
                print(f"✓ Regression output shape: {output_regression.shape}")
                print(f"✓ Regression output range (NORMALIZED): [{output_regression.min():.3f}, {output_regression.max():.3f}]")
                
                # Denormalize predictions for visualization and metrics
                u10_pred_norm = output_regression[0, 0].cpu().numpy()
                v10_pred_norm = output_regression[0, 1].cpu().numpy()
                
                u10_denorm = data_manager.denormalize_output(u10_pred_norm, "U10")
                v10_denorm = data_manager.denormalize_output(v10_pred_norm, "V10")
                
                print(f"✓ Predictions denormalized")
                print(f"  - U10 range: [{u10_denorm.min():.3f}, {u10_denorm.max():.3f}]")
                print(f"  - V10 range: [{v10_denorm.min():.3f}, {v10_denorm.max():.3f}]")
                
                return output_regression, u10_denorm, v10_denorm
                
        except Exception as e:
            print(f"✗ Regression prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "loaded": True,
            "architecture": "UNet",
            "input_channels": 20,
            "output_channels": 2,
            "resolution": "432x432",
            "parameters": total_params,
            "device": str(self.device),
            "checkpoint_path": self.checkpoint_path,
            "model_channels": self.model_config.get("model_channels", "unknown"),
            "channel_mult": self.model_config.get("channel_mult", "unknown")
        }