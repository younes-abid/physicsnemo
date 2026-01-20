"""
SAR2Height Regression Pipeline

Handles regression model loading, testing, and inference operations for SAR-to-height prediction.
Adapted from the CorrDiff weather pipeline for SAR2Height use case.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from physicsnemo.models.diffusion.unet import UNet
from physicsnemo.launch.utils.checkpoint import load_checkpoint
import logging

logger = logging.getLogger(__name__)


class SAR2HeightRegressionPipeline:
    """Manages regression model operations for SAR2Height Pipeline."""
    
    def __init__(self, model_config: Dict[str, Any], checkpoint_path: str):
        """
        Initialize SAR2HeightRegressionPipeline.
        
        Args:
            model_config: Dictionary containing model architecture parameters
            checkpoint_path: Full path to the regression checkpoint file
        """
        self.model_config = model_config
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.n_input_channels = None
        self.n_output_channels = 1  # DSM height
        
    def create_model(self, n_input_channels: int) -> UNet:
        """Create the regression UNet model for SAR2Height."""
        print("=== Creating SAR2Height Regression Model ===")
        
        self.n_input_channels = n_input_channels
        total_input_channels = n_input_channels + 4  # SAR features + 4 grid channels
        
        # Create regression model
        self.model = UNet(
            img_in_channels=total_input_channels,  
            img_out_channels=self.n_output_channels,  # DSM height
            img_resolution=432,
            **self.model_config,
        )
        
        print(f"SAR2Height Regression model created with:")
        print(f"  - Input channels: {total_input_channels} ({n_input_channels} SAR + 4 grid)")
        print(f"  - Output channels: {self.n_output_channels} (DSM height)")
        print(f"  - Resolution: 432x432")
        print(f"  - Model channels: {self.model_config['model_channels']}")
        print(f"  - Channel multipliers: {self.model_config['channel_mult']}")
        
        return self.model
    
    def load_model(self, n_input_channels: int) -> bool:
        """Load the SAR2Height regression model from checkpoint."""
        if self.model is None:
            self.create_model(n_input_channels)
        
        print("=== Loading SAR2Height Regression Model ===")
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
            
            print(f"✓ SAR2Height Regression model loaded successfully from epoch {regression_epoch}")
            print(f"✓ Model set to evaluation mode")
            print(f"✓ Model moved to device: {self.device}")
            
        except Exception as e:
            print(f"✗ Failed to load SAR2Height regression model: {e}")
            self.model_loaded = False
            return False
        
        return True
    
    def test_model(self, n_input_channels: int, batch_size: int = 2) -> bool:
        """Test the SAR2Height regression model on dummy data."""
        if not self.model_loaded:
            print("No SAR2Height regression model loaded to test")
            return False
        
        print("=== Testing SAR2Height Regression Model ===")
        
        # Create test inputs with correct dimensions
        height = width = 432
        total_input_channels = n_input_channels + 4  # SAR features + grid channels
        
        # Create test tensors
        hr_input = torch.zeros(batch_size, self.n_output_channels, height, width, device=self.device)
        lr_input = torch.randn(batch_size, n_input_channels, height, width, device=self.device)
        
        # Test forward pass
        try:
            with torch.no_grad():
                output = self.model(hr_input, lr_input)
                
                print(f"✓ Forward pass successful!")
                print(f"✓ Output shape: {output.shape}")
                print(f"✓ Expected shape: ({batch_size}, {self.n_output_channels}, {height}, {width})")
                print(f"✓ Output statistics: mean={output.mean().item():.6f}, std={output.std().item():.6f}")
                
                # Check for NaN or Inf
                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"⚠ Warning: Output contains NaN or infinite values")
                    return False
                
                return True
                
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            return False
    
    def predict(self, input_tensor: torch.Tensor, data_manager) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Run SAR2Height regression prediction on input data.
        
        Args:
            input_tensor: Normalized input tensor of shape (1, N_channels, H, W)
            data_manager: SAR2HeightDataManager instance for denormalization
            
        Returns:
            Tuple of (raw_output_normalized, dsm_denormalized)
        """
        if not self.model_loaded:
            raise ValueError("No regression model loaded. Call load_model() first.")
        
        print("=== Running SAR2Height Regression Prediction ===")
        
        try:
            with torch.no_grad():
                input_tensor = input_tensor.to(self.device)
                
                # Zero input for regression (as per training protocol)
                zero_input = torch.zeros(1, self.n_output_channels, 432, 432, device=self.device)
                
                # Run regression - output is NORMALIZED
                output_regression = self.model(zero_input, input_tensor)
                
                print(f"✓ Regression output shape: {output_regression.shape}")
                print(f"✓ Regression output range (NORMALIZED): [{output_regression.min():.3f}, {output_regression.max():.3f}]")
                
                # Denormalize predictions for visualization and metrics
                dsm_pred_norm = output_regression[0, 0].cpu().numpy()
                dsm_denorm = data_manager.denormalize_output(dsm_pred_norm, "dsm_height")
                
                print(f"✓ Prediction denormalized")
                print(f"  - DSM range: [{dsm_denorm.min():.3f}, {dsm_denorm.max():.3f}] meters")
                
                return output_regression, dsm_denorm
                
        except Exception as e:
            print(f"✗ SAR2Height regression prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded SAR2Height regression model."""
        if not self.model_loaded:
            return {"loaded": False}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        
        return {
            "loaded": True,
            "architecture": "UNet",
            "input_channels": self.n_input_channels + 4 if self.n_input_channels else "unknown",
            "sar_channels": self.n_input_channels,
            "grid_channels": 4,
            "output_channels": self.n_output_channels,
            "resolution": "432x432",
            "parameters": total_params,
            "device": str(self.device),
            "checkpoint_path": self.checkpoint_path,
            "model_channels": self.model_config.get("model_channels", "unknown"),
            "channel_mult": self.model_config.get("channel_mult", "unknown"),
            "task": "SAR-to-height regression"
        }