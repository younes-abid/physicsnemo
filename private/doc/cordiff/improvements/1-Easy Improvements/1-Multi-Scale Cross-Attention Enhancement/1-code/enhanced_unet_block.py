"""
Enhanced UNet Block Integration for CorrDiff Multi-Scale Cross-Attention

This module provides the integration layer that enhances the existing UNetBlock
in PhysicsNeMo with cross-attention capabilities while maintaining full
backward compatibility.

Key Components:
- EnhancedUNetBlock: Extended UNetBlock with cross-attention
- CrossAttentionConfig: Configuration management
- Integration utilities for seamless PhysicsNeMo integration

Author: CorrDiff Enhancement Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass

# Import the original UNetBlock from PhysicsNeMo
# Note: In actual implementation, this would be imported from:
# from physicsnemo.models.diffusion.layers import UNetBlock
# For this demo, we'll create a simplified version

from cross_attention import CrossAttentionBlock
from multiscale_processor import MultiScaleProcessor


@dataclass
class CrossAttentionConfig:
    """Configuration class for cross-attention enhancement."""
    
    # Basic attention configuration
    enable_cross_attention: bool = True
    num_heads: int = 8
    dropout: float = 0.1
    
    # Multi-scale configuration
    scales: list = None  # Will default to [1.0, 0.5, 0.25]
    fusion_method: str = "weighted_sum"  # Options: weighted_sum, conv_fusion, attention_fusion
    learnable_scale_weights: bool = True
    
    # LR feature configuration
    lr_channels: int = 16  # Number of channels in LR input
    lr_projection_dim: Optional[int] = None  # If None, uses block output channels
    
    # Weather-specific features
    use_weather_aware: bool = True
    apply_physics_constraints: bool = True
    constraint_strength: float = 0.1
    
    # Performance optimization
    use_efficient_attention: bool = True
    chunk_size: int = 1024
    use_gradient_checkpointing: bool = False
    
    # Position encoding
    use_positional_encoding: bool = True
    positional_encoding_type: str = "sinusoidal"  # Options: sinusoidal, learnable
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.scales is None:
            self.scales = [1.0, 0.5, 0.25]


class SimplifiedUNetBlock(nn.Module):
    """
    Simplified version of UNetBlock for demonstration.
    In actual implementation, this would inherit from the real UNetBlock.
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 up: bool = False, down: bool = False, attention: bool = False,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.up = up
        self.down = down
        self.use_attention = attention
        
        # Basic UNet components
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Embedding projection
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, out_channels)
        )
        
        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
            
        # Up/downsampling
        if up:
            self.upsample = nn.ConvTranspose2d(out_channels, out_channels, 4, 2, 1)
        elif down:
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        
        # Self-attention (original)
        if attention:
            self.self_attention = nn.MultiheadAttention(out_channels, 8, batch_first=True)
            self.attn_norm = nn.GroupNorm(32, out_channels)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Original UNetBlock forward pass."""
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add embedding
        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        
        # Skip connection
        h = h + self.skip_connection(x)
        
        # Up/downsampling
        if hasattr(self, 'upsample'):
            h = self.upsample(h)
        elif hasattr(self, 'downsample'):
            h = self.downsample(h)
            
        # Self-attention
        if self.use_attention:
            B, C, H, W = h.shape
            h_attn = h.flatten(2).transpose(1, 2)  # [B, HW, C]
            h_attn, _ = self.self_attention(h_attn, h_attn, h_attn)
            h = self.attn_norm(h + h_attn.transpose(1, 2).view(B, C, H, W))
        
        return h


class EnhancedUNetBlock(SimplifiedUNetBlock):
    """
    Enhanced UNet block with multi-scale cross-attention capabilities.
    
    Extends the original UNetBlock with cross-attention between HR and LR features
    while maintaining full backward compatibility.
    """
    
    def __init__(self, in_channels: int, out_channels: int, emb_channels: int,
                 cross_attention_config: Optional[CrossAttentionConfig] = None,
                 **kwargs):
        super().__init__(in_channels, out_channels, emb_channels, **kwargs)
        
        # Cross-attention configuration
        if cross_attention_config is None:
            cross_attention_config = CrossAttentionConfig(enable_cross_attention=False)
        
        self.cross_attention_config = cross_attention_config
        self.use_cross_attention = cross_attention_config.enable_cross_attention
        
        # Initialize cross-attention components
        if self.use_cross_attention:
            self._init_cross_attention_components()
    
    def _init_cross_attention_components(self):
        """Initialize cross-attention specific components."""
        config = self.cross_attention_config
        
        # LR feature projection
        lr_proj_dim = config.lr_projection_dim or self.out_channels
        self.lr_projection = nn.Conv2d(
            config.lr_channels, 
            lr_proj_dim, 
            kernel_size=1,
            bias=True
        )
        
        # Multi-scale cross-attention processor
        self.cross_attention_processor = MultiScaleProcessor(
            embed_dim=self.out_channels,
            scales=config.scales,
            num_heads=config.num_heads,
            use_weather_aware=config.use_weather_aware,
            fusion_method=config.fusion_method,
            learnable_scale_weights=config.learnable_scale_weights
        )
        
        # Additional normalization for cross-attention path
        self.cross_attn_norm = nn.GroupNorm(32, self.out_channels)
        
        # Gate mechanism for blending original and cross-attention features
        self.cross_attn_gate = nn.Sequential(
            nn.Conv2d(self.out_channels * 2, self.out_channels, 1),
            nn.Sigmoid()
        )
        
        # Initialize cross-attention weights
        self._init_cross_attention_weights()
    
    def _init_cross_attention_weights(self):
        """Initialize cross-attention component weights."""
        # Initialize LR projection
        nn.init.xavier_uniform_(self.lr_projection.weight)
        nn.init.constant_(self.lr_projection.bias, 0)
        
        # Initialize gate weights to favor original features initially
        with torch.no_grad():
            self.cross_attn_gate[0].weight.fill_(0.0)
            self.cross_attn_gate[0].bias.fill_(0.0)  # This makes gate output 0.5 initially
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor, 
                lr_features: Optional[torch.Tensor] = None,
                return_attention_maps: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Enhanced forward pass with optional cross-attention.
        
        Args:
            x: Input features [B, C, H, W]
            emb: Embedding features [B, emb_dim]
            lr_features: Optional LR features for cross-attention [B, lr_channels, H, W]
            return_attention_maps: Whether to return attention maps for analysis
            
        Returns:
            Enhanced features, optionally with attention maps
        """
        # Original UNet processing
        h = super().forward(x, emb)
        
        # Apply cross-attention enhancement if enabled and LR features provided
        attention_maps = None
        if self.use_cross_attention and lr_features is not None:
            # Apply gradient checkpointing if requested
            if self.cross_attention_config.use_gradient_checkpointing and self.training:
                h_cross, attention_maps = torch.utils.checkpoint.checkpoint(
                    self._cross_attention_forward,
                    h, lr_features, return_attention_maps,
                    use_reentrant=False
                )
            else:
                h_cross, attention_maps = self._cross_attention_forward(
                    h, lr_features, return_attention_maps
                )
            
            # Gate-controlled blending
            gate_input = torch.cat([h, h_cross], dim=1)
            gate_weights = self.cross_attn_gate(gate_input)
            h = gate_weights * h_cross + (1 - gate_weights) * h
        
        if return_attention_maps:
            return h, attention_maps
        else:
            return h
    
    def _cross_attention_forward(self, hr_features: torch.Tensor, 
                               lr_features: torch.Tensor,
                               return_attention_maps: bool = False) -> Tuple:
        """Internal cross-attention forward pass."""
        # Project LR features to match HR feature dimensions
        lr_projected = self.lr_projection(lr_features)
        
        # Ensure spatial dimensions match
        if lr_projected.shape[-2:] != hr_features.shape[-2:]:
            lr_projected = F.interpolate(
                lr_projected, 
                size=hr_features.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply multi-scale cross-attention
        if return_attention_maps:
            enhanced_features, _, attention_maps = self.cross_attention_processor(
                hr_features, lr_projected, 
                return_attention_maps=True
            )
        else:
            enhanced_features = self.cross_attention_processor(
                hr_features, lr_projected
            )
            attention_maps = None
        
        # Normalization
        enhanced_features = self.cross_attn_norm(enhanced_features)
        
        return enhanced_features, attention_maps
    
    def get_cross_attention_config(self) -> CrossAttentionConfig:
        """Get current cross-attention configuration."""
        return self.cross_attention_config
    
    def update_cross_attention_config(self, **kwargs):
        """Update cross-attention configuration."""
        for key, value in kwargs.items():
            if hasattr(self.cross_attention_config, key):
                setattr(self.cross_attention_config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


class EnhancedScalingFunction:
    """
    Enhanced scaling function that replaces simple concatenation in EDMPrecondSuperResolution
    with attention-based feature fusion.
    """
    
    def __init__(self, embed_dim: int = 256, use_attention_fusion: bool = True,
                 cross_attention_config: Optional[CrossAttentionConfig] = None):
        self.embed_dim = embed_dim
        self.use_attention_fusion = use_attention_fusion
        
        if use_attention_fusion:
            if cross_attention_config is None:
                cross_attention_config = CrossAttentionConfig()
            
            self.attention_fusion = CrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=cross_attention_config.num_heads,
                dropout=cross_attention_config.dropout,
                use_efficient_attention=cross_attention_config.use_efficient_attention,
                use_positional_encoding=cross_attention_config.use_positional_encoding
            )
            
            # Feature projection and combination layers
            self.hr_projection = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
            self.lr_projection = nn.Conv2d(cross_attention_config.lr_channels, embed_dim, 1)
            self.fusion_norm = nn.GroupNorm(32, embed_dim)
    
    def __call__(self, x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor) -> torch.Tensor:
        """
        Enhanced scaling function with attention-based fusion.
        
        Args:
            x: HR noise features [B, C_hr, H, W]
            img_lr: LR conditioning features [B, C_lr, H, W]
            c_in: Scaling coefficient
            
        Returns:
            Enhanced combined features [B, C_combined, H, W]
        """
        if not self.use_attention_fusion:
            # Fallback to original concatenation
            return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)
        
        # Scale HR features
        hr_scaled = c_in * x
        
        # Project features to common embedding dimension
        hr_projected = self.hr_projection(hr_scaled)
        lr_projected = self.lr_projection(img_lr)
        
        # Ensure spatial dimensions match
        if lr_projected.shape[-2:] != hr_projected.shape[-2:]:
            lr_projected = F.interpolate(
                lr_projected, 
                size=hr_projected.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply cross-attention fusion
        attended_hr, _ = self.attention_fusion(hr_projected, lr_projected)
        
        # Normalize and combine
        enhanced_hr = self.fusion_norm(attended_hr)
        
        # Concatenate enhanced HR with original LR for backward compatibility
        return torch.cat([enhanced_hr, img_lr.to(x.dtype)], dim=1)


# Integration utilities
class CrossAttentionIntegration:
    """Utilities for integrating cross-attention into existing PhysicsNeMo models."""
    
    @staticmethod
    def create_enhanced_config(base_config: Dict[str, Any], 
                             enhancement_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create enhanced model configuration by merging base and enhancement configs.
        
        Args:
            base_config: Original model configuration
            enhancement_config: Cross-attention enhancement configuration
            
        Returns:
            Merged configuration dictionary
        """
        enhanced_config = base_config.copy()
        
        # Add cross-attention configuration
        enhanced_config["use_cross_attention"] = enhancement_config.get("enable", True)
        enhanced_config["cross_attention_config"] = CrossAttentionConfig(**enhancement_config)
        
        return enhanced_config
    
    @staticmethod
    def convert_unet_blocks(original_blocks: nn.ModuleDict, 
                          cross_attention_config: CrossAttentionConfig) -> nn.ModuleDict:
        """
        Convert existing UNet blocks to enhanced versions with cross-attention.
        
        Args:
            original_blocks: Original UNet blocks
            cross_attention_config: Configuration for cross-attention
            
        Returns:
            Enhanced UNet blocks
        """
        enhanced_blocks = nn.ModuleDict()
        
        for name, block in original_blocks.items():
            if isinstance(block, SimplifiedUNetBlock):  # In real implementation: UNetBlock
                # Create enhanced version
                enhanced_block = EnhancedUNetBlock(
                    in_channels=block.in_channels,
                    out_channels=block.out_channels,
                    emb_channels=block.emb_channels,
                    up=block.up,
                    down=block.down,
                    attention=block.use_attention,
                    cross_attention_config=cross_attention_config
                )
                
                # Copy weights from original block
                enhanced_block.load_state_dict(block.state_dict(), strict=False)
                enhanced_blocks[name] = enhanced_block
            else:
                # Keep non-UNetBlock modules as-is
                enhanced_blocks[name] = block
                
        return enhanced_blocks
    
    @staticmethod
    def analyze_attention_patterns(attention_maps: torch.Tensor, 
                                 save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Analyze attention patterns for meteorological insights.
        
        Args:
            attention_maps: Attention weight tensors [B, H, N, N]
            save_path: Optional path to save visualization
            
        Returns:
            Dictionary of attention pattern statistics
        """
        stats = {}
        
        # Compute attention statistics
        stats['mean_attention'] = attention_maps.mean().item()
        stats['attention_entropy'] = -torch.sum(
            attention_maps * torch.log(attention_maps + 1e-8), dim=-1
        ).mean().item()
        
        # Compute locality vs. globality
        N = attention_maps.shape[-1]
        local_mask = torch.zeros_like(attention_maps)
        center = N // 2
        window = N // 8
        
        local_mask[:, :, center-window:center+window, center-window:center+window] = 1
        local_attention = (attention_maps * local_mask).sum(dim=(-2, -1))
        stats['locality_ratio'] = local_attention.mean().item()
        
        # Save visualization if requested
        if save_path:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(10, 10))
            
            # Plot average attention map
            avg_attention = attention_maps.mean(dim=(0, 1)).cpu().numpy()
            axes[0, 0].imshow(avg_attention, cmap='viridis')
            axes[0, 0].set_title('Average Attention Map')
            
            # Plot attention entropy
            entropy_map = -torch.sum(attention_maps * torch.log(attention_maps + 1e-8), dim=-1)
            avg_entropy = entropy_map.mean(dim=(0, 1)).cpu().numpy()
            axes[0, 1].imshow(avg_entropy, cmap='plasma')
            axes[0, 1].set_title('Attention Entropy')
            
            # Plot locality distribution
            axes[1, 0].hist(local_attention.flatten().cpu().numpy(), bins=50)
            axes[1, 0].set_title('Locality Distribution')
            
            # Plot attention weights distribution
            axes[1, 1].hist(attention_maps.flatten().cpu().numpy(), bins=50)
            axes[1, 1].set_title('Attention Weights Distribution')
            
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    # Test enhanced UNet block
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create configuration
    config = CrossAttentionConfig(
        enable_cross_attention=True,
        scales=[1.0, 0.5, 0.25],
        lr_channels=16,
        use_weather_aware=True
    )
    
    # Create test data
    B, C_hr, C_lr, H, W = 2, 256, 16, 64, 64
    x = torch.randn(B, C_hr, H, W, device=device)
    emb = torch.randn(B, 512, device=device)
    lr_features = torch.randn(B, C_lr, H, W, device=device)
    
    # Test enhanced UNet block
    enhanced_block = EnhancedUNetBlock(
        in_channels=C_hr,
        out_channels=C_hr, 
        emb_channels=512,
        cross_attention_config=config
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        output, attention_maps = enhanced_block(
            x, emb, lr_features, return_attention_maps=True
        )
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Enhanced UNet block test passed!")
    
    # Test enhanced scaling function
    scaling_fn = EnhancedScalingFunction(embed_dim=C_hr, cross_attention_config=config)
    
    with torch.no_grad():
        c_in = torch.ones(B, 1, 1, 1, device=device)
        scaled_output = scaling_fn(x, lr_features, c_in)
    
    print(f"Scaled output shape: {scaled_output.shape}")
    print("Enhanced scaling function test passed!")
    
    # Test attention pattern analysis
    if attention_maps is not None:
        stats = CrossAttentionIntegration.analyze_attention_patterns(attention_maps)
        print(f"Attention statistics: {stats}")
        print("Attention analysis test passed!")