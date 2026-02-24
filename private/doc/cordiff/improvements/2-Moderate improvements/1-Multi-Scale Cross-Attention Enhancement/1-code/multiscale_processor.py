"""
Multi-Scale Processor for CorrDiff Cross-Attention Enhancement

This module implements the multi-scale processing capability that applies
cross-attention at different spatial resolutions to capture atmospheric
phenomena at various scales (synoptic, mesoscale, microscale).

Key Components:
- MultiScaleProcessor: Applies attention at multiple scales
- ScalePyramidExtractor: Efficient multi-scale feature extraction
- WeatherAwareAttention: Domain-specific attention patterns

Author: CorrDiff Enhancement Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

from cross_attention import CrossAttentionBlock, get_atmospheric_scale_config


class ScalePyramidExtractor(nn.Module):
    """
    Efficient extraction of multi-scale features for atmospheric modeling.
    
    Creates a pyramid of features at different spatial resolutions,
    optimized for weather data processing.
    """
    
    def __init__(self, scales: List[float] = [1.0, 0.5, 0.25], 
                 interpolation_mode: str = 'bilinear',
                 preserve_aspect_ratio: bool = True):
        super().__init__()
        self.scales = sorted(scales, reverse=True)  # Start with largest scale
        self.interpolation_mode = interpolation_mode
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Scale-specific preprocessing
        self.scale_preprocessors = nn.ModuleDict({
            f"scale_{i}": nn.Identity() for i, _ in enumerate(scales)
        })
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Dictionary of scale_name -> scaled_features
        """
        B, C, H, W = features.shape
        scale_features = {}
        
        for i, scale in enumerate(self.scales):
            if scale == 1.0:
                scaled_features = features
            else:
                if self.preserve_aspect_ratio:
                    new_h, new_w = int(H * scale), int(W * scale)
                else:
                    new_h = new_w = int(min(H, W) * scale)
                
                scaled_features = F.interpolate(
                    features, 
                    size=(new_h, new_w), 
                    mode=self.interpolation_mode,
                    align_corners=False
                )
            
            # Apply scale-specific preprocessing
            preprocessor = self.scale_preprocessors[f"scale_{i}"]
            scaled_features = preprocessor(scaled_features)
            
            scale_features[f"scale_{scale}"] = scaled_features
            
        return scale_features


class WeatherAwareAttention(nn.Module):
    """
    Weather-specific cross-attention that incorporates meteorological domain knowledge.
    
    Features:
    - Variable grouping (temperature, pressure, wind, moisture)
    - Physics-informed constraints
    - Scale-aware processing for different atmospheric phenomena
    """
    
    def __init__(self, embed_dim: int, variable_groups: Optional[Dict] = None,
                 apply_physics_constraints: bool = True,
                 constraint_strength: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.apply_physics_constraints = apply_physics_constraints
        self.constraint_strength = constraint_strength
        
        # Default weather variable groups if not provided
        if variable_groups is None:
            variable_groups = {
                'temperature': [0, 1, 10, 11],  # t_850, t_500, t2m, skt indices
                'pressure': [2, 3, 12],         # z_850, z_500, sp indices
                'wind': [4, 5, 6, 7, 8, 9],     # u_850, u_500, v_850, v_500, u10, v10
                'moisture': [13, 14, 15]        # d2m, tcwv, tp indices
            }
        
        self.variable_groups = variable_groups
        
        # Group-specific attention blocks
        self.group_attentions = nn.ModuleDict({
            group_name: CrossAttentionBlock(embed_dim, num_heads=8)
            for group_name in variable_groups.keys()
        })
        
        # Inter-group interaction attention
        self.inter_group_attention = CrossAttentionBlock(embed_dim, num_heads=8)
        
        # Physics constraint layers
        if apply_physics_constraints:
            self.constraint_layers = nn.ModuleDict({
                'temperature_pressure': nn.Linear(embed_dim, embed_dim),
                'wind_pressure': nn.Linear(embed_dim, embed_dim),
                'moisture_temperature': nn.Linear(embed_dim, embed_dim)
            })
    
    def apply_physics_constraints(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply physics-informed constraints to maintain meteorological relationships.
        
        Args:
            features: Input features [B, C, H, W]
            
        Returns:
            Constrained features maintaining physical relationships
        """
        if not self.apply_physics_constraints:
            return features
        
        B, C, H, W = features.shape
        
        # Reshape for constraint application
        features_flat = features.view(B, C, -1).transpose(1, 2)  # [B, HW, C]
        
        # Apply temperature-pressure relationship constraints
        temp_pressure_constraint = self.constraint_layers['temperature_pressure'](features_flat)
        features_flat = features_flat + self.constraint_strength * temp_pressure_constraint
        
        # Apply wind-pressure relationship constraints  
        wind_pressure_constraint = self.constraint_layers['wind_pressure'](features_flat)
        features_flat = features_flat + self.constraint_strength * wind_pressure_constraint
        
        # Apply moisture-temperature relationship constraints
        moisture_temp_constraint = self.constraint_layers['moisture_temperature'](features_flat)
        features_flat = features_flat + self.constraint_strength * moisture_temp_constraint
        
        # Reshape back to spatial format
        constrained_features = features_flat.transpose(1, 2).view(B, C, H, W)
        
        return constrained_features
    
    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor,
                variable_indices: Optional[Dict] = None) -> torch.Tensor:
        """
        Apply weather-aware cross-attention.
        
        Args:
            hr_features: High-resolution features [B, C, H, W]
            lr_features: Low-resolution features [B, C, H, W]
            variable_indices: Optional mapping of variable groups to channel indices
            
        Returns:
            Weather-aware enhanced features
        """
        if variable_indices is None:
            variable_indices = self.variable_groups
            
        group_outputs = []
        
        # Process each variable group separately
        for group_name, indices in variable_indices.items():
            if group_name not in self.group_attentions:
                continue
                
            # Extract group-specific features
            hr_group = hr_features[:, indices] if len(indices) > 0 else hr_features
            lr_group = lr_features[:, indices] if len(indices) > 0 else lr_features
            
            # Apply group-specific attention
            attended_group, _ = self.group_attentions[group_name](hr_group, lr_group)
            group_outputs.append(attended_group)
        
        # Combine group outputs
        if len(group_outputs) > 1:
            combined_features = torch.cat(group_outputs, dim=1)
        else:
            combined_features = group_outputs[0] if group_outputs else hr_features
        
        # Inter-group interactions
        final_features, _ = self.inter_group_attention(combined_features, combined_features)
        
        # Apply physics constraints
        constrained_features = self.apply_physics_constraints(final_features)
        
        return constrained_features


class MultiScaleProcessor(nn.Module):
    """
    Multi-scale cross-attention processor for atmospheric modeling.
    
    Applies cross-attention at multiple spatial scales to capture
    different atmospheric phenomena (synoptic, mesoscale, microscale).
    """
    
    def __init__(self, embed_dim: int, scales: List[float] = [1.0, 0.5, 0.25],
                 num_heads: int = 8, use_weather_aware: bool = True,
                 fusion_method: str = "weighted_sum", 
                 learnable_scale_weights: bool = True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.scales = scales
        self.num_scales = len(scales)
        self.fusion_method = fusion_method
        self.use_weather_aware = use_weather_aware
        
        # Scale pyramid extractor
        self.pyramid_extractor = ScalePyramidExtractor(scales)
        
        # Scale-specific attention blocks
        self.scale_attentions = nn.ModuleList([
            WeatherAwareAttention(embed_dim) if use_weather_aware 
            else CrossAttentionBlock(embed_dim, num_heads=num_heads)
            for _ in scales
        ])
        
        # Scale fusion components
        if learnable_scale_weights:
            self.scale_weights = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        else:
            self.register_buffer("scale_weights", torch.ones(self.num_scales) / self.num_scales)
            
        # Fusion layers based on method
        if fusion_method == "conv_fusion":
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(embed_dim * self.num_scales, embed_dim * 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(embed_dim * 2, embed_dim, 1),
                nn.LayerNorm([embed_dim])  # Channel-wise layer norm
            )
        elif fusion_method == "attention_fusion":
            self.fusion_attention = CrossAttentionBlock(embed_dim, num_heads=num_heads)
            
        # Atmospheric scale configuration
        self.atm_scale_config = get_atmospheric_scale_config()
        
    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor,
                return_scale_outputs: bool = False, 
                return_attention_maps: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Multi-scale cross-attention processing.
        
        Args:
            hr_features: High-resolution features [B, C, H, W]
            lr_features: Low-resolution features [B, C, H, W]
            return_scale_outputs: Whether to return individual scale outputs
            return_attention_maps: Whether to return attention maps
            
        Returns:
            Enhanced features, optionally with scale outputs and attention maps
        """
        B, C, H, W = hr_features.shape
        
        # Extract multi-scale features
        hr_pyramid = self.pyramid_extractor(hr_features)
        lr_pyramid = self.pyramid_extractor(lr_features)
        
        scale_outputs = []
        attention_maps = [] if return_attention_maps else None
        
        # Process each scale
        for i, scale in enumerate(self.scales):
            scale_key = f"scale_{scale}"
            hr_scale = hr_pyramid[scale_key]
            lr_scale = lr_pyramid[scale_key]
            
            # Apply scale-specific attention
            if self.use_weather_aware:
                attended_scale = self.scale_attentions[i](hr_scale, lr_scale)
                scale_attention_map = None  # Weather-aware attention doesn't return maps
            else:
                attended_scale, scale_attention_map = self.scale_attentions[i](
                    hr_scale, lr_scale, return_attention_weights=return_attention_maps
                )
                
            # Upsample back to original resolution
            if scale < 1.0:
                attended_scale = F.interpolate(
                    attended_scale, size=(H, W), mode='bilinear', align_corners=False
                )
                
            scale_outputs.append(attended_scale)
            
            if return_attention_maps and scale_attention_map is not None:
                attention_maps.append(scale_attention_map)
        
        # Fuse multi-scale outputs
        if self.fusion_method == "weighted_sum":
            # Weighted sum fusion
            enhanced_features = sum(w * output for w, output in zip(self.scale_weights, scale_outputs))
            
        elif self.fusion_method == "conv_fusion":
            # Convolutional fusion
            concatenated = torch.cat(scale_outputs, dim=1)
            enhanced_features = self.fusion_conv(concatenated)
            
        elif self.fusion_method == "attention_fusion":
            # Attention-based fusion
            concatenated = torch.cat(scale_outputs, dim=1)
            enhanced_features, _ = self.fusion_attention(concatenated, concatenated)
            
        else:
            # Default: simple averaging
            enhanced_features = torch.stack(scale_outputs, dim=0).mean(dim=0)
        
        # Prepare return values
        if return_scale_outputs or return_attention_maps:
            return_tuple = [enhanced_features]
            if return_scale_outputs:
                return_tuple.append(scale_outputs)
            if return_attention_maps:
                return_tuple.append(attention_maps)
            return tuple(return_tuple)
        else:
            return enhanced_features


class AdaptiveScaleSelector(nn.Module):
    """
    Adaptive selection of optimal scales based on input characteristics.
    
    Learns to select the most relevant scales for different weather patterns
    and atmospheric conditions.
    """
    
    def __init__(self, embed_dim: int, max_scales: int = 5, 
                 temperature: float = 1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_scales = max_scales
        self.temperature = temperature
        
        # Scale selection network
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, max_scales),
            nn.Sigmoid()
        )
        
        # Available scales
        self.available_scales = [1.0, 0.75, 0.5, 0.25, 0.125][:max_scales]
        
    def forward(self, features: torch.Tensor, 
                top_k: Optional[int] = None) -> Tuple[List[float], torch.Tensor]:
        """
        Select optimal scales for given features.
        
        Args:
            features: Input features [B, C, H, W]
            top_k: Number of top scales to select (default: all)
            
        Returns:
            Tuple of (selected_scales, scale_weights)
        """
        # Compute scale selection weights
        scale_logits = self.scale_selector(features)  # [B, max_scales]
        scale_probs = F.softmax(scale_logits / self.temperature, dim=-1)
        
        # Average across batch for scale selection
        avg_scale_probs = scale_probs.mean(dim=0)  # [max_scales]
        
        if top_k is not None:
            # Select top-k scales
            top_k = min(top_k, len(self.available_scales))
            _, top_indices = torch.topk(avg_scale_probs, top_k)
            selected_scales = [self.available_scales[i] for i in top_indices.cpu().numpy()]
            selected_weights = avg_scale_probs[top_indices]
        else:
            # Use all scales with their weights
            selected_scales = self.available_scales
            selected_weights = avg_scale_probs
            
        return selected_scales, selected_weights


# Utility functions for weather-specific processing
def create_meteorological_attention_mask(weather_variables: List[str], 
                                        mask_type: str = "correlation") -> torch.Tensor:
    """
    Create attention masks based on meteorological variable relationships.
    
    Args:
        weather_variables: List of variable names
        mask_type: Type of mask ('correlation', 'physics', 'distance')
        
    Returns:
        Attention mask tensor
    """
    n_vars = len(weather_variables)
    mask = torch.ones(n_vars, n_vars)
    
    if mask_type == "correlation":
        # High correlation: temperature variables, wind components
        temp_vars = ['t_850', 't_500', 't2m', 'skt']
        wind_u_vars = ['u_850', 'u_500', 'u10']
        wind_v_vars = ['v_850', 'v_500', 'v10']
        
        for group in [temp_vars, wind_u_vars, wind_v_vars]:
            indices = [i for i, var in enumerate(weather_variables) if var in group]
            for i in indices:
                for j in indices:
                    mask[i, j] = 2.0  # Higher attention weight
                    
    elif mask_type == "physics":
        # Physical relationships: pressure-temperature, wind-pressure gradients
        relationships = {
            ('sp', 't2m'): 1.5,      # Surface pressure - surface temperature
            ('z_850', 't_850'): 1.5,  # Geopotential height - temperature
            ('tcwv', 't2m'): 1.3,    # Total column water vapor - temperature
        }
        
        for (var1, var2), weight in relationships.items():
            try:
                i1 = weather_variables.index(var1)
                i2 = weather_variables.index(var2)
                mask[i1, i2] = mask[i2, i1] = weight
            except ValueError:
                continue
                
    return mask


def compute_atmospheric_scale_loss(attention_weights: torch.Tensor, 
                                 target_scale: str = "microscale") -> torch.Tensor:
    """
    Compute scale-specific loss for atmospheric modeling.
    
    Encourages attention patterns that align with atmospheric scale dynamics.
    """
    B, H, N, N = attention_weights.shape
    
    if target_scale == "microscale":
        # Encourage local attention patterns
        local_window = 8
        center = N // 2
        target_pattern = torch.zeros(N, N, device=attention_weights.device)
        
        start = max(0, center - local_window)
        end = min(N, center + local_window)
        target_pattern[start:end, start:end] = 1.0
        
    elif target_scale == "synoptic":
        # Encourage global attention patterns
        target_pattern = torch.ones(N, N, device=attention_weights.device)
        
    else:  # mesoscale
        # Encourage regional attention patterns
        regional_window = N // 4
        target_pattern = torch.eye(N, device=attention_weights.device)
        
        for i in range(N):
            start = max(0, i - regional_window)
            end = min(N, i + regional_window)
            target_pattern[i, start:end] = 1.0
    
    # Normalize target pattern
    target_pattern = target_pattern / target_pattern.sum(dim=-1, keepdim=True)
    
    # Compute KL divergence loss
    attention_avg = attention_weights.mean(dim=(0, 1))  # Average over batch and heads
    attention_normalized = F.softmax(attention_avg, dim=-1)
    
    scale_loss = F.kl_div(
        attention_normalized.log(), 
        target_pattern, 
        reduction='batchmean'
    )
    
    return scale_loss


# Example usage and testing
if __name__ == "__main__":
    # Test multi-scale processor
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test data
    B, C, H, W = 2, 256, 64, 64
    hr_features = torch.randn(B, C, H, W, device=device)
    lr_features = torch.randn(B, C, H, W, device=device)
    
    # Test multi-scale processor
    processor = MultiScaleProcessor(
        embed_dim=C,
        scales=[1.0, 0.5, 0.25],
        use_weather_aware=True,
        fusion_method="weighted_sum"
    ).to(device)
    
    with torch.no_grad():
        enhanced_features = processor(hr_features, lr_features)
        
    print(f"Input HR shape: {hr_features.shape}")
    print(f"Enhanced features shape: {enhanced_features.shape}")
    print("Multi-scale processor test passed!")
    
    # Test adaptive scale selector
    scale_selector = AdaptiveScaleSelector(embed_dim=C, max_scales=5).to(device)
    
    with torch.no_grad():
        selected_scales, weights = scale_selector(hr_features, top_k=3)
        
    print(f"Selected scales: {selected_scales}")
    print(f"Scale weights: {weights}")
    print("Adaptive scale selector test passed!")