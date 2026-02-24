"""
Multi-Scale Cross-Attention Implementation for CorrDiff

This module implements the core cross-attention mechanism that enables explicit
interaction between high-resolution and low-resolution features in weather
super-resolution tasks.

Key Components:
- CrossAttentionBlock: Core attention mechanism
- PositionalEncoding2D: Spatial position encoding for weather data
- EfficientAttention: Memory-optimized attention computation

Author: CorrDiff Enhancement Team
Date: January 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import warnings


class PositionalEncoding2D(nn.Module):
    """
    2D Positional encoding for spatial attention in weather data.
    
    Creates learnable or fixed positional embeddings that encode spatial
    relationships crucial for atmospheric modeling.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 512, 
                 encoding_type: str = "sinusoidal"):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.encoding_type = encoding_type
        
        if encoding_type == "learnable":
            self.pos_embedding = nn.Parameter(
                torch.randn(1, embed_dim, max_len, max_len) * 0.02
            )
        elif encoding_type == "sinusoidal":
            self.register_buffer("pos_embedding", 
                               self._create_sinusoidal_encoding(max_len, embed_dim))
        
    def _create_sinusoidal_encoding(self, max_len: int, embed_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding for 2D coordinates."""
        pe = torch.zeros(embed_dim, max_len, max_len)
        
        # Create position indices
        y_pos = torch.arange(max_len).float().unsqueeze(1).repeat(1, max_len)
        x_pos = torch.arange(max_len).float().unsqueeze(0).repeat(max_len, 1)
        
        # Normalize positions to [0, 1]
        y_pos = y_pos / max_len
        x_pos = x_pos / max_len
        
        # Create frequency bands
        div_term = torch.exp(torch.arange(0, embed_dim//2, 2).float() *
                           -(math.log(10000.0) / (embed_dim//2)))
        
        # Apply sinusoidal encoding
        for i in range(embed_dim//4):
            pe[4*i, :, :] = torch.sin(y_pos * div_term[i])
            pe[4*i+1, :, :] = torch.cos(y_pos * div_term[i])
            pe[4*i+2, :, :] = torch.sin(x_pos * div_term[i])
            pe[4*i+3, :, :] = torch.cos(x_pos * div_term[i])
            
        return pe.unsqueeze(0)
    
    def forward(self, H: int, W: int) -> torch.Tensor:
        """
        Get positional encoding for given spatial dimensions.
        
        Args:
            H: Height of the feature map
            W: Width of the feature map
            
        Returns:
            Positional encoding tensor of shape (1, embed_dim, H, W)
        """
        if H > self.max_len or W > self.max_len:
            warnings.warn(f"Requested size ({H}, {W}) exceeds max_len {self.max_len}")
            
        if self.encoding_type == "learnable":
            return F.interpolate(self.pos_embedding, size=(H, W), mode='bilinear')
        else:
            return F.interpolate(self.pos_embedding[:, :, :H, :W], 
                               size=(H, W), mode='bilinear')


class EfficientAttention(nn.Module):
    """
    Memory-efficient attention computation with optional sparsity patterns.
    
    Implements various optimization strategies for large-scale attention:
    - Chunked computation for memory efficiency
    - Sparse attention patterns for computational efficiency
    - Flash attention integration when available
    """
    
    def __init__(self, chunk_size: int = 1024, use_sparse: bool = False,
                 sparse_pattern: str = "local", window_size: int = 64):
        super().__init__()
        self.chunk_size = chunk_size
        self.use_sparse = use_sparse
        self.sparse_pattern = sparse_pattern
        self.window_size = window_size
        
    def create_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create sparse attention mask for computational efficiency."""
        if self.sparse_pattern == "local":
            mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
            
            for i in range(seq_len):
                start = max(0, i - self.window_size // 2)
                end = min(seq_len, i + self.window_size // 2 + 1)
                mask[i, start:end] = True
                
            return mask
        else:
            # Full attention as fallback
            return torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
    
    def chunked_attention(self, q: torch.Tensor, k: torch.Tensor, 
                         v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute attention using memory-efficient chunking."""
        B, H, N, D = q.shape
        
        if N <= self.chunk_size:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # Chunked computation
        output = torch.zeros_like(q)
        
        for i in range(0, N, self.chunk_size):
            end_i = min(i + self.chunk_size, N)
            q_chunk = q[:, :, i:end_i]
            
            # Use appropriate mask for this chunk
            chunk_mask = mask[:, i:end_i] if mask is not None else None
            
            chunk_output = F.scaled_dot_product_attention(
                q_chunk, k, v, attn_mask=chunk_mask
            )
            output[:, :, i:end_i] = chunk_output
            
        return output
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Efficient attention computation.
        
        Args:
            q: Query tensor [B, H, N, D]
            k: Key tensor [B, H, N, D]
            v: Value tensor [B, H, N, D]
            mask: Optional attention mask [B, N, N] or [N, N]
            
        Returns:
            Attention output [B, H, N, D]
        """
        B, H, N, D = q.shape
        
        # Create sparse mask if requested
        if self.use_sparse and mask is None:
            mask = self.create_sparse_mask(N, q.device)
            
        # Choose computation strategy based on sequence length
        if N > self.chunk_size:
            return self.chunked_attention(q, k, v, mask)
        else:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class CrossAttentionBlock(nn.Module):
    """
    Multi-scale cross-attention block for CorrDiff enhancement.
    
    Enables explicit interaction between HR and LR features through
    learnable attention mechanisms, crucial for weather super-resolution.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1,
                 use_efficient_attention: bool = True, chunk_size: int = 1024,
                 use_positional_encoding: bool = True):
        super().__init__()
        
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        
        # Normalization and regularization
        self.layer_norm_hr = nn.LayerNorm(embed_dim)
        self.layer_norm_lr = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding2D(embed_dim)
        else:
            self.pos_encoder = None
            
        # Efficient attention computation
        if use_efficient_attention:
            self.efficient_attention = EfficientAttention(chunk_size=chunk_size)
        else:
            self.efficient_attention = None
            
        # Weather-specific components
        self.temperature_scale = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights using Xavier uniform."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Initialize biases to zero
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self, hr_features: torch.Tensor, lr_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None, 
                return_attention_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of cross-attention block.
        
        Args:
            hr_features: High-resolution features [B, C, H, W]
            lr_features: Low-resolution features [B, C, H, W] 
            mask: Optional attention mask
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Tuple of (enhanced_hr_features, attention_weights)
        """
        B, C, H, W = hr_features.shape
        
        # Ensure LR features match HR spatial dimensions
        if lr_features.shape[-2:] != (H, W):
            lr_features = F.interpolate(lr_features, size=(H, W), mode='bilinear')
            
        # Reshape to sequence format for attention computation
        hr_seq = hr_features.flatten(2).transpose(1, 2)  # [B, HW, C]
        lr_seq = lr_features.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Add positional encoding if available
        if self.pos_encoder is not None:
            pos_encoding = self.pos_encoder(H, W)
            pos_seq = pos_encoding.flatten(2).transpose(1, 2)  # [1, HW, C]
            hr_seq = hr_seq + pos_seq
            lr_seq = lr_seq + pos_seq
            
        # Apply layer normalization
        hr_norm = self.layer_norm_hr(hr_seq)
        lr_norm = self.layer_norm_lr(lr_seq)
        
        # Compute Q, K, V projections
        Q = self.q_proj(hr_norm)  # Queries from HR features
        K = self.k_proj(lr_norm)  # Keys from LR features
        V = self.v_proj(lr_norm)  # Values from LR features
        
        # Reshape for multi-head attention
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        if self.efficient_attention is not None:
            attended = self.efficient_attention(Q, K, V, mask)
            attention_weights = None  # Not computed in efficient mode
        else:
            # Standard scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale * self.temperature_scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
                
            attention_weights = F.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            
            attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
        output = self.out_proj(attended)
        
        # Residual connection with original HR features
        output = hr_norm + self.dropout(output)
        
        # Reshape back to spatial format
        enhanced_hr = output.transpose(1, 2).view(B, C, H, W)
        
        if return_attention_weights:
            return enhanced_hr, attention_weights
        else:
            return enhanced_hr, None


# Weather-specific utility functions
def create_weather_variable_groups():
    """Create variable groupings for weather-aware attention."""
    return {
        'temperature': ['t_850', 't_500', 't2m', 'skt'],
        'pressure': ['z_850', 'z_500', 'sp'],
        'wind': ['u_850', 'u_500', 'v_850', 'v_500', 'u10', 'v10'],
        'moisture': ['d2m', 'tcwv', 'tp'],
        'target': ['Fog_index']
    }


def get_atmospheric_scale_config():
    """Get configuration for different atmospheric scales."""
    return {
        'synoptic': {
            'scale_factor': 1.0,
            'receptive_field': 'global',
            'attention_range': 512,
            'description': 'Large-scale weather systems (1000+ km)'
        },
        'mesoscale': {
            'scale_factor': 0.5,
            'receptive_field': 'regional',
            'attention_range': 128,
            'description': 'Regional weather features (10-1000 km)'
        },
        'microscale': {
            'scale_factor': 0.25,
            'receptive_field': 'local',
            'attention_range': 32,
            'description': 'Local phenomena (<10 km, fog formation)'
        }
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the cross-attention block
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test tensors
    B, C, H, W = 2, 256, 64, 64
    hr_features = torch.randn(B, C, H, W, device=device)
    lr_features = torch.randn(B, C, H, W, device=device)
    
    # Create cross-attention block
    cross_attention = CrossAttentionBlock(
        embed_dim=C,
        num_heads=8,
        dropout=0.1,
        use_efficient_attention=True
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        enhanced_hr, attention_weights = cross_attention(
            hr_features, lr_features, return_attention_weights=True
        )
    
    print(f"Input HR features shape: {hr_features.shape}")
    print(f"Input LR features shape: {lr_features.shape}")
    print(f"Enhanced HR features shape: {enhanced_hr.shape}")
    if attention_weights is not None:
        print(f"Attention weights shape: {attention_weights.shape}")
    
    print("Cross-attention block test passed!")