# Detailed Architecture Design

## 1. Technical Architecture Overview

This document provides detailed technical specifications for implementing Multi-Scale Cross-Attention Enhancement in CorrDiff, including mathematical formulations, algorithmic details, and integration strategies.

## 2. Mathematical Foundation

### 2.1 Cross-Attention Formulation

#### Standard Attention Mechanism
For HR features H ∈ R^(B×C_h×H×W) and LR features L ∈ R^(B×C_l×H×W):

```
Q = H W_q                    # Query projection: R^(B×HW×d_k)
K = L W_k                    # Key projection: R^(B×HW×d_k)  
V = L W_v                    # Value projection: R^(B×HW×d_v)

Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

#### Multi-Scale Extension
For scale set S = {s₁, s₂, ..., s_n}, the multi-scale attention is:

```
H_enhanced = Σᵢ αᵢ × ↑[CrossAttention(↓_sᵢ(H), ↓_sᵢ(L))]

where:
- ↓_s(·): downsample to scale s
- ↑[·]: upsample to original resolution  
- αᵢ: learnable scale weights
```

### 2.2 Weather-Specific Attention

#### Variable-Grouped Attention
For meteorological variable groups G = {temperature, pressure, wind, moisture}:

```
H_enhanced = Σ_g∈G W_g × CrossAttention(H_g, L_g)

where H_g, L_g are features for variable group g
```

#### Spatial Scale Attention
Different atmospheric scales require different receptive fields:

```
Att_synoptic = CrossAttention(H, L, kernel_size=global)
Att_mesoscale = CrossAttention(H, L, kernel_size=regional)  
Att_microscale = CrossAttention(H, L, kernel_size=local)

H_final = α₁×Att_synoptic + α₂×Att_mesoscale + α₃×Att_microscale
```

## 3. Detailed Component Design

### 3.1 CrossAttentionBlock Architecture

#### Core Components
```python
class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projection layers
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Normalization and regularization
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Position encoding for spatial awareness
        self.pos_encoder = PositionalEncoding2D(embed_dim)
```

#### Forward Pass Algorithm
```python
def forward(self, hr_features, lr_features, mask=None):
    B, C, H, W = hr_features.shape
    
    # Reshape to sequence format: (B, HW, C)
    hr_seq = hr_features.flatten(2).transpose(1, 2)
    lr_seq = lr_features.flatten(2).transpose(1, 2)
    
    # Add positional encoding
    hr_seq = hr_seq + self.pos_encoder(H, W)
    lr_seq = lr_seq + self.pos_encoder(H, W)
    
    # Multi-head attention computation
    Q = self.q_proj(hr_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
    K = self.k_proj(lr_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
    V = self.v_proj(lr_seq).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
    
    # Scaled dot-product attention
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = self.dropout(attention_weights)
    
    attended = torch.matmul(attention_weights, V)
    
    # Concatenate heads and project
    attended = attended.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)
    output = self.out_proj(attended)
    
    # Residual connection and normalization
    output = self.layer_norm(hr_seq + output)
    
    # Reshape back to spatial format
    output = output.transpose(1, 2).view(B, C, H, W)
    
    return output, attention_weights
```

### 3.2 Multi-Scale Processing Module

#### Scale Pyramid Architecture
```python
class MultiScaleProcessor(nn.Module):
    def __init__(self, scales=[1.0, 0.5, 0.25], embed_dim=256):
        super().__init__()
        self.scales = scales
        self.attention_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim) for _ in scales
        ])
        self.scale_weights = nn.Parameter(torch.ones(len(scales)))
        self.fusion_conv = nn.Conv2d(embed_dim * len(scales), embed_dim, 1)
```

#### Multi-Scale Processing Algorithm
```python
def forward(self, hr_features, lr_features):
    B, C, H, W = hr_features.shape
    scale_outputs = []
    all_attention_weights = []
    
    for i, scale in enumerate(self.scales):
        # Compute target size for this scale
        scale_h, scale_w = int(H * scale), int(W * scale)
        
        # Downsample features
        if scale < 1.0:
            hr_scaled = F.interpolate(hr_features, size=(scale_h, scale_w), mode='bilinear')
            lr_scaled = F.interpolate(lr_features, size=(scale_h, scale_w), mode='bilinear')
        else:
            hr_scaled, lr_scaled = hr_features, lr_features
        
        # Apply cross-attention at this scale
        attended, attn_weights = self.attention_blocks[i](hr_scaled, lr_scaled)
        
        # Upsample back to original resolution
        if scale < 1.0:
            attended = F.interpolate(attended, size=(H, W), mode='bilinear')
        
        scale_outputs.append(attended)
        all_attention_weights.append(attn_weights)
    
    # Weighted fusion of multi-scale features
    weighted_outputs = [w * output for w, output in zip(self.scale_weights, scale_outputs)]
    fused_features = torch.cat(weighted_outputs, dim=1)
    
    # Final fusion convolution
    enhanced_features = self.fusion_conv(fused_features)
    
    return enhanced_features, all_attention_weights
```

### 3.3 Weather-Aware Attention Module

#### Variable Grouping Strategy
```python
class WeatherAwareAttention(nn.Module):
    def __init__(self, embed_dim, variable_config):
        super().__init__()
        self.variable_groups = variable_config
        
        # Separate attention for each variable group
        self.group_attentions = nn.ModuleDict({
            group: CrossAttentionBlock(embed_dim) 
            for group in self.variable_groups.keys()
        })
        
        # Cross-group interaction
        self.inter_group_attention = CrossAttentionBlock(embed_dim)
        
        # Weather physics constraints
        self.physics_constraint = PhysicsConstraintLayer()
```

#### Physics-Informed Processing
```python
def forward(self, hr_features, lr_features, variable_indices):
    group_outputs = {}
    
    # Process each variable group separately
    for group_name, var_indices in variable_indices.items():
        hr_group = hr_features[:, var_indices]
        lr_group = lr_features[:, var_indices]
        
        attended = self.group_attentions[group_name](hr_group, lr_group)
        group_outputs[group_name] = attended
    
    # Inter-group interactions (e.g., temperature-pressure coupling)
    combined_features = torch.cat(list(group_outputs.values()), dim=1)
    inter_group_attended = self.inter_group_attention(combined_features, combined_features)
    
    # Apply physics constraints
    constrained_features = self.physics_constraint(inter_group_attended)
    
    return constrained_features
```

## 4. Integration Strategy

### 4.1 UNetBlock Enhancement

#### Enhanced UNetBlock Design
```python
class EnhancedUNetBlock(UNetBlock):
    def __init__(self, in_channels, out_channels, emb_channels, 
                 use_cross_attention=False, cross_attention_config=None, **kwargs):
        super().__init__(in_channels, out_channels, emb_channels, **kwargs)
        
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            self.cross_attention = MultiScaleProcessor(
                scales=cross_attention_config.get('scales', [1.0, 0.5, 0.25]),
                embed_dim=out_channels
            )
            
            # Additional components for LR feature processing
            self.lr_projection = nn.Conv2d(
                cross_attention_config.get('lr_channels', 16), 
                out_channels, 1
            )
```

#### Forward Pass with Cross-Attention
```python
def forward(self, x, emb, lr_features=None, return_attention=False):
    # Standard UNet processing
    residual = x
    x = self.norm1(x)
    x = self.act(x)
    x = self.conv1(x)
    
    # Embed conditioning (time, noise level, etc.)
    if emb is not None:
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x = x + emb_out
    
    # Cross-attention enhancement
    attention_weights = None
    if self.use_cross_attention and lr_features is not None:
        # Project LR features to match dimensionality
        lr_projected = self.lr_projection(lr_features)
        
        # Apply multi-scale cross-attention
        x_attended, attention_weights = self.cross_attention(x, lr_projected)
        
        # Residual connection for attention
        x = x + x_attended
    
    # Continue standard processing
    x = self.norm2(x)
    x = self.act(x)
    x = self.dropout(x)
    x = self.conv2(x)
    x = x + residual
    
    if return_attention:
        return x, attention_weights
    return x
```

### 4.2 Scaling Function Enhancement

#### Enhanced Scaling Function
```python
class EnhancedScalingFunction:
    def __init__(self, embed_dim=256, use_attention_fusion=True):
        self.use_attention_fusion = use_attention_fusion
        
        if use_attention_fusion:
            self.attention_fusion = CrossAttentionBlock(embed_dim)
            self.feature_projection = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
    
    def __call__(self, x, img_lr, c_in):
        if not self.use_attention_fusion:
            # Fallback to original concatenation
            return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)
        
        # Scale high-resolution features
        hr_scaled = c_in * x
        
        # Apply cross-attention for adaptive fusion
        attended_hr, _ = self.attention_fusion(hr_scaled, img_lr)
        
        # Project and combine
        enhanced_hr = self.feature_projection(attended_hr)
        
        # Concatenate enhanced HR with original LR
        return torch.cat([enhanced_hr, img_lr.to(x.dtype)], dim=1)
```

## 5. Computational Optimization

### 5.1 Efficient Attention Implementation

#### Memory-Efficient Attention
```python
def efficient_attention(q, k, v, chunk_size=1024):
    """Memory-efficient attention computation using chunking"""
    B, H, N, D = q.shape
    
    if N <= chunk_size:
        # Standard computation for small sequences
        return F.scaled_dot_product_attention(q, k, v)
    
    # Chunked computation for large sequences
    output = torch.zeros_like(q)
    
    for i in range(0, N, chunk_size):
        end_i = min(i + chunk_size, N)
        q_chunk = q[:, :, i:end_i]
        
        # Compute attention for this chunk
        chunk_output = F.scaled_dot_product_attention(q_chunk, k, v)
        output[:, :, i:end_i] = chunk_output
    
    return output
```

#### Sparse Attention Pattern
```python
def create_sparse_attention_mask(seq_len, window_size=64, global_tokens=8):
    """Create sparse attention mask for computational efficiency"""
    mask = torch.zeros(seq_len, seq_len)
    
    # Local attention windows
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = 1
    
    # Global attention tokens
    mask[:global_tokens, :] = 1
    mask[:, :global_tokens] = 1
    
    return mask.bool()
```

### 5.2 Progressive Training Strategy

#### Resolution Scheduling
```python
class ProgressiveTrainingScheduler:
    def __init__(self, start_resolution=64, target_resolution=256, 
                 schedule_steps=10000):
        self.start_res = start_resolution
        self.target_res = target_resolution
        self.schedule_steps = schedule_steps
    
    def get_current_resolution(self, step):
        if step >= self.schedule_steps:
            return self.target_res
        
        progress = step / self.schedule_steps
        current_res = int(self.start_res + 
                         (self.target_res - self.start_res) * progress)
        
        # Ensure resolution is power of 2
        return 2 ** int(math.log2(current_res))
    
    def should_increase_resolution(self, step):
        current_res = self.get_current_resolution(step)
        prev_res = self.get_current_resolution(step - 1)
        return current_res > prev_res
```

## 6. Performance Analysis

### 6.1 Computational Complexity

#### Time Complexity
- **Standard Concatenation**: O(1)
- **Cross-Attention**: O(n²d) where n = H×W, d = embed_dim
- **Multi-Scale (3 scales)**: O(n² + (n/4)² + (n/16)²) ≈ O(1.3n²)
- **Optimized Sparse**: O(n×w×d) where w = window_size

#### Space Complexity
- **Attention Weights**: O(n²) per head
- **Multi-Head (8 heads)**: O(8n²)
- **Multi-Scale**: Additional O(0.3n²) for smaller scales

### 6.2 Memory Optimization Strategies

#### Gradient Checkpointing Integration
```python
def checkpoint_cross_attention(hr_features, lr_features, attention_module):
    """Apply gradient checkpointing to cross-attention computation"""
    return torch.utils.checkpoint.checkpoint(
        attention_module, hr_features, lr_features, use_reentrant=False
    )
```

#### Dynamic Attention Resolution
```python
def adaptive_attention_resolution(features, memory_budget):
    """Dynamically adjust attention resolution based on memory"""
    H, W = features.shape[-2:]
    max_resolution = int(math.sqrt(memory_budget / (64 * features.shape[1])))
    
    if H * W > max_resolution ** 2:
        scale_factor = max_resolution / math.sqrt(H * W)
        return F.interpolate(features, scale_factor=scale_factor)
    
    return features
```

## 7. Validation Framework

### 7.1 Component Testing

#### Unit Tests for Cross-Attention
```python
def test_cross_attention_forward():
    """Test cross-attention forward pass"""
    B, C, H, W = 2, 256, 32, 32
    attention = CrossAttentionBlock(C, num_heads=8)
    
    hr_features = torch.randn(B, C, H, W)
    lr_features = torch.randn(B, C, H, W)
    
    output, weights = attention(hr_features, lr_features)
    
    assert output.shape == hr_features.shape
    assert weights.shape == (B, 8, H*W, H*W)

def test_multi_scale_consistency():
    """Test multi-scale processing consistency"""
    processor = MultiScaleProcessor(scales=[1.0, 0.5], embed_dim=256)
    
    features = torch.randn(1, 256, 64, 64)
    output, _ = processor(features, features)
    
    assert output.shape == features.shape
    assert not torch.allclose(output, features)  # Should be different
```

#### Integration Tests
```python
def test_unet_integration():
    """Test enhanced UNet block integration"""
    block = EnhancedUNetBlock(
        in_channels=256, out_channels=256, emb_channels=512,
        use_cross_attention=True, 
        cross_attention_config={'lr_channels': 16, 'scales': [1.0, 0.5]}
    )
    
    x = torch.randn(2, 256, 32, 32)
    emb = torch.randn(2, 512)
    lr_features = torch.randn(2, 16, 32, 32)
    
    output = block(x, emb, lr_features)
    assert output.shape == x.shape
```

## Key Implementation Notes

1. **Modular Design**: Each component is independently testable and replaceable
2. **Backward Compatibility**: Enhanced modules gracefully degrade to original behavior
3. **Memory Efficiency**: Multiple optimization strategies for different hardware constraints
4. **Weather-Specific**: Incorporates domain knowledge for atmospheric modeling
5. **Scalable**: Architecture supports different resolutions and scales

## References to Implementation

- **Complete Code**: See [1-code/cross_attention.py](./1-code/cross_attention.py)
- **UNet Integration**: See [1-code/enhanced_unet_block.py](./1-code/enhanced_unet_block.py)  
- **Multi-Scale Module**: See [1-code/multiscale_processor.py](./1-code/multiscale_processor.py)
- **Testing Framework**: See [1-code/test_implementation.py](./1-code/test_implementation.py)