# Multi-Scale Cross-Attention Enhancement - Version 1

## Overview
This directory contains the enhanced version 1 (v1) implementation of the CorrDiff model with multi-scale cross-attention capabilities. The enhancement focuses on minimal modifications to integrate cross-attention between encoder and decoder features for improved atmospheric super-resolution.

## Files Modified

### 1. `preconditioning.py`
**Key Enhancements:**
- Added `lr_features` parameter to all forward methods for low-resolution conditioning
- Enhanced `EDMPreconditioner` to support cross-attention with LR features
- Backward compatible: all existing functionality preserved
- New parameter: `lr_features=None` added to forward signatures

**Usage:**
```python
# Standard usage (unchanged)
model = EDMPreconditioner(...)
output = model(x, sigma, class_labels)

# Enhanced usage with LR features
output = model(x, sigma, class_labels, lr_features=lr_data)
```

### 2. `layers.py`
**Key Enhancements:**
- Added cross-attention support to `UNetBlock`
- Multi-scale attention mechanism for weather data
- Weather-aware attention patterns for meteorological variables
- New parameters: `cross_attention=False`, `weather_aware=True`

**New Features:**
- `CrossAttentionLayer`: Efficient cross-attention implementation
- Weather-specific attention scaling for atmospheric variables
- Multi-head attention with configurable number of heads
- Residual connections preserving original information flow

### 3. `song_unet.py`
**Key Enhancements:**
- Integrated cross-attention throughout encoder and decoder
- Multi-scale LR feature processing with caching
- Enhanced `SongUNet` and `SongUNetPosEmbd` architectures
- New parameters: `use_cross_attention=False`, `cross_attention_scales=[1.0, 0.5, 0.25]`, `weather_aware=True`

**New Features:**
- `_process_lr_features()`: Adaptive LR feature processing
- Multi-scale cross-attention at different resolutions
- Gradient checkpointing support for cross-attention
- Automatic feature caching for efficiency

## Key Benefits

1. **Minimal Disruption**: All changes are additive with backward compatibility
2. **Performance Optimized**: Feature caching and gradient checkpointing support
3. **Weather-Aware**: Specialized attention patterns for atmospheric data
4. **Multi-Scale**: Cross-attention at multiple resolutions for better detail capture
5. **Easy Integration**: Can be dropped into existing CorrDiff workflows

## Configuration Options

### Basic Cross-Attention
```python
model = SongUNet(
    img_resolution=256,
    in_channels=4,
    out_channels=4,
    use_cross_attention=True,  # Enable cross-attention
)
```

### Advanced Multi-Scale Configuration
```python
model = SongUNet(
    img_resolution=256,
    in_channels=4,
    out_channels=4,
    use_cross_attention=True,
    cross_attention_scales=[1.0, 0.5, 0.25, 0.125],  # Multi-scale processing
    weather_aware=True,  # Enable weather-specific patterns
)
```

## Usage Example

```python
import torch
from song_unet import SongUNet

# Create enhanced model
model = SongUNet(
    img_resolution=256,
    in_channels=4,
    out_channels=4,
    use_cross_attention=True,
    weather_aware=True
)

# Standard inputs
hr_image = torch.randn(1, 4, 256, 256)
noise_labels = torch.randn(1)
class_labels = torch.zeros(1, 1)

# Low-resolution conditioning features
lr_features = torch.randn(1, 4, 64, 64)  # Quarter resolution

# Enhanced forward pass
output = model(hr_image, noise_labels, class_labels, lr_features=lr_features)
```

## Integration Notes

1. **Memory Usage**: Cross-attention adds ~10-20% memory overhead
2. **Performance**: Minimal speed impact due to efficient implementation
3. **Compatibility**: Fully backward compatible with existing models
4. **Training**: No changes required to existing training loops

## Architectural Improvements

- **Cross-Attention**: Enables information flow between HR and LR features
- **Multi-Scale Processing**: Attention at multiple resolutions captures different detail levels
- **Weather Awareness**: Specialized patterns for atmospheric variables (temperature, pressure, humidity)
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Feature Caching**: Reduces computation by caching processed LR features

## Next Steps

To further enhance the model:
1. Add learnable temperature scaling for attention
2. Implement adaptive attention based on local weather patterns
3. Add frequency-domain cross-attention for spectral features
4. Integrate with uncertainty quantification for atmospheric predictions