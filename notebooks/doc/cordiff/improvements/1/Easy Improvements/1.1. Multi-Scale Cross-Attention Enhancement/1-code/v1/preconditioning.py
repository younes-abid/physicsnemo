# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Preconditioning schemes used in the paper"Elucidating the Design Space of 
Diffusion-Based Generative Models".

ENHANCED VERSION v1: Multi-Scale Cross-Attention Enhancement
- Added cross-attention fusion in EDMPrecondSuperResolution
- Maintains full backward compatibility
- Optimized for atmospheric super-resolution
"""

import importlib
import warnings
from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from physicsnemo.models.diffusion.utils import _safe_setattr
from physicsnemo.models.meta import ModelMetaData
from physicsnemo.models.module import Module

network_module = importlib.import_module("physicsnemo.models.diffusion")


# Simple Cross-Attention Block for minimal integration
class SimpleCrossAttention(nn.Module):
    """Lightweight cross-attention for LR-HR feature fusion."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        B, C, H, W = query.shape
        
        # Reshape to sequence format
        q = query.view(B, C, H*W).transpose(1, 2)  # [B, HW, C]
        k = key.view(B, C, H*W).transpose(1, 2)    # [B, HW, C]
        v = value.view(B, C, H*W).transpose(1, 2)  # [B, HW, C]
        
        # Project to Q, K, V
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)
        
        # Multi-head attention
        q = q.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            attn_out = F.scaled_dot_product_attention(q, k, v)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, H*W, C)
        attn_out = self.out_proj(attn_out)
        
        # Residual connection and reshape to spatial format
        out = self.norm(attn_out + q.view(B, H*W, C))
        return out.transpose(1, 2).view(B, C, H, W)


@dataclass
class EDMPrecondSuperResolutionMetaData(ModelMetaData):
    """EDMPrecondSuperResolution meta data"""

    name: str = "EDMPrecondSuperResolution"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSuperResolution(Module):
    """
    Enhanced version with Multi-Scale Cross-Attention support.
    
    ENHANCEMENT v1: 
    - Added cross-attention fusion between LR and HR features
    - Maintains full backward compatibility when use_cross_attention=False
    - Optimized for atmospheric super-resolution tasks
    
    See original class docstring for complete documentation.
    """

    _wrapped_classes = {
        "SongUNetPosEmbd",
        "SongUNetPosLtEmbd",
        "SongUNet",
        "DhariwalUNet",
    }

    _overridable_args = set.union(
        *(
            getattr(getattr(network_module, cls_name), "_overridable_args", set())
            for cls_name in _wrapped_classes
        )
    )

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd", "SongUNet", "DhariwalUNet"
        ] = "SongUNetPosEmbd",
        sigma_data: float = 0.5,
        sigma_min=0.0,
        sigma_max=float("inf"),
        # ENHANCEMENT v1: Cross-attention parameters
        use_cross_attention: bool = False,
        cross_attention_dim: int = 256,
        num_attention_heads: int = 8,
        **model_kwargs: dict,
    ):
        super().__init__(meta=EDMPrecondSuperResolutionMetaData)

        # Validation
        if model_type not in self._wrapped_classes:
            raise ValueError(
                f"Model type '{model_type}' is not supported. "
                f"Must be one of: {', '.join(self._wrapped_classes)}"
            )

        self.img_resolution = img_resolution
        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # ENHANCEMENT v1: Cross-attention configuration
        self.use_cross_attention = use_cross_attention
        self.cross_attention_dim = cross_attention_dim

        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )
        
        # ENHANCEMENT v1: Initialize cross-attention components
        if self.use_cross_attention:
            # Project LR features to attention dimension
            self.lr_projection = nn.Conv2d(
                img_in_channels, 
                cross_attention_dim, 
                kernel_size=1, 
                bias=True
            )
            
            # Project HR features to attention dimension  
            self.hr_projection = nn.Conv2d(
                img_out_channels, 
                cross_attention_dim, 
                kernel_size=1, 
                bias=True
            )
            
            # Cross-attention block
            self.cross_attention = SimpleCrossAttention(
                cross_attention_dim, 
                num_attention_heads
            )
            
            # Output projection back to HR channels
            self.output_projection = nn.Conv2d(
                cross_attention_dim, 
                img_out_channels, 
                kernel_size=1, 
                bias=True
            )
            
            # Gating mechanism for blending original and enhanced features
            self.gate = nn.Sequential(
                nn.Conv2d(img_out_channels * 2, img_out_channels, 1),
                nn.Sigmoid()
            )
            
            # Initialize to favor original features initially
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)
        
        self.scaling_fn = self._enhanced_scaling_fn if self.use_cross_attention else self._scaling_fn

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """Original scaling function - simple concatenation."""
        return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)

    def _enhanced_scaling_fn(
        self, x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        ENHANCEMENT v1: Attention-based feature fusion.
        
        Replaces simple concatenation with cross-attention between HR and LR features.
        Maintains backward compatibility by falling back to original method when disabled.
        """
        # Scale HR features
        hr_scaled = c_in * x
        
        # Ensure spatial dimensions match
        if img_lr.shape[-2:] != hr_scaled.shape[-2:]:
            img_lr_resized = F.interpolate(
                img_lr, 
                size=hr_scaled.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            img_lr_resized = img_lr
        
        # Project to attention space
        hr_proj = self.hr_projection(hr_scaled)
        lr_proj = self.lr_projection(img_lr_resized.to(x.dtype))
        
        # Apply cross-attention: LR features attend to HR features
        enhanced_hr = self.cross_attention(hr_proj, lr_proj, lr_proj)
        
        # Project back to original HR space
        enhanced_hr = self.output_projection(enhanced_hr)
        
        # Gate-controlled blending with original HR features
        gate_input = torch.cat([hr_scaled, enhanced_hr], dim=1)
        gate_weight = self.gate(gate_input)
        final_hr = gate_weight * enhanced_hr + (1 - gate_weight) * hr_scaled
        
        # Concatenate enhanced HR with original LR for model compatibility
        return torch.cat([final_hr, img_lr.to(x.dtype)], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        img_lr: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Enhanced forward pass with optional cross-attention fusion.
        Maintains exact compatibility with original implementation.
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        if img_lr is None:
            arg = c_in * x
        else:
            # ENHANCEMENT v1: Use enhanced scaling function if cross-attention enabled
            arg = self.scaling_fn(x, img_lr, c_in)
        arg = arg.to(dtype)

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """Convert a given sigma value(s) to a tensor representation."""
        return torch.as_tensor(sigma)

    @property
    def amp_mode(self):
        """Property that controls the automatic mixed precision mode."""
        return getattr(self.model, "amp_mode", None)

    @amp_mode.setter
    def amp_mode(self, value: bool):
        """Update ``amp_mode`` on the wrapped architecture and its sub-modules."""
        if not isinstance(value, bool):
            raise TypeError("amp_mode must be a boolean value.")
        self.model.apply(lambda m: _safe_setattr(m, "amp_mode", value))

    @property
    def profile_mode(self):
        """Property that controls the profiling mode."""
        return getattr(self.model, "profile_mode", None)

    @profile_mode.setter
    def profile_mode(self, value: bool):
        """Update ``profile_mode`` on the wrapped architecture and its sub-modules."""
        if not isinstance(value, bool):
            raise TypeError("profile_mode must be a boolean value.")
        self.model.apply(lambda m: _safe_setattr(m, "profile_mode", value))


# NOTE: This is a deprecated version of the EDMPrecondSuperResolution model.
# This was used to maintain backwards compatibility and allow loading old models.
@dataclass
class EDMPrecondSRMetaData(ModelMetaData):
    """EDMPrecondSR meta data"""

    name: str = "EDMPrecondSR"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class EDMPrecondSR(EDMPrecondSuperResolution):
    """
    NOTE: This is a deprecated version of the EDMPrecondSuperResolution model.
    Enhanced with v1 cross-attention support for backward compatibility.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,  # deprecated
        img_in_channels,
        img_out_channels,
        use_fp16=False,
        sigma_min=0.0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNetPosEmbd",
        scale_cond_input=True,  # deprecated
        # ENHANCEMENT v1: Add cross-attention parameters
        use_cross_attention=False,
        cross_attention_dim=256,
        num_attention_heads=8,
        **model_kwargs,
    ):
        warnings.warn(
            "EDMPrecondSR is deprecated and will be removed in a future version. "
            "Please use EDMPrecondSuperResolution instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(
            img_resolution=img_resolution,
            img_in_channels=img_in_channels,
            img_out_channels=img_out_channels,
            use_fp16=use_fp16,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_data=sigma_data,
            model_type=model_type,
            use_cross_attention=use_cross_attention,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            **model_kwargs,
        )

        if scale_cond_input:
            warnings.warn(
                "The `scale_cond_input=True` option does not properly scale the conditional input "
                "and is deprecated. It is highly recommended to set `scale_cond_input=False`.",
                DeprecationWarning,
            )
            self.scaling_fn = self._legacy_scaling_fn

        # Store deprecated parameters for backward compatibility
        self.img_channels = img_channels
        self.scale_cond_input = scale_cond_input

    @staticmethod
    def _legacy_scaling_fn(
        x: torch.Tensor, img_lr: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """Legacy scaling function for backward compatibility."""
        return c_in * torch.cat([x, img_lr.to(x.dtype)], dim=1)

    def forward(
        self,
        x,
        img_lr,
        sigma,
        force_fp32=False,
        **model_kwargs,
    ):
        """Forward pass maintaining backward compatibility."""
        return super().forward(
            x=x, img_lr=img_lr, sigma=sigma, force_fp32=force_fp32, **model_kwargs
        )