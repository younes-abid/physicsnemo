# U-Net Architecture

## What is it?

The **U-Net** is the neural network architecture used inside most diffusion models. It takes a noisy image (and a timestep) as input and outputs a prediction (noise, score, or clean data) of the same spatial size.

Originally designed for medical image segmentation (Ronneberger et al., 2015), U-Net became the default backbone for diffusion models because of its ability to capture both **fine-grained details** and **global context**.

## The U Shape

The architecture gets its name from its U-shaped structure when drawn as a diagram:

```
Input (noisy image)                          Output (predicted noise)
    │                                              ↑
    ▼                                              │
┌─────────┐                                  ┌─────────┐
│ Conv 64  │ ──── skip connection ──────────→ │ Conv 64  │
└────┬────┘                                  └────┬────┘
     ▼                                              ↑
  ┌─────────┐                              ┌─────────┐
  │ Conv 128│ ──── skip connection ───────→ │ Conv 128│
  └────┬────┘                              └────┬────┘
       ▼                                          ↑
    ┌─────────┐                          ┌─────────┐
    │ Conv 256│ ──── skip connection ──→ │ Conv 256│
    └────┬────┘                          └────┬────┘
         ▼                                    ↑
      ┌──────────────────────────────────┐
      │         Bottleneck (512)          │
      └──────────────────────────────────┘

      ◄──── Encoder (downsampling) ────►◄──── Decoder (upsampling) ────►
```

## The Three Key Components

### 1. Encoder (Downsampling Path — Left Side of U)
- Progressively **reduces spatial resolution** (e.g., 64→32→16→8)
- Progressively **increases channel count** (e.g., 64→128→256→512)
- Each level applies convolutional blocks, then downsamples (stride-2 conv or pooling)
- Captures increasingly **abstract, global features**

### 2. Bottleneck (Bottom of U)
- The lowest resolution, highest channel representation
- Captures the most **global context** (overall structure, layout)
- Often includes attention layers for long-range dependencies

### 3. Decoder (Upsampling Path — Right Side of U)
- Progressively **increases spatial resolution** back to the original size
- Progressively **decreases channel count**
- Each level applies convolutional blocks, then upsamples (transposed conv or interpolation)
- Reconstructs **spatial details**

### The Magic Ingredient: Skip Connections

**Skip connections** directly link each encoder level to the corresponding decoder level (the horizontal arrows in the diagram). They concatenate (or add) the encoder features to the decoder features.

Why are they essential?
- **Preserve fine details**: The encoder captures fine spatial details that would be lost through the bottleneck. Skip connections pass them directly to the decoder.
- **Help gradient flow**: Gradients can flow directly through skip connections, making training easier.
- **Multi-scale information**: The decoder has access to both high-level (from bottleneck) and low-level (from skip connections) features simultaneously.

## Timestep Conditioning

In diffusion models, the network must know **what noise level** the input is at. The timestep $t$ (or noise level $\sigma$) is injected into the network, typically via:

### Sinusoidal Positional Embedding
The scalar timestep $t$ is transformed into a high-dimensional vector using sinusoidal functions (borrowed from Transformers):

$$\text{emb}(t) = [\sin(t \cdot \omega_1), \cos(t \cdot \omega_1), \sin(t \cdot \omega_2), \cos(t \cdot \omega_2), \ldots]$$

where $\omega_i$ are different frequencies. This gives the network a rich representation of the noise level.

### Injection via Adaptive Normalization
The timestep embedding is typically injected into each residual block via **adaptive group normalization** (AdaGN):

$$\text{AdaGN}(\mathbf{h}, t) = \gamma(t) \cdot \text{GroupNorm}(\mathbf{h}) + \beta(t)$$

where $\gamma(t)$ and $\beta(t)$ are learned linear projections of the timestep embedding. This modulates the features at every layer based on the noise level.

## Key Building Blocks

### Residual Blocks (ResBlocks)
Each level consists of one or more **residual blocks**:

```
Input h ──→ [GroupNorm → SiLU → Conv → GroupNorm → SiLU → Conv] ──→ + ──→ Output
   │                                                                  ↑
   └────────────────── residual connection ──────────────────────────┘
```

The residual connection ($h + f(h)$) makes it easier to learn small corrections rather than full transformations.

### Attention Layers
At lower resolutions (e.g., 16×16, 8×8), **self-attention layers** are added between residual blocks. They allow every spatial position to attend to every other position, capturing **long-range dependencies** (e.g., symmetry in faces, spatial patterns in weather fields).

Attention is computationally expensive ($O(n^2)$ where $n$ = number of spatial positions), so it's only used at low resolutions where $n$ is manageable.

### Group Normalization
Instead of Batch Normalization, diffusion U-Nets use **Group Normalization** — it's more stable for small batch sizes and works better for generative models.

## Input and Output

| | Shape | Description |
|---|-------|------------|
| **Input** | $(B, C_{\text{in}}, H, W)$ | Noisy image/data, $B$=batch, $C_{\text{in}}$=channels, $H \times W$=spatial |
| **Timestep** | $(B,)$ | Scalar timestep per sample → embedded to vector |
| **Output** | $(B, C_{\text{out}}, H, W)$ | Same spatial size as input; predicted noise/score/clean data |

$C_{\text{in}}$ and $C_{\text{out}}$ are typically the same (e.g., 3 for RGB images, or the number of weather channels).

## Conditional U-Net

For **conditional generation** (e.g., class-conditional image generation, or weather downscaling), additional information is injected:

- **Class label**: embedded and added to the timestep embedding
- **Text prompt**: encoded by a text encoder (e.g., CLIP), then injected via cross-attention layers
- **Low-resolution input** (super-resolution / downscaling): concatenated with the noisy input along the channel dimension

In CorrDiff-style weather models, the low-resolution conditioning $\mathbf{c}$ is typically concatenated with the noisy input:
$$\text{Input} = \text{concat}(\mathbf{x}_t, \mathbf{c}) \in \mathbb{R}^{(C_{\text{noise}} + C_{\text{cond}}) \times H \times W}$$

## Alternatives to U-Net

While U-Net dominates, newer architectures are emerging:

| Architecture | Used in | Key difference |
|-------------|---------|----------------|
| **U-Net** | DDPM, EDM, Stable Diffusion 1.x/2.x | The classic; CNN-based |
| **U-ViT** | UniDiffuser | Replace ResBlocks with Transformer blocks |
| **DiT** (Diffusion Transformer) | Stable Diffusion 3, FLUX | Pure Transformer, no convolutions |
| **ADM U-Net** | Guided Diffusion (Dhariwal & Nichol) | U-Net + classifier guidance |

> 💡 For scientific applications (weather, physics), U-Net remains the most common choice due to its strong inductive bias for spatially structured data.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | U-Net that predicts noise |
| $D_\theta(\mathbf{x}; \sigma)$ | U-Net as a denoiser (EDM notation) |
| $F_\theta(\cdot)$ | The raw network (before preconditioning in EDM) |
| ResBlock | Residual block with normalization and activation |
| AdaGN | Adaptive Group Normalization (timestep injection) |
| Self-attention | Attention layer within the U-Net |

## Knowledge Check ✅

1. Why is U-Net shaped like a "U"? What are the three main parts?
2. What role do skip connections play, and why are they essential for diffusion?
3. How is the timestep $t$ injected into the U-Net?
4. Why is attention used only at low resolutions?
5. What is the input and output shape of a diffusion U-Net? Are they the same spatial size?
6. In conditional diffusion (e.g., weather downscaling), how is the conditioning input typically provided?
7. What is one alternative architecture to U-Net that is gaining popularity?

---
*Previous: [23-reverse-process.md](23-reverse-process.md) · Next: [25-elbo.md](25-elbo.md)*
