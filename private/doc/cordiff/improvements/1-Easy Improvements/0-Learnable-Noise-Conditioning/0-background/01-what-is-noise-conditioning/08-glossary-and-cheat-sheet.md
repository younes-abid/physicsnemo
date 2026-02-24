# Part 8: Glossary and Cheat Sheet

## Quick Reference: The Noise Conditioning Pipeline

```
σ (noise level, scalar)
│
├── c_skip, c_out, c_in  ← preconditioning scalars (NOT changed)
│
▼
log(σ) / 4  +  MLP(log(σ))  =  c_noise  (scalar)
  ↑ fixed        ↑ learned       ↑ the noise label
                  ↑ starts at 0
│
▼
PositionalEmbedding(c_noise)  →  128-dim vector
│
▼
map_layer0 + SiLU  →  512-dim vector
map_layer1 + SiLU  →  512-dim vector  =  emb
│
▼
Every UNetBlock:  scale, shift = affine(emb)
                  features = SiLU(shift + norm(features) × (scale + 1))
```

## Glossary

### Core Concepts

| Term | Definition |
|------|-----------|
| **σ (sigma)** | The noise level — standard deviation of Gaussian noise added to the image. Range: 0.002 to 80 in CorrDiff. |
| **c_noise** | The noise label — a transformed version of σ that's fed to the neural network. Currently `log(σ)/4`. |
| **Noise conditioning** | The mechanism for telling the neural network what the noise level is, so it can adapt its denoising behavior. |
| **Diffusion model** | A neural network trained to remove noise from images. Generates new images by iteratively denoising from pure noise. |
| **EDM** | "Elucidating the Design Space of Diffusion-Based Generative Models" — the framework used by CorrDiff. Defines how σ maps to c_noise and the preconditioning scalars. |

### Architecture Components

| Term | Definition |
|------|-----------|
| **EDMPrecondSuperResolution** | The preconditioning wrapper around the UNet. Computes c_skip, c_out, c_in, c_noise from σ, then calls the UNet. Lives in `preconditioning.py`. |
| **SongUNet** | The U-Net backbone network that does the actual denoising. Takes the noisy image + c_noise and produces the denoised output. Lives in `song_unet.py`. |
| **PositionalEmbedding** | Converts a scalar (c_noise) into a 128-dim vector using sin/cos waves at multiple frequencies. Lives in `layers.py`. |
| **FourierEmbedding** | Alternative to PositionalEmbedding using random frequencies instead of geometric ones. Also in `layers.py`. |
| **UNetBlock** | A single block in the U-Net. Contains convolutions, normalization, and FiLM conditioning. Repeated ~40× in the encoder+decoder. |
| **FiLM** | Feature-wise Linear Modulation — the mechanism by which the noise embedding controls the UNet's behavior. Applies scale and shift to feature maps. |
| **map_noise** | The PositionalEmbedding instance inside SongUNet that converts c_noise to a 128-dim vector. |
| **map_layer0/1** | Two linear layers (with SiLU) inside SongUNet that project the 128-dim embedding to 512-dim. |
| **affine** | A linear layer inside each UNetBlock that converts the 512-dim embedding to scale+shift parameters. |

### Preconditioning Scalars

| Term | Formula | Purpose |
|------|---------|---------|
| **c_in** | `1 / √(σ_data² + σ²)` | Scales the input image to normalize its magnitude |
| **c_out** | `σ × σ_data / √(σ² + σ_data²)` | Scales the network output to the correct magnitude |
| **c_skip** | `σ_data² / (σ² + σ_data²)` | Skip connection weight — how much of the input to pass through |
| **c_noise** | `log(σ) / 4` | Noise label passed to the UNet (★ this is what we change) |
| **σ_data** | `0.5` (default) | Expected standard deviation of the training data |

### The Proposed Change

| Term | Definition |
|------|-----------|
| **Learnable noise conditioning** | Replacing the fixed `log(σ)/4` formula with `log(σ)/4 + MLP(log(σ))`, where the MLP is a tiny learned network. |
| **noise_embed** | The small MLP (1→64→1, 193 params) that learns the correction term δ(σ). |
| **c_noise_base** | The fixed baseline: `log(σ)/4`. Always computed, never changed. |
| **c_noise_delta** | The learned correction: output of the noise_embed MLP. Starts at 0 (zero init). |
| **Residual formulation** | `c_noise = c_noise_base + c_noise_delta`. Ensures we start at the EDM baseline and can only improve. |
| **Zero initialization** | The MLP's last layer is initialized with all-zero weights and bias, so it outputs 0 at the start of training. |

### General ML Terms

| Term | Definition |
|------|-----------|
| **MLP** | Multi-Layer Perceptron — a stack of linear layers with nonlinear activations. The simplest neural network. |
| **SiLU** | Sigmoid Linear Unit: `SiLU(x) = x × sigmoid(x)`. A smooth activation function used throughout CorrDiff. |
| **Linear layer** | A layer that computes `y = Wx + b`. The basic building block of neural networks. |
| **Activation function** | A nonlinear function applied element-wise between linear layers. Makes the network capable of learning non-linear relationships. |
| **Embedding** | Converting a simple input (like a scalar) into a rich high-dimensional vector. |
| **Sinusoidal embedding** | Using sin/cos waves at multiple frequencies to create embeddings. Same technique as position encoding in Transformers. |
| **Residual connection** | `output = input + f(input)`. Allows the network to learn "corrections" rather than full transformations. |

## Cheat Sheet: File Locations

| What | Where |
|------|-------|
| EDMPrecondSuperResolution | `physicsnemo/models/diffusion/preconditioning.py` |
| SongUNet | `physicsnemo/models/diffusion/song_unet.py` |
| PositionalEmbedding | `physicsnemo/models/diffusion/layers.py` |
| FourierEmbedding | `physicsnemo/models/diffusion/layers.py` |
| UNetBlock | `physicsnemo/models/diffusion/layers.py` |
| Training script | `examples/weather/corrdiff/train.py` |
| Generation script | `examples/weather/corrdiff/generate.py` |
| Loss function | `physicsnemo/models/diffusion/loss.py` (presumably) |

## Cheat Sheet: The Change at a Glance

**What:** Replace fixed `log(σ)/4` with `log(σ)/4 + MLP(log(σ))`

**Where:** `EDMPrecondSuperResolution` in `preconditioning.py`

**How much code:** ~15 lines (6 in `__init__`, 4 in `forward`, rest is imports)

**Parameters added:** 193 (0.0001% of model)

**Compute added:** ~0.00001%

**Risk:** Zero (starts at exact baseline due to zero init)

**Files changed:** 1

**Expected improvement:** 0–3% RMSE (most likely 0–0.5%)

## Cheat Sheet: Reading Order

For the full background, read in this order:

1. **[Part 1: What Is a Diffusion Model?](./01-what-is-a-diffusion-model.md)** — The big picture
2. **[Part 2: σ — The Noise Level](./02-sigma-the-noise-level.md)** — What σ is and why it spans 5 orders of magnitude
3. **[Part 3: Why Raw Numbers Are Bad Inputs](./03-why-raw-numbers-are-bad-inputs.md)** — Why we can't just pass σ directly
4. **[Part 4: Sinusoidal Embeddings Explained](./04-sinusoidal-embeddings-explained.md)** — How scalars become vectors
5. **[Part 5: How the UNet Uses the Embedding](./05-how-the-unet-uses-the-embedding.md)** — FiLM conditioning in every block
6. **[Part 6: The Full Pipeline and What We Want to Change](./06-the-full-pipeline-and-what-we-want-to-change.md)** — End-to-end view + the proposed change
7. **[Part 7: What Is an MLP?](./07-what-is-an-mlp.md)** — The tiny network we're adding
8. **[Part 8: Glossary and Cheat Sheet](./08-glossary-and-cheat-sheet.md)** — This file

For the implementation details, see the theory document:

- **[Learnable Noise Conditioning in CorrDiff — Where and How to Intervene](../../1-theory/01-learnable-noise-conditioning-in-corrdiff.md)**

---

## References

1. Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. "Elucidating the
   Design Space of Diffusion-Based Generative Models." NeurIPS.

2. Vaswani, A. et al., 2017. "Attention Is All You Need." NeurIPS.

3. Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A., 2018.
   "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI.

4. Mardani, M., Brenowitz, N., et al., 2023. "Generative Residual Diffusion
   Modeling for Km-scale Atmospheric Downscaling." arXiv:2309.15214.
