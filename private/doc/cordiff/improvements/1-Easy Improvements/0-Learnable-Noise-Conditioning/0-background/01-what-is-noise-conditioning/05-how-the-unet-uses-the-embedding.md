# Part 5: How the UNet Uses the Noise Embedding

## Recap: Where We Are in the Pipeline

By this point, the noise level σ has been transformed through two stages:

```
σ (scalar) → log(σ)/4 (scalar) → Sinusoidal Embedding (128-dim vector)
```

Now this 128-dim vector needs to enter the UNet and **influence every layer**.
This document explains exactly how that happens.

## Stage 1: MLP Projection (128 → 512 dimensions)

Inside `SongUNet.forward()`, the 128-dim sinusoidal embedding is first projected
to a larger 512-dim vector through two linear layers with SiLU activation:

```python
# In SongUNet.forward():
emb = self.map_noise(noise_labels)          # PositionalEmbedding: (B,) → (B, 128)
emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
emb = silu(self.map_layer0(emb))            # Linear: (B, 128) → (B, 512)
emb = silu(self.map_layer1(emb))            # Linear: (B, 512) → (B, 512)
```

**Why project to 512?** The UNet blocks use 512-dimensional conditioning vectors.
The two MLP layers also give the network a chance to **learn** which aspects of
the noise embedding are most important — the fixed sinusoidal features are
transformed into task-specific features.

**Note:** If class labels or augmentation labels are provided, they are added
to `emb` before the MLP layers. This means the final `emb` vector encodes BOTH
the noise level AND any class conditioning.

```python
# Optional: add class label embedding
if self.map_label is not None:
    emb = emb + self.map_label(class_labels * sqrt(label_dim))

# Optional: add augmentation label embedding
if self.map_augment is not None:
    emb = emb + self.map_augment(augment_labels)
```

## Stage 2: FiLM Conditioning in Every UNet Block

The 512-dim `emb` vector is passed to **every UNet block** in both the encoder
and decoder. Inside each block, the embedding modulates the feature maps via
**FiLM** (Feature-wise Linear Modulation).

### What Is FiLM?

FiLM is a conditioning mechanism where an external signal (the noise embedding)
controls the **scale and shift** of internal feature maps:

```
Standard normalization:
    output = normalize(features)

FiLM conditioning:
    output = scale × normalize(features) + shift
    where (scale, shift) are derived from the noise embedding
```

The noise level literally **scales and shifts** every feature map in the network.

### The Actual Code

Inside `UNetBlock.forward()`:

```python
def forward(self, x, emb):
    # x shape: (B, C, H, W) — feature maps
    # emb shape: (B, 512) — noise embedding

    # Step 1: Normalize and convolve
    x = self.conv0(self.norm0(x))

    # Step 2: Derive scale and shift from noise embedding
    params = self.affine(emb)              # Linear: (B, 512) → (B, 2×C)
    params = params.unsqueeze(2).unsqueeze(3)  # → (B, 2×C, 1, 1) for broadcasting
    scale, shift = params.chunk(2, dim=1)  # Each: (B, C, 1, 1)

    # Step 3: Apply FiLM — modulate features with noise-dependent scale/shift
    x = silu(shift + self.norm1(x) * (scale + 1))
    #         ↑                        ↑
    #    noise-dependent bias     noise-dependent gain

    # Step 4: Second convolution + skip connection
    x = self.conv1(dropout(x))
    x = x + skip(orig)
    x = x * skip_scale

    return x
```

### What This Means Intuitively

At **low noise** (σ ≈ 0.01), the scale and shift values tell the network:
> "The input is almost clean. Make minimal changes. Fine-tune details."

At **high noise** (σ ≈ 80), different scale and shift values tell the network:
> "The input is pure noise. Ignore it. Generate structure from the conditioning."

The network learns completely different internal behaviors for different noise
levels — all controlled through these scale/shift parameters.

## The Full Signal Flow (Complete Picture)

```
σ = 2.7 (noise level for this training sample)
│
▼
c_noise = log(2.7) / 4 = 0.248               ← Fixed formula (THE TARGET)
│
▼
PositionalEmbedding(0.248)                    ← Sinusoidal: scalar → 128-dim
│ → [0.97, 0.25, -0.73, 0.88, ...]
│
▼
map_layer0: Linear(128 → 512) + SiLU         ← Learned projection
│
▼
map_layer1: Linear(512 → 512) + SiLU         ← Learned projection
│ → emb = [0.23, -0.87, 0.11, ...]  (512-dim)
│
▼
┌─────────── emb is broadcast to ALL UNet blocks ──────────┐
│                                                           │
▼                          ▼                          ▼
UNetBlock (level 0)    UNetBlock (level 1)    UNetBlock (level 2)
│                          │                          │
├─ affine(emb) → scale,shift  ← noise controls this block
├─ x = shift + norm(x) × (scale + 1)
├─ conv + dropout
└─ skip connection

... repeated for all ~40 blocks in encoder + decoder ...
│
▼
Output: denoised image
```

**Every single UNet block** receives the same `emb` vector and derives its own
scale/shift from it. This means:

- ~40 `affine` layers, each with its own weights
- Each block learns a **different response** to the noise level
- Early blocks might focus on low-frequency noise adjustment
- Late blocks might focus on high-frequency detail recovery

## Why This Architecture Matters for Learnable Noise Conditioning

The noise level influences the network through a **chain** of transformations:

```
σ → [log/4] → [sinusoidal] → [MLP] → [FiLM in every block]
     ↑            ↑              ↑           ↑
   FIXED        FIXED         LEARNED     LEARNED
```

The last two stages (MLP and FiLM) are already learned. They can theoretically
compensate for a suboptimal first stage. However:

1. **The sinusoidal embedding's effectiveness depends on its input range.** If
   `log/4` places values in a suboptimal range, the sinusoidal features may have
   poor discriminability, and no amount of downstream learning can fully recover.

2. **Information bottleneck.** The scalar c_noise is a bottleneck — all noise
   information must pass through this single number before being expanded. If the
   mapping from σ to this scalar loses information (e.g., by compressing important
   σ regions too much), it can't be recovered later.

3. **Gradient flow.** During training, gradients flow back through the entire
   chain. A learnable first stage gives the optimizer one more knob to turn,
   potentially making the whole chain easier to optimize.

## Summary

| Stage | What happens | Learned? | Dimensions |
|-------|-------------|----------|------------|
| Noise mapping | σ → c_noise scalar | ❌ Fixed (`log/4`) | 1 → 1 |
| Sinusoidal embed | c_noise → frequency features | ❌ Fixed | 1 → 128 |
| MLP projection | Features → conditioning vector | ✅ Learned | 128 → 512 |
| FiLM (per block) | Conditioning → scale + shift | ✅ Learned | 512 → 2×C |
| Feature modulation | scale × norm(x) + shift | — | — |

The learnable noise conditioning proposal adds learning to the **first stage**,
making the entire chain end-to-end learnable.

---

**Previous:** [Part 4: Sinusoidal Embeddings Explained](./04-sinusoidal-embeddings-explained.md)
**Next:** [Part 6: The Full Pipeline and What We Want to Change](./06-the-full-pipeline-and-what-we-want-to-change.md)
