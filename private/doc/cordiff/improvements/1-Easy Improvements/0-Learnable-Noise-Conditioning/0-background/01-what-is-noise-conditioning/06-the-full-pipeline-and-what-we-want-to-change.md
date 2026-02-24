# Part 6: The Full Pipeline and What We Want to Change

## The Complete Noise Conditioning Pipeline

Let's put together everything from Parts 1–5 into one end-to-end view. This is
the exact signal flow in CorrDiff when a training sample with noise level σ = 2.7
is processed:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    EDMPrecondSuperResolution.forward()              │
│                                                                     │
│  Inputs: x (noisy HR image), img_lr (low-res conditioning), σ=2.7  │
│                                                                     │
│  ┌─── Preconditioning scalars (from σ directly) ──────────────┐    │
│  │  c_skip = 0.5² / (2.7² + 0.5²) = 0.033                   │    │
│  │  c_out  = 2.7 × 0.5 / √(2.7² + 0.5²) = 0.492            │    │
│  │  c_in   = 1 / √(0.5² + 2.7²) = 0.365                     │    │
│  │  (These are FIXED math — not affected by our change)       │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── Noise label (what we want to change) ───────────────────┐    │
│  │                                                             │    │
│  │  c_noise = log(2.7) / 4 = 0.248    ← CURRENT (fixed)      │    │
│  │                                                             │    │
│  │  c_noise = log(2.7)/4 + MLP(log(2.7))  ← PROPOSED         │    │
│  │          = 0.248 + δ                     (learnable)        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌─── Input preparation ──────────────────────────────────────┐    │
│  │  scaled_x  = c_in × x           (attenuate noisy image)   │    │
│  │  arg = cat(scaled_x, img_lr)     (concat HR + LR channels) │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  F_x = SongUNet(arg, c_noise=0.248, class_labels)                  │
│  │                                                                  │
│  │  ┌─── Inside SongUNet ────────────────────────────────────┐     │
│  │  │                                                         │     │
│  │  │  emb = PositionalEmbedding(0.248)    → 128-dim vector  │     │
│  │  │  emb = swap_sin_cos(emb)                                │     │
│  │  │  emb = emb + map_label(class_labels) (if provided)     │     │
│  │  │  emb = SiLU(map_layer0(emb))         → 512-dim vector  │     │
│  │  │  emb = SiLU(map_layer1(emb))         → 512-dim vector  │     │
│  │  │                                                         │     │
│  │  │  ┌── Encoder (with emb conditioning) ──────────────┐   │     │
│  │  │  │  for each block:                                 │   │     │
│  │  │  │    scale, shift = affine(emb)                    │   │     │
│  │  │  │    x = SiLU(shift + norm(x) × (scale + 1))     │   │     │
│  │  │  │    x = conv(dropout(x)) + skip(orig)            │   │     │
│  │  │  └──────────────────────────────────────────────────┘   │     │
│  │  │                                                         │     │
│  │  │  ┌── Decoder (with emb conditioning + skip conns) ─┐   │     │
│  │  │  │  (mirror of encoder, same FiLM mechanism)        │   │     │
│  │  │  └──────────────────────────────────────────────────┘   │     │
│  │  │                                                         │     │
│  │  └── output: F_x (raw network prediction)                 │     │
│  │                                                                  │
│  └──────────────────────────────────────────────────────────────────│
│                                                                     │
│  D_x = c_skip × x + c_out × F_x    ← Final EDM combination       │
│  return D_x                          (denoised prediction)          │
└─────────────────────────────────────────────────────────────────────┘
```

## What Exactly We Want to Change

### The Current Code (one line)

In `EDMPrecondSuperResolution.forward()`:

```python
c_noise = sigma.log() / 4
```

### The Proposed Code (four lines)

```python
log_sigma = sigma.log()
c_noise_base = log_sigma / 4                                           # original
c_noise_delta = self.noise_embed(log_sigma.view(-1, 1)).view_as(c_noise_base)  # learned
c_noise = c_noise_base + c_noise_delta                                 # combined
```

Plus in `__init__`, a small MLP:

```python
self.noise_embed = nn.Sequential(
    nn.Linear(1, 64),
    nn.SiLU(),
    nn.Linear(64, 1),
)
nn.init.zeros_(self.noise_embed[2].weight)   # output layer starts at zero
nn.init.zeros_(self.noise_embed[2].bias)     # so delta starts at zero
```

### What This Changes in the Pipeline

```
BEFORE:                              AFTER:

σ ──→ log(σ)/4 ──→ sinusoidal       σ ──→ log(σ)/4 + MLP(log(σ)) ──→ sinusoidal
      │                                    │           │
      │ fixed scalar                       │ fixed     │ learned correction
      │                                    │           │ (starts at 0)
      ▼                                    ▼           ▼
    c_noise = 0.248                      c_noise = 0.248 + δ
```

Everything downstream (sinusoidal embedding, MLP projection, FiLM conditioning)
is completely unchanged. The UNet still receives a scalar noise label — it just
might be a slightly different scalar than `log(σ)/4`.

## Why Residual (Additive) Formulation?

We use `c_noise = baseline + delta` rather than `c_noise = MLP(log(σ))` for
three critical reasons:

### Reason 1: Safe Initialization

With the last layer initialized to zero:
```
At step 0:  c_noise = log(σ)/4 + 0 = log(σ)/4    (identical to current code)
At step N:  c_noise = log(σ)/4 + δ(σ)             (δ is whatever helps)
```

The model starts at the **exact EDM baseline**. It can only get better (or stay
the same). There's zero risk of degrading performance.

### Reason 2: Easy Optimization

The MLP only needs to learn a **small correction** to an already-good formula.
Learning "the optimal noise mapping from scratch" is much harder than learning
"how to slightly improve `log(σ)/4`."

Think of it like this:
- **From scratch:** "What's the best function f(σ)?" → infinite search space
- **Residual:** "What small δ(σ) makes log(σ)/4 + δ(σ) better?" → easy

### Reason 3: Interpretability

After training, we can plot δ(σ) and immediately see what the network learned:
- If δ ≈ 0 everywhere → `log(σ)/4` was already optimal
- If δ > 0 for small σ → the model wants to "spread out" low-noise labels
- If δ < 0 for large σ → the model wants to "compress" high-noise labels

This tells us something about the data.

## What the MLP Could Learn — Scenarios

### Scenario A: Identity (δ ≈ 0)

```
Learned mapping ≈ log(σ)/4
Interpretation: EDM's formula is already optimal for weather data
Result: No improvement, no regression
```

### Scenario B: Different Scaling

```
Learned mapping ≈ log(σ)/3.5
(equivalent to δ(σ) ≈ log(σ)/28)
Interpretation: Weather data benefits from a wider noise label range
```

### Scenario C: Shifted Operating Point

```
Learned mapping ≈ log(σ)/4 + 0.3
(equivalent to δ(σ) ≈ 0.3, constant)
Interpretation: The sinusoidal embedding works better when centered differently
```

### Scenario D: Non-Linear Correction

```
Learned mapping ≈ log(σ)/4 + tanh(log(σ))×0.2
Interpretation: The model wants to expand certain σ regions and compress others
```

Scenario D is the most interesting — it would mean that weather data has a
fundamentally different noise-sensitivity profile than natural images, and the
MLP discovered it automatically.

## The Size of the Change

### Parameters Added

```
noise_embed.0.weight:  (64, 1)   = 64 parameters
noise_embed.0.bias:    (64,)     = 64 parameters
noise_embed.2.weight:  (1, 64)   = 64 parameters
noise_embed.2.bias:    (1,)      = 1 parameter
                                   ─────────────
                        Total:     193 parameters
```

For context, a typical CorrDiff model has **~100–200 million** parameters.
The MLP adds 0.0001%.

### Compute Added

Per forward pass:
- One matrix multiply: (B, 1) × (1, 64) = 64B multiplications
- One SiLU activation: 64B operations
- One matrix multiply: (B, 64) × (64, 1) = 64B multiplications
- One addition: B operations

Total: ~193B floating point operations per forward pass.
For comparison, the UNet itself does **billions** of operations.
The overhead is unmeasurable.

### Memory Added

193 × 4 bytes (float32) = 772 bytes ≈ 0.75 KB.
Plus optimizer states (Adam): 193 × 2 × 4 = 1,544 bytes ≈ 1.5 KB.
Total: ~2.3 KB. The model itself uses ~400–800 MB.

## Summary

| Aspect | Current | Proposed |
|--------|---------|----------|
| Noise mapping | `log(σ)/4` (fixed) | `log(σ)/4 + MLP(log(σ))` (learnable) |
| Parameters | 0 | 193 (0.0001% of model) |
| Compute overhead | 0 | ~0.00001% |
| Memory overhead | 0 | ~2.3 KB |
| Risk | Baseline | Zero (residual init = identical start) |
| Files changed | — | 1 (preconditioning.py) |
| Lines changed | — | ~15 |
| Downstream impact | — | None (UNet, loss, data all unchanged) |

---

**Previous:** [Part 5: How the UNet Uses the Embedding](./05-how-the-unet-uses-the-embedding.md)
**Next:** [Part 7: What Is an MLP?](./07-what-is-an-mlp.md)
