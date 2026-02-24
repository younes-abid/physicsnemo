# Learnable Noise Conditioning in CorrDiff — Where and How to Intervene

## 1. Current CorrDiff Noise Conditioning Flow (What Exists)

The noise level σ enters the model through `EDMPrecondSuperResolution.forward()`
in `physicsnemo/models/diffusion/preconditioning.py`:

```
EDMPrecondSuperResolution.forward(x, img_lr, sigma)
│
├── c_skip = σ_data² / (σ² + σ_data²)           ← output scaling
├── c_out  = σ × σ_data / √(σ² + σ_data²)       ← output scaling
├── c_in   = 1 / √(σ_data² + σ²)                ← input scaling
├── c_noise = sigma.log() / 4                     ← ★ THIS IS THE TARGET
│
├── arg = scaling_fn(x, img_lr, c_in)             ← concatenate scaled HR + LR
│
├── F_x = self.model(arg, c_noise.flatten(), ...) ← pass to SongUNet
│         │
│         └── SongUNet.forward(x, noise_labels, ...)
│             │
│             ├── emb = self.map_noise(noise_labels)  ← PositionalEmbedding
│             │         sinusoidal: c_noise → 128-dim vector
│             │
│             ├── emb = silu(map_layer0(emb))          ← MLP: 128 → 512
│             ├── emb = silu(map_layer1(emb))          ← MLP: 512 → 512
│             │
│             └── for each UNetBlock:
│                   scale, shift = linear(emb)
│                   x = silu(shift + norm(x) * (scale + 1))  ← FiLM
│
└── D_x = c_skip * x + c_out * F_x               ← EDM preconditioning
```

### The exact line to change

In `EDMPrecondSuperResolution.forward()` (line ~650 in `preconditioning.py`):

```python
c_noise = sigma.log() / 4    # ← Fixed formula. This is the intervention point.
```

This same formula also appears in:
- `EDMPrecond.forward()` (line ~460) — the non-super-resolution variant
- Both use the identical `sigma.log() / 4` pattern

### What c_noise becomes

After `c_noise` is computed, it is:
1. **Flattened** to shape `(B,)` — one scalar per batch element
2. **Passed to** `self.model(arg, c_noise.flatten(), ...)` as `noise_labels`
3. **Consumed by** `PositionalEmbedding.forward(noise_labels)` inside SongUNet

The `PositionalEmbedding` expects a 1D tensor of scalars. It produces a
`(B, 128)` tensor of sinusoidal features. This means our learnable mapping must
also output a **scalar per batch element** — the interface is unchanged.

## 2. The Proposed Change

### 2.1 What Changes

Replace the single line:
```python
c_noise = sigma.log() / 4
```

With:
```python
c_noise = c_noise_base + self.noise_embed(sigma.log().view(-1, 1)).view_as(c_noise_base)
```

Where `self.noise_embed` is a small MLP defined in `__init__`:
```python
self.noise_embed = nn.Sequential(
    nn.Linear(1, 64),
    nn.SiLU(),
    nn.Linear(64, 1),
)
# Initialize to zero output → starts at exact EDM baseline
nn.init.zeros_(self.noise_embed[2].weight)
nn.init.zeros_(self.noise_embed[2].bias)
```

### 2.2 Shape Analysis

```
sigma shape in forward():  (B, 1, 1, 1)   ← reshaped at entry

Current:
  sigma.log()             → (B, 1, 1, 1)
  sigma.log() / 4         → (B, 1, 1, 1)
  c_noise.flatten()       → (B,)           ← passed to SongUNet

Proposed:
  sigma.log()             → (B, 1, 1, 1)
  c_noise_base = log/4    → (B, 1, 1, 1)   ← fixed baseline
  sigma.log().view(-1, 1) → (B, 1)         ← reshape for Linear input
  noise_embed(...)        → (B, 1)         ← MLP output
  .view_as(c_noise_base)  → (B, 1, 1, 1)   ← restore original shape
  c_noise = base + delta  → (B, 1, 1, 1)   ← residual sum
  c_noise.flatten()       → (B,)           ← passed to SongUNet (unchanged)
```

The key detail: `nn.Linear(1, 64)` expects input shape `(B, 1)`, so we need
to reshape sigma from `(B, 1, 1, 1)` to `(B, 1)` before the MLP, then reshape
back. The final `.flatten()` call before passing to the UNet is already there
and handles the `(B, 1, 1, 1) → (B,)` conversion.

### 2.3 Initialization Strategy — Why Residual Formulation

**Critical:** The MLP should be initialized so that it outputs **zero** at the
start of training. This ensures:
- Old checkpoints produce identical results before any fine-tuning
- Training starts from the proven EDM baseline, not random noise conditioning
- The model can only improve (or stay the same), never regress

The **residual formulation**:
```python
log_sigma = sigma.log()
c_noise_base = log_sigma / 4                                          # baseline (fixed)
c_noise_delta = self.noise_embed(log_sigma.view(-1, 1))               # learned correction
c_noise = c_noise_base + c_noise_delta.view_as(c_noise_base)          # sum
```

With `noise_embed`'s last layer initialized to zero weights and zero bias,
`c_noise_delta` starts at exactly 0 for all inputs. So at step 0:
```
c_noise = log(σ)/4 + 0 = log(σ)/4    ← identical to EDM baseline
```

As training progresses, the MLP learns whatever correction improves the loss.

## 3. Impact Analysis

### 3.1 What Is NOT Affected

| Component | Affected? | Why |
|-----------|-----------|-----|
| SongUNet / SongUNetPosEmbd | ❌ No | Receives `c_noise.flatten()` — same interface |
| PositionalEmbedding | ❌ No | Receives scalar noise labels — same interface |
| FiLM conditioning in UNetBlock | ❌ No | Receives emb vector — same interface |
| ResidualLoss | ❌ No | Sigma sampling is in the loss, not preconditioning |
| Training loop (train.py) | ❌ No | Model API unchanged |
| Dataset / DataLoader | ❌ No | No data changes |
| Inference (generate.py) | ❌ No | Model API unchanged |
| Checkpoint format | ⚠️ Minor | New parameters in state_dict (see §3.2) |

### 3.2 Checkpoint Compatibility

Adding `self.noise_embed` introduces new parameters to the model's state dict:
```
noise_embed.0.weight  (64, 1)    ← 64 params
noise_embed.0.bias    (64,)      ← 64 params
noise_embed.2.weight  (1, 64)    ← 64 params
noise_embed.2.bias    (1,)       ← 1 param
                                   Total: 193 params
```

**Loading old checkpoints:** Old checkpoints won't have these keys. Two options:
1. Use `strict=False` when loading: `model.load_state_dict(state_dict, strict=False)`
2. Add a migration path that initializes the MLP to output zeros (equivalent to baseline)

**Loading new checkpoints in old code:** Will fail because the old model doesn't
have `noise_embed`. This is acceptable — the new code is a strict superset.

### 3.3 The c_noise Value Is Also Used for Preconditioning Scalars

In `EDMPrecondSuperResolution.forward()`, the four preconditioning constants are:

```python
c_skip  = sigma_data² / (σ² + sigma_data²)
c_out   = σ × sigma_data / √(σ² + sigma_data²)
c_in    = 1 / √(sigma_data² + σ²)
c_noise = sigma.log() / 4                        ← only this changes
```

**Important:** `c_skip`, `c_out`, and `c_in` are computed directly from `sigma`,
NOT from `c_noise`. They are **not affected** by this change. Only the noise
label passed to the UNet changes. This is by design — the EDM preconditioning
scalars have their own theoretical justification and should remain fixed.

### 3.4 Both EDMPrecond and EDMPrecondSuperResolution

The `c_noise = sigma.log() / 4` formula appears in both:

1. **`EDMPrecondSuperResolution`** — Used by CorrDiff for weather super-resolution
2. **`EDMPrecond`** — Used for unconditional/class-conditional generation

The change should be applied to `EDMPrecondSuperResolution` first (since that's
what CorrDiff uses). If successful, it can be extended to `EDMPrecond`.

### 3.5 VP, VE, and iDDPM Preconditioners

The other preconditioners use different noise-to-label mappings:
- `VPPrecond`: `c_noise = (M - 1) * sigma_inv(sigma)` — discrete, hard to make learnable
- `VEPrecond`: `c_noise = (0.5 * sigma).log()` — could benefit from same treatment
- `iDDPMPrecond`: `c_noise = M - 1 - round_sigma(sigma)` — discrete lookup, different approach

**Recommendation:** Only modify `EDMPrecondSuperResolution` for now. The other
preconditioners are not used by CorrDiff and have different design considerations.

## 4. Detailed Architecture of the Learnable Mapping

### 4.1 Option A: Simple MLP (Minimal)

```
log(σ) ──→ Linear(1, 64) ──→ SiLU ──→ Linear(64, 1) ──→ c_noise
```

- **Params:** 193
- **Pros:** Simplest possible change
- **Cons:** SiLU saturates for large negative inputs; no guarantee it starts at baseline

### 4.2 Option B: Residual MLP (Recommended)

```
log(σ) ──→ /4 ──→ c_noise_base
       │
       └──→ Linear(1, 64) ──→ SiLU ──→ Linear(64, 1) ──→ c_noise_delta
                                                            │
c_noise = c_noise_base + c_noise_delta ◄────────────────────┘
```

- **Params:** 193
- **Pros:** Starts exactly at EDM baseline; MLP only needs to learn a correction
- **Cons:** Slightly more code (2 extra lines)

### 4.3 Option C: Multi-Output MLP (Advanced — NOT recommended for v-1)

Instead of outputting a single scalar, output a vector that directly replaces
the sinusoidal embedding:

```
log(σ) ──→ Linear(1, 64) ──→ SiLU ──→ Linear(64, 128) ──→ emb
```

This **skips** the PositionalEmbedding entirely. More expressive but:
- **Much larger change** (bypasses map_noise entirely)
- **More parameters** (8,320)
- **Higher risk** — removes the theoretically motivated sinusoidal structure
- **Harder to initialize** to match the baseline

**Not recommended for v-1.**

### 4.4 Recommendation: Option B (Residual MLP)

The residual approach is the safest. It:
1. Cannot perform worse than baseline (initialized to zero correction)
2. Preserves the EDM theoretical foundation
3. Gives the optimizer a clear optimization landscape (learn a small δ)
4. Is trivially reversible (set MLP weights to zero = exact baseline)

## 5. Where Exactly in the Code

### 5.1 File: `physicsnemo/models/diffusion/preconditioning.py`

**Class:** `EDMPrecondSuperResolution`

**In `__init__`** — add after `self.scaling_fn = self._scaling_fn`:
```python
import torch.nn as nn

# Learnable noise-level mapping (residual formulation)
self.noise_embed = nn.Sequential(
    nn.Linear(1, 64),
    nn.SiLU(),
    nn.Linear(64, 1),
)
# Initialize to zero output → starts at exact EDM baseline
nn.init.zeros_(self.noise_embed[2].weight)
nn.init.zeros_(self.noise_embed[2].bias)
```

**In `forward`** — replace `c_noise = sigma.log() / 4` with:
```python
log_sigma = sigma.log()
c_noise_base = log_sigma / 4
c_noise_delta = self.noise_embed(log_sigma.view(-1, 1)).view_as(c_noise_base)
c_noise = c_noise_base + c_noise_delta
```

### 5.2 No Other Files Need Changes

The beauty of this improvement is its isolation:

| File | Change needed |
|------|---------------|
| `preconditioning.py` | ✅ `EDMPrecondSuperResolution.__init__` and `.forward()` |
| `song_unet.py` | ❌ None |
| `layers.py` | ❌ None |
| `loss.py` | ❌ None |
| `train.py` | ❌ None |
| `generate.py` | ❌ None |
| Config YAML | ❌ None (unless we want a flag to enable/disable) |

## 6. Optional: Config Flag to Enable/Disable

For A/B testing, we could add a config flag:

```yaml
model:
  model_args:
    learnable_noise_conditioning: true  # default: false for backward compat
```

And in `EDMPrecondSuperResolution.__init__`:
```python
self.learnable_noise_conditioning = model_kwargs.pop("learnable_noise_conditioning", False)
if self.learnable_noise_conditioning:
    self.noise_embed = nn.Sequential(...)
```

And in `forward()`:
```python
if self.learnable_noise_conditioning:
    c_noise = c_noise_base + self.noise_embed(log_sigma.view(-1, 1)).view_as(...)
else:
    c_noise = sigma.log() / 4  # original EDM formula
```

This allows clean A/B experiments without code changes.

## 7. Monitoring and Diagnostics

During training, log these values to understand what the MLP learns:

1. **c_noise_delta statistics:** Log mean, std, min, max of the correction term.
   If it stays near zero, the MLP isn't finding a better mapping. If it diverges,
   something is wrong.

2. **Learned mapping visualization:** Periodically plot the learned function
   `f(log(σ)) = log(σ)/4 + noise_embed(log(σ))` vs the baseline `log(σ)/4`.
   This shows what the network has learned about noise-level representation.

3. **MLP gradient norms:** If gradients through the MLP are tiny, the model
   isn't finding the noise conditioning to be a bottleneck. If they're large,
   it's actively learning a different mapping.

## 8. Expected Outcomes

### Optimistic scenario (10% chance)
The MLP learns a meaningfully different mapping that improves RMSE by 1-3%.
This would suggest that weather data has different noise-sensitivity from natural
images, and the `log/4` formula is suboptimal for this domain.

### Likely scenario (70% chance)
The MLP learns a very small correction (|c_noise_delta| < 0.1). Metrics improve
by 0-0.5%. The improvement is not statistically significant but doesn't hurt.

### Worst case scenario (20% chance)
The MLP doesn't help at all. c_noise_delta stays at zero. No regression from
baseline since we initialized to zero. Remove the MLP and move on.

In all cases, the cost is minimal: ~193 extra parameters, negligible compute,
and the change is fully reversible.

---

## References

1. Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. "Elucidating the
   Design Space of Diffusion-Based Generative Models." NeurIPS.
   — *Derives the `log(σ)/4` formula; Table 1 compares preconditioning schemes*

2. Nichol, A.Q. and Dhariwal, P., 2021. "Improved Denoising Diffusion
   Probabilistic Models." ICML.
   — *Learned variance schedules; motivation for learnable noise handling*

3. Chen, T., Zhang, R., and Hinton, G., 2023. "Analog Bits: Generating Discrete
   Data using Diffusion Models with Self-Conditioning." ICLR.
   — *Learned noise-level transformations for non-standard data*

4. Mardani, M., Brenowitz, N., et al., 2023. "Generative Residual Diffusion
   Modeling for Km-scale Atmospheric Downscaling." arXiv:2309.15214.
   — *CorrDiff paper; uses EDM preconditioning with fixed `log/4`*
