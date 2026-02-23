# CorrDiff Model Improvements: Roadmap & NVIDIA Discussion Notes

## Overview

This document outlines improvements for the CorrDiff (Conditional Residual Diffusion) model,
organized into two categories:

1. **Training Practice Gaps** — Standard diffusion training techniques that are **missing**
   from the current implementation and should be addressed first.
2. **Architecture Improvements** — Model-level enhancements, honestly assessed for difficulty,
   SOTA grounding, and integration risk.

**Base Implementation:** CorrDiff uses `EDMPrecondSuperResolution` wrapping `SongUNetPosEmbd`,
trained via `ResidualLoss` in `examples/weather/corrdiff/train.py`.

> **Key principle:** Fix what's missing before adding what's new.
> The training gaps below are proven techniques used in the original CorrDiff paper
> (Mardani et al., 2023) and virtually all SOTA diffusion implementations, but are
> absent from our current training script.

---

## ⚠️ CRITICAL: Training Practice Gaps (Must-Fix)

These are not "improvements" — they are **missing standard practices** that the current
`train.py` lacks. They should be the first items discussed with NVIDIA.

---

### GAP 1: Exponential Moving Average (EMA) — NOT IMPLEMENTED 🔴

| Attribute | Detail |
|-----------|--------|
| **Severity** | Critical — affects all inference quality |
| **Effort** | ~50 lines of code, 1 day |
| **Risk** | Near zero — additive change, no existing code modified |
| **SOTA basis** | Used in EDM (Karras 2022), DDPM (Ho 2020), Stable Diffusion, original CorrDiff paper |

**Current state:** The training script (`train.py`) maintains only ONE copy of the model.
Gradients update this model directly. The same noisy, gradient-updated weights are used
for both training and any evaluation/sampling.

**What's missing:** All production diffusion models keep TWO copies:
- **Training network:** Updated by optimizer (noisy, fluctuating weights)
- **EMA network:** Exponential moving average of training weights (smooth, stable)

The EMA network is used for all inference/sampling. This is critical because diffusion
model weights fluctuate significantly during training, and the EMA provides a much more
stable set of weights.

**Evidence it's missing:** Searched the entire `examples/weather/corrdiff/` directory
for "EMA", "ema", "exponential moving average" — no results. The training loop in
`train.py` has no EMA update step after `optimizer.step()`.

**Implementation:**
```python
# After optimizer.step() in training_iteration_block():
ema_decay = 0.9999
for p_ema, p_train in zip(ema_model.parameters(), model.parameters()):
    p_ema.data.mul_(ema_decay).add_(p_train.data, alpha=1 - ema_decay)
```

**Target file:** `examples/weather/corrdiff/train.py` — add after `optimizer.step()`

---

### GAP 2: Learning Rate Scheduling — BASIC ONLY 🟡

| Attribute | Detail |
|-----------|--------|
| **Severity** | Moderate — affects convergence speed and final quality |
| **Effort** | ~10-20 lines of code, < 1 day |
| **Risk** | Very low — drop-in replacement for existing LR logic |
| **SOTA basis** | CosineAnnealing used in EDM, DDPM++, Stable Diffusion training |

**Current state:** The LR schedule in `train.py` → `update_learning_rates()` is a simple
linear warmup followed by step decay:
```python
g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)         # warmup
g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // lr_decay_rate)  # step decay
```

**What's missing:** No cosine annealing, no warm restarts (SGDR). Step decay creates
abrupt LR drops that can destabilize diffusion training. Cosine annealing provides
smoother transitions and is standard in modern diffusion model training.

**Evidence it's missing:** Searched entire `examples/weather/corrdiff/` for
"CosineAnnealing" — no results. No PyTorch scheduler objects are used anywhere.

**Recommended replacement:**
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=2)
```

**Target file:** `examples/weather/corrdiff/train.py` — replace `update_learning_rates()`

---

### GAP 3: Sigma Clamping for Numerical Safety 🟡

| Attribute | Detail |
|-----------|--------|
| **Severity** | Low-moderate — rare but catastrophic when it occurs (NaN) |
| **Effort** | ~5 lines of code, < 1 hour |
| **Risk** | Zero — purely defensive |
| **SOTA basis** | Standard practice in all robust diffusion implementations |

**Current state:** In `ResidualLoss.__call__()` (`physicsnemo/metrics/diffusion/loss.py`):
```python
rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
sigma = (rnd_normal * self.P_std + self.P_mean).exp()
```
No clamping. Since `rnd_normal` is Gaussian, extreme values (e.g., ±5σ) produce
`sigma` values of ~1500 or ~2e-4. The weighting formula
`weight = (sigma² + sigma_data²) / (sigma * sigma_data)²` amplifies these extremes.

**What's missing:** A simple clamp to prevent numerical instability:
```python
sigma = (rnd_normal * self.P_std + self.P_mean).exp()
sigma = sigma.clamp(min=1e-7, max=1e3)  # prevent overflow/underflow
```

**Note on CorrDiff vs DDPM:** The EDM formulation (continuous sigma) doesn't suffer from
the `sqrt(1 - alpha_bar)` NaN failure mode of DDPM. However, extreme sigma values can
still produce very large loss weights, causing gradient spikes.

**Target file:** `physicsnemo/metrics/diffusion/loss.py` — in `ResidualLoss.__call__()`

---

### GAP 4: Batch Composition — ALREADY CORRECT ✅

**Status:** No action needed.

The current implementation in `ResidualLoss.__call__()` samples sigma independently
per sample: `rnd_normal = torch.randn([y.shape[0], 1, 1, 1])`. This is the standard
"mixed batch" approach (random noise levels per sample in each batch), which is the
recommended practice used in EDM and all major implementations.

---

## Architecture Improvements

Improvements below are organized into three tiers with **honest** difficulty assessments
based on actual code analysis of the CorrDiff codebase.

---

## Tier 1: Easy Improvements (3-7 days each)

### 1. Adaptive Loss Weighting

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★☆☆☆ — Truly easy, self-contained change |
| **Effort** | 60-100 lines, 3-5 days |
| **Risk** | Low — change is isolated to loss computation |
| **SOTA basis** | ✅ Strong: P2 weighting (Choi et al., ICLR 2022), Min-SNR (Hang et al., ICCV 2023) |

**Current state:** Fixed EDM weighting in `ResidualLoss.__call__()`:
```python
weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
```

**Problem:** This fixed formula assigns uniform importance across noise levels. In
weather super-resolution, some noise levels (mid-range sigma) contribute more useful
gradient signal than others. Very high sigma (pure noise) and very low sigma (near-clean)
contribute less to learning fine-grained weather features.

**Improvement options (from least to most complex):**
1. **Min-SNR-γ weighting** (Hang et al., 2023): `weight = min(SNR, γ) / SNR` — clips
   the maximum weight, reducing emphasis on low-noise timesteps. Drop-in replacement.
2. **P2 (Perception Prioritized) weighting** (Choi et al., 2022): Down-weights
   high-SNR timesteps that already have low loss. Proven on image diffusion.
3. **Learnable per-sigma weights**: Small MLP that maps sigma → weight. Adds a few
   hundred parameters. Requires careful training to avoid degenerate solutions.

**Recommended approach:** Start with Min-SNR-γ (option 1) — it's a single-line change
with strong empirical support.

**Target file:** `physicsnemo/metrics/diffusion/loss.py` — modify weight computation
in `ResidualLoss.__call__()`

---

### 2. Learnable Noise Conditioning

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★☆☆☆ — Small, isolated change |
| **Effort** | 50-80 lines, 2-3 days |
| **Risk** | Low — only changes how sigma is embedded before entering UNet |
| **SOTA basis** | ✅ Moderate: Improved DDPM (Nichol & Dhariwal, 2021), Analog Bits (Chen et al., 2023) |

**Current state:** In `EDMPrecondSuperResolution.forward()`:
```python
c_noise = sigma.log() / 4
```
This fixed formula maps sigma to noise labels fed into the UNet's time embedding.

**Improvement:** Replace the fixed `log/4` formula with a small learnable MLP:
```python
self.noise_embed = nn.Sequential(
    nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 1)
)
c_noise = self.noise_embed(sigma.log())
```
This lets the network learn its own optimal noise-level representation.

**Caveat:** The EDM `log/4` formula is well-tested and theoretically motivated. The
improvement may be marginal. Worth trying but don't expect dramatic gains.

**Target file:** `physicsnemo/models/diffusion/preconditioning.py` —
`EDMPrecondSuperResolution`

---

### 3. Weather-Aware Positional Embeddings

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★☆☆☆ — Extends existing infrastructure |
| **Effort** | 70-110 lines, 3-5 days |
| **Risk** | Low — additive to existing positional embedding system |
| **SOTA basis** | ✅ Strong: GraphCast (Lam et al., Science 2023), Pangu-Weather (Bi et al., Nature 2023) |

**Current state:** `SongUNetPosEmbd._get_positional_embedding()` generates sinusoidal
grids (sin/cos of linear coordinates) with `N_grid_channels=4`. These encode spatial
position but carry no geographical or meteorological meaning.

**Improvement options:**
1. **Geographical encoding:** Replace generic sin/cos with actual lat/lon-based
   embeddings (encode latitude, longitude, altitude, land/sea mask). Requires passing
   geographical metadata from the dataset.
2. **Multi-frequency sinusoidal:** Use multiple frequency bands (already partially
   supported via `N_grid_channels > 4` path) to capture different atmospheric scales.
3. **Learnable grid + geographic prior:** Initialize learnable embeddings with
   geographic features rather than random initialization.

**Note:** The `gridtype="learnable"` option already exists in `SongUNetPosEmbd`. The
improvement here is about **what information** to encode, not the mechanism.

**Target file:** `physicsnemo/models/diffusion/song_unet.py` —
`SongUNetPosEmbd._get_positional_embedding()`

---

## Tier 2: Moderate Improvements (1-3 weeks each)

### 4. Multi-Scale Cross-Attention Enhancement

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★☆ — Significant architectural change, NOT easy |
| **Honest effort** | 2-4 weeks including integration and debugging |
| **Risk** | Medium-High — changes UNetBlock signature, cascades everywhere |
| **SOTA basis** | ⚠️ Mixed — see analysis below |

**Current state:** LR→HR fusion is a simple concatenation in `_scaling_fn()`:
```python
return torch.cat([c_in * x, img_lr.to(x.dtype)], dim=1)
```
After this, the LR information flows through the UNet only via convolutions — there is
no explicit attention mechanism between LR and HR features at intermediate layers.

**The existing self-attention** in `UNetBlock` (at `attn_resolutions`) operates only on
the block's own features (Q, K, V all from same input `x`). There is NO cross-attention
between LR and HR features anywhere in the architecture.

**Why this is harder than it looks:**

1. **UNetBlock signature change:** Currently `forward(self, x, emb)`. Adding LR features
   means `forward(self, x, emb, lr_features=None)`. This cascades through:
   - All `SongUNet.forward()` encoder/decoder loops
   - `SongUNetPosEmbd.forward()`
   - Gradient checkpointing calls (`checkpoint(block, x, emb)` → need to pass lr_features)
   - All existing tests
2. **Memory/compute at full resolution:** For 448×448 images, n=200K tokens.
   Full cross-attention is O(n²) — infeasible without FlashAttention or windowed attention.
3. **LR feature routing:** Need to properly downscale/route LR features to match each
   UNet level's spatial resolution.

**Honest SOTA assessment:**
- ❌ SwinIR, HAT, CAT — image restoration transformers, NOT diffusion models.
   Architecture is fundamentally different.
- ❌ Stable Diffusion cross-attention — for text→image (low-dim token sequences),
   not spatial grid→grid conditioning.
- ✅ SR3, CDM, CorrDiff — all use **concatenation**, which is the standard approach
   for diffusion-based super-resolution.
- The truth: **concatenation-based conditioning is the established standard** for
  diffusion SR. Cross-attention alternatives in this specific setup are a research bet.

**Recommended minimal viable version (1 week):**
Instead of modifying all UNetBlocks, add a single cross-attention layer at the
`_scaling_fn` level — replace simple concatenation with attention-weighted fusion:
```python
# Instead of: torch.cat([c_in * x, img_lr], dim=1)
# Do: attention_fuse(c_in * x, img_lr) then concatenate
```
This avoids cascading changes throughout the UNet while testing the core hypothesis.

**Target files:**
- `physicsnemo/models/diffusion/preconditioning.py` — `_scaling_fn()`
- `physicsnemo/models/diffusion/layers.py` — new `CrossAttentionFusion` class

---

### 5. Physics-Informed Noise Scheduling

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★☆☆ |
| **Effort** | 200-300 lines, 1-2 weeks |
| **Risk** | Medium — changes noise distribution, needs careful validation |
| **SOTA basis** | ✅ Moderate: importance sampling for diffusion (Nichol & Dhariwal 2021), noise schedule search (Chen 2023) |

**Current state:** Log-normal noise sampling in `ResidualLoss`:
```python
sigma = (rnd_normal * self.P_std + self.P_mean).exp()
```
With `P_mean=0.0`, `P_std=1.2` — this is the standard EDM schedule, not tuned for
weather data characteristics.

**Improvement idea:** Different weather variables have different frequency spectra and
noise sensitivity. A physics-informed schedule could:
1. **Importance-sample sigma** based on actual data statistics per variable
2. **Variable-dependent P_mean/P_std** — tune noise schedule per output channel
3. **Learned noise schedule** — parameterize P_mean/P_std as learnable

**Target file:** `physicsnemo/metrics/diffusion/loss.py` — new loss class or modified
`ResidualLoss`

---

### 6. Multi-Variable Attention Mechanism

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★☆☆ |
| **Effort** | 250-400 lines, 2-3 weeks |
| **Risk** | Medium — adds complexity to UNet internals |
| **SOTA basis** | ⚠️ Weak for diffusion models — MetNet-3 and FourCastNet use this in non-diffusion contexts |

**Current state:** All meteorological variables (temperature, pressure, wind, humidity)
are treated as generic channels. The UNet's convolutions process them uniformly without
explicit modeling of inter-variable physical relationships.

**Improvement idea:** Add channel-group attention that models known physical relationships
(e.g., temperature-pressure coupling, wind component correlations).

**Honest assessment:** This is more of a research direction than a proven technique. No
existing diffusion model implements variable-aware channel attention. The standard UNet
convolutions already learn cross-channel relationships implicitly. The benefit is uncertain.

**Target files:**
- `physicsnemo/models/diffusion/layers.py` — new attention module
- `physicsnemo/models/diffusion/song_unet.py` — integration

---

### 7. Temporal Consistency Module

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★☆ |
| **Effort** | 300-450 lines, 2-3 weeks |
| **Risk** | Medium-High — changes data pipeline and model architecture |
| **SOTA basis** | ✅ Moderate: Video Diffusion Models (Ho et al., 2022), temporal attention in weather (GenCast 2024) |

**Current state:** Each weather snapshot is processed independently. No temporal context
from adjacent time steps is used during training or inference.

**Improvement idea:** Add temporal attention or 3D convolution layers that process
sequences of weather snapshots, enforcing physical temporal consistency.

**Caveat:** Requires changes to the data pipeline to provide temporal sequences, not
just single snapshots. This is a significant infrastructure change beyond the model itself.

**Target files:**
- New `physicsnemo/models/diffusion/temporal_module.py`
- `physicsnemo/models/diffusion/song_unet.py`
- Dataset classes in `examples/weather/corrdiff/datasets/`

---

### 8. Hierarchical Multiscale Training

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★☆☆ |
| **Effort** | 180-280 lines, 1-2 weeks |
| **Risk** | Medium — changes training curriculum, needs hyperparameter tuning |
| **SOTA basis** | ✅ Strong: Progressive GAN (Karras 2018), curriculum learning in diffusion (Gu et al., 2023) |

**Current state:** Training operates at a single fixed resolution throughout.

**Improvement idea:** Start training at lower resolution, progressively increase. This
speeds up early training and can improve final quality.

**Target file:** `examples/weather/corrdiff/train.py` — training loop modifications

---

## Tier 3: Hard Improvements (1-2 months each)

### 9. Neural ODE-Based Diffusion Process

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★★ |
| **Effort** | 800-1200 lines, 4-6 weeks |
| **Risk** | High — fundamental architecture change |
| **SOTA basis** | ✅ Strong: Score SDE (Song et al., ICLR 2021), Flow Matching (Lipman et al., 2023) |

Replace discrete diffusion with continuous-time ODE/SDE formulation. This is the most
theoretically grounded improvement but requires the largest engineering effort.

**Target:** New module + major modifications to preconditioning and loss.

---

### 10. Wavelet-Based Frequency Decomposition

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★☆ |
| **Effort** | 600-900 lines, 3-5 weeks |
| **Risk** | High — modifies core UNet processing |
| **SOTA basis** | ✅ Moderate: WaveDiff (Phung et al., 2023), frequency-aware diffusion (Yang et al., 2023) |

Integrate wavelet transforms to decompose features into frequency bands, allowing the
model to process different atmospheric scales explicitly.

**Target:** New wavelet layers + UNet modifications.

---

### 11. Graph Neural Network Integration

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★★ |
| **Effort** | 1000-1500 lines, 6-8 weeks |
| **Risk** | Very High — complete architecture redesign |
| **SOTA basis** | ✅ Strong for weather (GraphCast), but untested in diffusion SR |

Hybrid grid-graph architecture for handling irregular geometries. Extremely ambitious.

---

### 12. Uncertainty-Aware Ensemble Diffusion

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★☆ |
| **Effort** | 700-1000 lines, 4-6 weeks |
| **Risk** | High — changes training and inference pipelines |
| **SOTA basis** | ✅ Strong: Deep Ensembles (Lakshminarayanan 2017), but infrastructure-heavy |

Multiple model copies with explicit uncertainty calibration. Note that diffusion models
inherently provide stochastic samples — the question is whether explicit ensemble
approaches add value beyond simply running multiple sampling passes.

---

### 13. Foundation Model Architecture

| Attribute | Detail |
|-----------|--------|
| **Difficulty** | ★★★★★ |
| **Effort** | 1500-2500 lines, 2-3 months |
| **Risk** | Very High — complete rewrite |
| **SOTA basis** | ✅ Active research: Aurora (Microsoft, 2024), ClimaX (Nguyen et al., 2023) |

Multi-task, multi-phenomenon weather foundation model. This is essentially a new project.

---

## Implementation Priority (Revised)

### 🔴 Phase 0: Fix Training Gaps FIRST (1-2 weeks)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| G1 | **EMA implementation** | 1 day | 🔴 Critical |
| G2 | **Cosine LR scheduling** | < 1 day | 🟡 Moderate |
| G3 | **Sigma clamping** | < 1 hour | 🟡 Safety |

> **These should be discussed with NVIDIA as immediate action items.**
> They are standard practices present in the original CorrDiff paper but missing from
> the current PhysicsNeMo training script.

### Phase 1: Low-Hanging Fruit (1-2 months)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 1 | **Adaptive loss weighting (Min-SNR-γ)** | 3-5 days | High |
| 2 | **Learnable noise conditioning** | 2-3 days | Low-Moderate |
| 3 | **Weather-aware positional embeddings** | 3-5 days | Moderate |

### Phase 2: Research Bets (2-4 months)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 4 | **Cross-attention LR-HR fusion (minimal)** | 1-2 weeks | Unknown |
| 5 | **Physics-informed noise scheduling** | 1-2 weeks | Moderate |
| 6 | **Hierarchical multiscale training** | 1-2 weeks | Moderate |

### Phase 3: Major Efforts (6+ months)

| # | Item | Effort | Impact |
|---|------|--------|--------|
| 7 | Temporal consistency | 2-3 weeks | High if data supports it |
| 8 | Multi-variable attention | 2-3 weeks | Unknown |
| 9 | Neural ODE / Flow Matching | 4-6 weeks | Potentially high |
| 10-13 | Wavelet, GNN, Ensemble, Foundation | 1-3 months each | Research-level |

---

## Success Metrics

### Model Performance
- **R² Score:** Target improvement from current baseline (measure before/after EMA first!)
- **RMSE Reduction:** 10-20% improvement in root mean square error
- **Physical Consistency:** Improved conservation law adherence
- **Multi-Variable Correlation:** Better cross-variable relationships

### Training Efficiency
- **Convergence Speed:** 20-40% faster convergence (EMA + cosine LR alone may achieve this)
- **Memory Usage:** Maintained or reduced memory footprint
- **Stability:** Reduced training instability and NaN occurrences (sigma clamping)

### Inference Quality
- **Sampling Stability:** EMA network should produce noticeably smoother/better samples
- **Physical Realism:** Improved adherence to atmospheric physics
- **Multi-Scale Coherence:** Consistent predictions across spatial scales

---

## Risk Assessment (Revised)

### Zero Risk (Training Gaps)
- ✅ EMA: Additive, no existing code modified
- ✅ Sigma clamping: Purely defensive
- ✅ LR scheduling: Drop-in replacement

### Low Risk (Tier 1 Architecture)
- ✅ Adaptive loss weighting: Isolated to loss function
- ✅ Learnable noise conditioning: Small addition to preconditioning
- ✅ Weather positional embeddings: Extends existing system

### Medium Risk (Tier 2 Architecture)
- ⚠️ Cross-attention: Cascading UNet changes if done at block level
- ⚠️ Physics noise scheduling: Changes noise distribution
- ⚠️ Temporal module: Requires data pipeline changes

### High Risk (Tier 3 Architecture)
- ⚠️ Neural ODE: Fundamental reformulation
- ⚠️ GNN integration: Architecture redesign
- ⚠️ Foundation model: New project scope

---

## NVIDIA Meeting Talking Points

1. **EMA is the #1 priority.** Ask whether NVIDIA's internal CorrDiff training uses EMA
   (the paper implies yes). The PhysicsNeMo training script lacks it entirely.

2. **LR scheduling is trivially improvable.** The current step-decay is outdated.
   Cosine annealing is standard and easy to add.

3. **For architecture improvements, start with adaptive loss weighting** (Min-SNR-γ),
   not cross-attention. It's simpler, better grounded in SOTA, and lower risk.

4. **Cross-attention for LR-HR fusion is a research bet**, not a proven improvement for
   diffusion super-resolution. All existing diffusion SR models (SR3, CDM, CorrDiff)
   use concatenation. Propose it as an experiment, not a guaranteed improvement.

5. **Ask NVIDIA about their internal training recipe** — they likely have EMA, better
   LR scheduling, and possibly other training tricks not reflected in the open-source
   code.

---

*This roadmap prioritizes proven, high-impact changes over speculative architecture
modifications. Fix the foundation before building additions.*