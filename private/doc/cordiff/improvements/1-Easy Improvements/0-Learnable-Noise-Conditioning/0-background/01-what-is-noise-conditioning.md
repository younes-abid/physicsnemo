# Noise Conditioning in Diffusion Models — Background

## 1. The Core Problem: The Model Needs to Know the Noise Level

A diffusion model is trained to **denoise** images. During training, the model
receives an image corrupted with some amount of Gaussian noise and must predict
how to clean it. But there's a critical question:

> **How much noise was added?**

The model sees a noisy image, but it can't tell whether:
- The image is *almost clean* (low noise, σ ≈ 0.01) → needs tiny corrections
- The image is *moderately noisy* (σ ≈ 1.0) → needs significant denoising
- The image is *nearly pure noise* (σ ≈ 100) → needs to hallucinate structure

**The denoising strategy is completely different at each noise level.** Without
knowing σ, the model would have to guess the noise level from the image itself —
an impossible task since noise and signal are mixed together.

```
Low noise (σ = 0.01):      High noise (σ = 80):
┌──────────────────┐       ┌──────────────────┐
│ ▓▓▓▓▓▓▒▒▓▓▓▓▓▓▓ │       │ ░░▒░░▒▒░░▒░░▒░░ │
│ ▓▓▓▓▒▒▒▒▒▓▓▓▓▓▓ │       │ ▒░░░▒░▒░▒░░▒░░▒ │
│ ▓▓▒▒▒▒▒▒▒▒▒▓▓▓▓ │       │ ░▒░▒░░▒░░▒░░░▒░ │
│ ▓▓▒▒▒▒▒▒▒▒▒▒▓▓▓ │       │ ░░▒░▒▒░▒░░▒▒░▒░ │
│ Image clearly    │       │ Almost pure      │
│ visible          │       │ static           │
└──────────────────┘       └──────────────────┘
   Strategy: fine             Strategy: invent
   pixel adjustments          global structure
```

**Solution:** Tell the model what σ is. This is **noise conditioning** — feeding
the noise level σ as an extra input so the model can adapt its behavior.

## 2. How Noise Conditioning Works in Practice

### 2.1 The Signal Flow

In the EDM framework (used by CorrDiff), the noise level σ enters the model
through this pipeline:

```
  σ (scalar per sample in batch)
  │
  ▼
┌─────────────────┐
│ Noise Mapping   │  σ → c_noise (scalar)
│ (fixed formula) │  e.g., c_noise = log(σ) / 4
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Positional or   │  c_noise → embedding vector (128-dim)
│ Fourier Embed   │  Uses sin/cos at multiple frequencies
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MLP Layers      │  embedding → emb (512-dim)
│ (map_layer0/1)  │  Two linear layers with SiLU activation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ UNet Blocks     │  emb modulates every conv block
│ (via FiLM)      │  via adaptive scale/shift (FiLM)
└─────────────────┘
```

Each step in this pipeline transforms σ into a progressively richer
representation that the UNet can use.

### 2.2 Step 1: The Noise Mapping (c_noise)

The first step converts the raw σ value into a **noise label** `c_noise`. In the
EDM framework, this is a single line of code:

```python
c_noise = sigma.log() / 4
```

**Why log?** The sigma values span several orders of magnitude (e.g., 0.002 to
80). The logarithm compresses this range:

| σ (raw)  | log(σ)  | c_noise = log(σ)/4 |
|----------|---------|---------------------|
| 0.002    | -6.21   | -1.55               |
| 0.1      | -2.30   | -0.58               |
| 1.0      |  0.00   |  0.00               |
| 10.0     |  2.30   |  0.58               |
| 80.0     |  4.38   |  1.10               |

The `/4` further compresses the range to roughly [-1.5, 1.1], which is a good
input range for the downstream sinusoidal embedding.

**Why is this specific formula used?** In the EDM paper (Karras et al., 2022),
the authors derive this from the signal-to-noise ratio (SNR) analysis. The
`log/4` scaling places the values in a range where the sinusoidal positional
embedding has good frequency coverage. It is empirically validated and
theoretically motivated.

### 2.3 Step 2: Sinusoidal Positional Embedding

The scalar `c_noise` is then expanded into a high-dimensional vector using
sinusoidal functions — the same trick used for position encoding in Transformers.

This is the `PositionalEmbedding` class in `physicsnemo/models/diffusion/layers.py`:

```python
# PositionalEmbedding.forward(c_noise):
freqs = torch.arange(0, num_channels // 2)  # e.g., [0, 1, 2, ..., 63]
freqs = (1 / 10000) ** (freqs / 63)         # geometric frequency series
embedding = [cos(c_noise * freqs), sin(c_noise * freqs)]  # 128-dim vector
```

**Why sinusoidal?** Each frequency captures a different "scale" of variation in
the noise level:
- Low frequencies → distinguish "very noisy" from "very clean"
- High frequencies → distinguish subtle differences between nearby noise levels

This gives the network a rich, multi-resolution view of the noise level.

### 2.4 Step 3: MLP Projection

The sinusoidal embedding is then processed by two linear layers with SiLU
activation inside SongUNet:

```python
emb = silu(map_layer0(embedding))  # 128 → 512
emb = silu(map_layer1(emb))        # 512 → 512
```

This MLP can learn to emphasize or suppress different frequency components of
the embedding. The output `emb` is the final conditioning vector.

### 2.5 Step 4: FiLM Conditioning in UNet Blocks

The embedding vector `emb` modulates every UNet block via **FiLM** (Feature-wise
Linear Modulation):

```python
# Inside each UNetBlock:
scale, shift = linear(emb).chunk(2)           # emb → (scale, shift)
x = silu(shift + norm(x) * (scale + 1))       # Modulate features
```

This means the noise level affects **every layer** of the UNet — it controls the
gain and bias of all intermediate feature maps.

## 3. The Three Preconditioning Families

Different diffusion formulations use different noise-to-label mappings. CorrDiff
uses EDM, but it's helpful to understand the alternatives:

### 3.1 VP (Variance Preserving) — DDPM++

```python
c_noise = (M - 1) * sigma_inv(sigma)
```
Maps σ to a discrete timestep index [0, M-1]. Used in the original DDPM
framework. The mapping is non-trivial and involves inverting the noise schedule.

### 3.2 VE (Variance Exploding) — NCSN++

```python
c_noise = (0.5 * sigma).log()
```
Also logarithmic, but with a different scaling constant. Used in score-based
models.

### 3.3 EDM — CorrDiff's Choice

```python
c_noise = sigma.log() / 4
```
The "cleanest" formulation. The `/4` constant is derived from the data statistics
(specifically `sigma_data = 0.5`).

### 3.4 iDDPM (Improved DDPM)

```python
c_noise = M - 1 - round_sigma(sigma)  # discrete index
```
Uses a discrete lookup table (`u`) that maps sigma to timestep indices. The
mapping is pre-computed from the alpha-bar schedule.

### Comparison

| Formulation | Mapping | Type | Learned? |
|-------------|---------|------|----------|
| VP (DDPM++) | `(M-1) × σ_inv(σ)` | Continuous → discrete | ❌ Fixed |
| VE (NCSN++) | `log(0.5σ)` | Log-scale | ❌ Fixed |
| **EDM (CorrDiff)** | **`log(σ)/4`** | **Log-scale** | **❌ Fixed** |
| iDDPM | Lookup table | Discrete | ❌ Fixed |

**Key observation:** In all four families, the noise mapping is **fixed** — it's
a hand-designed formula, not a learned function.

## 4. What Could Go Wrong with a Fixed Mapping?

The fixed `log/4` formula makes assumptions:

1. **Log-space is the right space.** This assumes that the model benefits equally
   from distinguishing σ=0.01 vs σ=0.02 and σ=10 vs σ=20. In log space, both
   differences are the same.

2. **The `/4` scaling is optimal.** This constant determines where in the
   sinusoidal embedding's frequency spectrum the noise levels land. If the
   constant is wrong, important noise levels might map to regions where the
   sinusoidal functions have poor discriminability.

3. **One size fits all.** The same mapping is used for all variables (temperature,
   wind, pressure, humidity). But these variables may have different noise
   sensitivities.

### The Case for Weather Data

Weather super-resolution is different from image generation:

- **Multi-variable:** The model predicts 10+ physical variables simultaneously.
  Each variable has its own noise sensitivity profile.
- **Physical constraints:** Temperature varies smoothly; precipitation is sparse
  and spiky. The "important" noise levels differ per variable.
- **Residual prediction:** CorrDiff predicts residuals (HR - regression_mean),
  not raw images. The residual statistics may not match the `sigma_data = 0.5`
  assumption.

These differences suggest that the optimal noise-level representation might differ
from the generic `log/4` formula derived for natural images.

## 5. What Is "Learnable Noise Conditioning"?

The idea is simple: **replace the fixed `log/4` formula with a small neural
network** that can learn the optimal noise-level mapping from data.

```
CURRENT (fixed):           PROPOSED (learnable):

σ ──→ log(σ)/4 ──→ embed   σ ──→ log(σ) ──→ [MLP] ──→ embed
     (1 line)                              (3 layers)
```

The learnable MLP is tiny (e.g., 1 → 64 → 1, ~130 parameters) but it can:

1. **Learn a non-linear mapping** — maybe quadratic or piecewise-linear is better
   than log for weather data
2. **Shift the operating point** — center the values where the sinusoidal
   embedding is most discriminative
3. **Scale adaptively** — expand regions where the model needs fine-grained noise
   discrimination, compress where it doesn't

### What Stays the Same

Everything after the noise mapping is unchanged:
- The sinusoidal/positional embedding still converts the scalar to a vector
- The MLP projection still produces the 512-dim conditioning vector
- The FiLM modulation in UNet blocks is untouched

Only the **first transformation** (σ → scalar) changes.

## 6. What Is an MLP?

An MLP (Multi-Layer Perceptron) is the simplest kind of neural network — just
stacked linear layers with nonlinear activations between them:

```
Input (1 value)
    │
    ▼
┌────────────────┐
│ Linear(1 → 64) │   y = W₁x + b₁  (64 weights + 64 biases = 128 params)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ SiLU activation│   y = x × sigmoid(x)  (no parameters)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│ Linear(64 → 1) │   y = W₂x + b₂  (64 weights + 1 bias = 65 params)
└───────┬────────┘
        │
        ▼
Output (1 value)

Total: 193 parameters (0.0001% of the full model)
```

The SiLU activation (`x × sigmoid(x)`) allows the MLP to learn non-linear
mappings. Without it, stacking linear layers would just be equivalent to a
single linear layer.

## 7. Prior Work

### Improved DDPM (Nichol & Dhariwal, 2021)

The improved DDPM paper explored **learned variance schedules** — instead of
fixing the noise schedule, they let the model learn it. This is conceptually
similar: making part of the noise-handling pipeline learnable. Their key finding:
learning the variance (even with a simple parameterization) improved sample
quality without significant overhead.

### Analog Bits (Chen et al., 2023)

This paper uses **continuous noise-level embeddings** with learned transformations
for discrete data diffusion. They parameterize the noise-to-embedding mapping
with a small network, finding that learned mappings outperform fixed ones when the
data distribution differs from standard assumptions.

### Score SDE (Song et al., 2021)

While Song et al. used fixed noise conditioning, they noted that "the choice of
noise schedule significantly impacts sample quality." This motivates making the
noise representation itself adaptive.

### Flow Matching (Lipman et al., 2023)

Flow matching uses a learned velocity field parameterized by time `t`. The
time-conditioning is analogous to noise conditioning in diffusion models. Some
flow-matching implementations use learned time embeddings.

## 8. Why This Improvement Is "Easy"

| Aspect | Assessment |
|--------|-----------|
| **Lines of code** | ~50-80 (small MLP + wiring) |
| **Files changed** | 1 (`preconditioning.py`) |
| **Architecture impact** | None — UNet, loss, dataset all untouched |
| **Backward compatibility** | Full — old checkpoints work (MLP initialized to approximate `log/4`) |
| **Risk** | Low — worst case, the MLP learns to approximate `log/4` and we're back to baseline |
| **Training overhead** | Negligible — ~193 extra parameters (0.0001% of total model) |

## 9. Why the Improvement May Be Marginal

Honesty is important. The `log/4` formula is **not arbitrary** — it's derived from
a principled analysis of the EDM preconditioning:

1. **Karras et al. tested alternatives.** The EDM paper compared multiple noise
   conditioning schemes and found `log/4` to be near-optimal for natural images.

2. **The downstream MLP already adapts.** Even with a fixed `log/4` mapping, the
   two-layer MLP (`map_layer0/1`) in the UNet can learn arbitrary nonlinear
   transformations of the embedding. Adding a learnable layer *before* the
   sinusoidal embedding gives marginal additional capacity.

3. **The embedding is already high-dimensional.** The sinusoidal embedding expands
   the scalar to 128 dimensions with multiple frequency bands. This is already a
   very rich representation — there's limited room for improvement in the scalar
   pre-processing.

**Bottom line:** This is a low-risk, low-cost experiment. If it helps (even 1-2%
improvement in RMSE), it's worth keeping. If not, it costs almost nothing and
can be reverted.

---

## References

1. Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. "Elucidating the
   Design Space of Diffusion-Based Generative Models." NeurIPS.
   — *Derives the `log(σ)/4` formula used in CorrDiff*

2. Nichol, A.Q. and Dhariwal, P., 2021. "Improved Denoising Diffusion
   Probabilistic Models." ICML.
   — *Learned variance schedules; conceptually related to learnable conditioning*

3. Chen, T., Zhang, R., and Hinton, G., 2023. "Analog Bits: Generating Discrete
   Data using Diffusion Models with Self-Conditioning." ICLR.
   — *Learned noise-level transformations for non-standard data distributions*

4. Song, Y., Sohl-Dickstein, J., Kingma, D.P., et al., 2021. "Score-Based
   Generative Modeling through Stochastic Differential Equations." ICLR.
   — *Score SDE framework; discusses impact of noise schedule on quality*

5. Vaswani, A. et al., 2017. "Attention Is All You Need." NeurIPS.
   — *Original sinusoidal positional embedding used for noise conditioning*

6. Perez, E., Strub, F., de Vries, H., Dumoulin, V., and Courville, A., 2018.
   "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI.
   — *Feature-wise Linear Modulation used in UNet blocks*

7. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., and Le, M., 2023.
   "Flow Matching for Generative Modeling." ICLR.
   — *Learned time conditioning in flow-based models*
