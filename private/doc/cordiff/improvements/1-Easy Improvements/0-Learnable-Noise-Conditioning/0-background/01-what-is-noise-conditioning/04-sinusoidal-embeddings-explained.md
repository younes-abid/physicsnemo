# Part 4: Sinusoidal Embeddings Explained

## The Goal

We have a single number `c_noise = log(σ)/4` (a scalar in the range ~[-1.5, 1.1]).
We need to convert it into a **rich vector** that the neural network can use
effectively. The standard approach: **sinusoidal positional embeddings**.

## The Core Idea

Use sine and cosine waves at **different frequencies** to encode the scalar:

```python
embedding = [
    cos(c_noise × freq_0), sin(c_noise × freq_0),   # Low frequency
    cos(c_noise × freq_1), sin(c_noise × freq_1),   # ...
    cos(c_noise × freq_2), sin(c_noise × freq_2),   # ...
    ...
    cos(c_noise × freq_63), sin(c_noise × freq_63), # High frequency
]
# Result: 128-dimensional vector (64 frequencies × 2 for sin/cos)
```

Each pair of (cos, sin) at a given frequency creates a **unique fingerprint**
for every value of `c_noise`.

## Analogy: A Clock with Many Hands

Imagine a clock with 64 hands, each rotating at a different speed:

```
Frequency 0 (slowest):     Frequency 32 (medium):    Frequency 63 (fastest):

    12                          12                         12
  /    \                      /    \                     /    \
9 ──●── 3                  9 ──●── 3                  9 ──●── 3
  \    /                      \    /                     \    /
    6                           6                          6
    
Hand barely moves           Hand rotates               Hand spins
when c_noise changes        at moderate speed           very fast
by 0.1                      
```

- **Slow hands** (low frequencies): distinguish very different noise levels
  (e.g., σ = 0.01 vs σ = 10). They change smoothly across the full range.

- **Fast hands** (high frequencies): distinguish nearby noise levels
  (e.g., σ = 1.0 vs σ = 1.1). They oscillate rapidly, creating different
  patterns for similar inputs.

Together, the 64 hands create a **unique combination** for every noise level.
It's like a combination lock with 64 dials.

## The Actual Code

From `physicsnemo/models/diffusion/layers.py`, the `PositionalEmbedding` class:

```python
class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels=128, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels      # 128 output dimensions
        self.max_positions = max_positions     # 10000 (controls frequency range)
        self.endpoint = endpoint

    def forward(self, x):
        # x shape: (B,) — one c_noise value per batch element

        # Create 64 frequency indices: [0, 1, 2, ..., 63]
        freqs = torch.arange(0, self.num_channels // 2)

        # Convert to geometric frequency series:
        # freq_0 = 1.0 (slowest)
        # freq_63 = 10000 (fastest)
        freqs = (1 / self.max_positions) ** (freqs / 63)

        # Outer product: each c_noise value × each frequency
        # Shape: (B, 64)
        x = x.ger(freqs)

        # Concatenate cos and sin: (B, 64) → (B, 128)
        x = torch.cat([x.cos(), x.sin()], dim=1)

        return x  # Shape: (B, 128)
```

### The Frequency Series

The frequencies form a **geometric progression** from 1.0 to 1/10000:

```
freq_0  = (1/10000)^(0/63)  = 1.0        (slowest — one full cycle over ΔC = 2π)
freq_1  = (1/10000)^(1/63)  = 0.862
freq_2  = (1/10000)^(2/63)  = 0.743
...
freq_32 = (1/10000)^(32/63) = 0.01        (medium)
...
freq_63 = (1/10000)^(63/63) = 0.0001      (fastest — oscillates rapidly)
```

This geometric spacing ensures good coverage across all scales. Low frequencies
capture coarse differences; high frequencies capture fine differences.

## Why Not Just Use a Learned Embedding?

You might ask: why use fixed sin/cos functions instead of a learned embedding?

### Option A: Lookup Table (like word embeddings in NLP)

```python
# Discretize σ into 1000 bins, learn an embedding for each
embedding = nn.Embedding(1000, 128)
index = discretize(c_noise, bins=1000)
emb = embedding(index)
```

**Problems:**
- Requires discretizing continuous σ → loses precision
- Can't generalize to unseen σ values (between bins)
- 1000 × 128 = 128,000 parameters just for the embedding

### Option B: Sinusoidal (what EDM uses)

```python
emb = PositionalEmbedding()(c_noise)
```

**Advantages:**
- Works with **continuous** inputs — no discretization needed
- **Zero parameters** — the frequencies are fixed, not learned
- **Smooth** — nearby c_noise values get similar embeddings
- **Unique** — different c_noise values get distinct embeddings
- **Proven** — used in Transformers, diffusion models, NeRFs, etc.

### Option C: Fourier (alternative in the codebase)

```python
emb = FourierEmbedding()(c_noise)
```

Uses **random frequencies** instead of the geometric series. The frequencies
are drawn from a normal distribution and frozen. Also zero learned parameters
after initialization, but the frequency distribution is different.

CorrDiff uses **positional** (Option B) by default.

## Visualizing the Embedding

For two different noise levels, the 128-dim embeddings look like:

```
c_noise = -1.0 (low noise, σ ≈ 0.02):
[0.54, 0.84, 0.99, 0.91, ..., -0.23, 0.65, 0.12, -0.88, ...]
 ████  ████  ████  ████        ░░░░   ████  █░░░  ░░░░

c_noise = 0.5 (moderate noise, σ ≈ 7.4):
[-0.42, 0.27, 0.78, -0.15, ..., 0.91, -0.33, 0.87, 0.44, ...]
 ░░░░   ██░░  ████  ░░░░        ████  ░░░░   ████  ████
```

Every c_noise value produces a **unique pattern** across the 128 dimensions.
The network learns to "read" these patterns and adjust its behavior accordingly.

## The Key Property: Smoothness

Nearby noise levels produce **similar** embeddings:

```
c_noise = 0.50: [ 0.878,  0.479, -0.145, ...]
c_noise = 0.51: [ 0.872,  0.490, -0.139, ...]   ← very similar
c_noise = 0.52: [ 0.867,  0.501, -0.133, ...]   ← very similar

c_noise = 1.10: [-0.453,  0.891,  0.660, ...]   ← completely different
```

This smoothness means the network's behavior changes **gradually** as the noise
level changes — there are no sudden jumps in denoising strategy.

## What This Means for Learnable Noise Conditioning

The sinusoidal embedding is a **fixed** function. It maps c_noise values to
128-dim vectors using predetermined frequencies. The frequencies were designed
for a specific input range (roughly [-1, 1] to [0, 10000] depending on the
application).

The `log(σ)/4` formula places c_noise in the range [-1.5, 1.1]. If a different
formula placed values in, say, [-0.5, 0.5], the sinusoidal embedding would use
a different part of its frequency spectrum — potentially a part with better or
worse discriminability.

**This is exactly what learnable noise conditioning can optimize:** find the
input range that makes the sinusoidal embedding most effective for weather data.

## Summary

| Concept | What it means |
|---------|---------------|
| Sinusoidal embedding | Converts scalar → 128-dim vector using sin/cos waves |
| Multiple frequencies | Each frequency captures a different scale of variation |
| Geometric spacing | Frequencies span from coarse (1.0) to fine (0.0001) |
| Zero parameters | The embedding is fixed; no learning needed |
| Smoothness | Nearby inputs → similar embeddings → gradual behavior change |
| Input range matters | Where c_noise values fall in the frequency spectrum affects quality |

---

**Previous:** [Part 3: Why Raw Numbers Are Bad Inputs](./03-why-raw-numbers-are-bad-inputs.md)
**Next:** [Part 5: How the UNet Uses the Embedding](./05-how-the-unet-uses-the-embedding.md)
