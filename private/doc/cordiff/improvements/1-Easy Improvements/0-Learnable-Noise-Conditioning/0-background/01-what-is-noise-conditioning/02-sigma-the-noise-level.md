# Part 2: σ (Sigma) — The Noise Level

## What Is σ?

σ (sigma) is a single number that tells you **how much noise** was added to an
image. It's the standard deviation of the Gaussian noise:

```
x_noisy = x_clean + σ × ε    where ε ~ N(0, I)
```

- **σ = 0** → no noise at all, image is perfectly clean
- **σ = 0.01** → tiny noise, image looks almost identical to original
- **σ = 1.0** → moderate noise, image is clearly corrupted
- **σ = 80** → massive noise, image is nearly pure random static

## The Range of σ in CorrDiff

In the EDM framework used by CorrDiff, σ spans a huge range during training:

```
σ_min = 0.002                          σ_max = 80
  │                                        │
  ▼                                        ▼
  ├────────┼────────┼────────┼────────┼────┤
  0.002    0.02     0.2      2.0      20   80

  ◄─── clean ───►          ◄─── noisy ────►
  Fine details              Global structure
  preserved                 destroyed
```

This is a **5-order-of-magnitude** range: from 0.002 to 80. The ratio between
the largest and smallest σ is 40,000×.

## Why the Range Matters

During training, σ is sampled randomly for each training example. The model must
learn to handle ALL noise levels:

| σ range | What the model sees | What it must do |
|---------|--------------------|-----------------| 
| 0.002 – 0.01 | Almost clean image with faint grain | Fix subtle pixel-level artifacts |
| 0.01 – 0.1 | Slightly blurry, fine details lost | Restore textures and sharp edges |
| 0.1 – 1.0 | Noticeably corrupted, shapes visible | Reconstruct medium-scale patterns |
| 1.0 – 10 | Heavily corrupted, only blobs visible | Infer large-scale structure |
| 10 – 80 | Nearly pure noise | Hallucinate global composition from LR input |

**The model's behavior must be completely different at each noise level.** At
σ = 0.01, it should barely touch the image. At σ = 80, it must essentially
generate the entire image from scratch.

## σ in the EDM Preconditioning

In `EDMPrecondSuperResolution`, σ is used for **four** different purposes:

```python
# 1. Scale the input (attenuate the noisy image)
c_in = 1 / (sigma_data² + σ²).sqrt()

# 2. Scale the output (how much of the prediction to trust)  
c_out = σ × sigma_data / (σ² + sigma_data²).sqrt()

# 3. Skip connection weight (how much of the input to keep)
c_skip = sigma_data² / (σ² + sigma_data²)

# 4. Noise label for the neural network ← THIS IS WHAT WE'RE CHANGING
c_noise = sigma.log() / 4
```

The first three (`c_in`, `c_out`, `c_skip`) are **mathematically derived** 
scaling factors that normalize the signal. They ensure the network always sees
inputs and produces outputs with similar magnitudes, regardless of σ.

The fourth one (`c_noise`) is the **noise label** — it tells the neural network
what σ is, so the network can adapt its internal behavior. This is noise
conditioning.

## The Problem with Raw σ

Why not just feed the raw σ value directly into the neural network?

### Problem 1: Extreme Range

The network would see inputs ranging from 0.002 to 80. Neural networks work
best when inputs are roughly in the range [-1, 1] or [-5, 5]. An input of 80
vs 0.002 would cause numerical issues.

### Problem 2: Unequal Resolution

If σ is fed directly, the network has to spend equal "capacity" on:
- Distinguishing σ = 0.01 from σ = 0.02 (a 2× difference that matters a LOT)
- Distinguishing σ = 40 from σ = 40.01 (a 0.025% difference that barely matters)

In linear space, 0.01 and 0.02 are very close (distance = 0.01), but 40 and
40.01 are also very close (distance = 0.01). The network would treat these the
same, even though the first difference is crucial and the second is irrelevant.

### Problem 3: Saturation

With σ values up to 80, most of the input range is devoted to high-noise
regimes. The low-noise regime (where fine details matter most) is crammed into
a tiny corner of the input space.

## The Solution: Logarithmic Compression

The EDM framework applies `log(σ) / 4`:

```
log(σ) compresses the range:
  log(0.002) = -6.21       log(80) = 4.38
  
  Total range: ~10.6 (vs 79.998 for raw σ)

log(σ)/4 further compresses:
  -6.21 / 4 = -1.55        4.38 / 4 = 1.10
  
  Total range: ~2.65 (nice for neural networks!)
```

In log space, **multiplicative differences become additive:**
- log(0.01) - log(0.02) = -0.69 (a 2× ratio)
- log(40) - log(80) = -0.69 (also a 2× ratio)

Now equal ratios get equal "space" in the input. This makes physical sense:
doubling the noise level has roughly the same perceptual impact whether you
go from 0.01→0.02 or from 40→80.

| σ | log(σ) | log(σ)/4 | Interpretation |
|---|--------|----------|----------------|
| 0.002 | -6.21 | -1.55 | Almost clean |
| 0.01 | -4.61 | -1.15 | Very low noise |
| 0.1 | -2.30 | -0.58 | Low noise |
| 0.5 | -0.69 | -0.17 | σ = σ_data (balanced) |
| 1.0 | 0.00 | 0.00 | Moderate noise |
| 10 | 2.30 | 0.58 | High noise |
| 80 | 4.38 | 1.10 | Maximum noise |

**Notice:** The values are nicely centered around 0 and span roughly [-1.5, 1.1].
This is an excellent input range for sinusoidal embeddings (discussed in Part 4).

## Why `/4` Specifically?

The divisor 4 is not arbitrary. In the EDM paper, it comes from the relationship
with `sigma_data` (the expected standard deviation of the training data):

```
sigma_data = 0.5 (default in CorrDiff)
log(sigma_data) = -0.69
4 ≈ -4 × log(sigma_data) ≈ 2.77 ... (approximately)
```

The exact derivation involves the signal-to-noise ratio and the desired
operating range of the positional embedding. The key point: `/4` was chosen
to place the noise labels in a range where the downstream sinusoidal embedding
has good frequency coverage.

**This `/4` constant is one of the things that a learnable noise conditioning
could potentially improve** — maybe the optimal constant for weather data is
`/3.7` or `/4.5`. A small MLP can learn this automatically.

## Summary

| Concept | What it means |
|---------|---------------|
| σ (sigma) | Standard deviation of added noise; controls corruption level |
| Range | 0.002 to 80 in CorrDiff (5 orders of magnitude) |
| Raw σ is bad input | Too wide a range, unequal resolution, saturation |
| log(σ) | Compresses range, equalizes multiplicative differences |
| log(σ)/4 | Further compresses to [-1.5, 1.1] — ideal for embeddings |
| The `/4` constant | Derived from sigma_data=0.5; potentially suboptimal for weather |

---

**Previous:** [Part 1: What Is a Diffusion Model?](./01-what-is-a-diffusion-model.md)
**Next:** [Part 3: Why Raw Numbers Are Bad Inputs](./03-why-raw-numbers-are-bad-inputs.md)
