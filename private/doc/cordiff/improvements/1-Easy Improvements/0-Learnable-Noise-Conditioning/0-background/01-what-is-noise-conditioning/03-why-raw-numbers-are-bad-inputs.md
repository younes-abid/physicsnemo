# Part 3: Why Raw Numbers Are Bad Inputs for Neural Networks

## The Problem

We need to tell the neural network what the noise level σ is. The simplest
approach would be to just pass the number directly:

```python
# Naive approach
model(noisy_image, sigma=2.7)
```

This doesn't work well. Here's why.

## Problem 1: Neural Networks Think in Vectors, Not Scalars

A neural network's basic operation is matrix multiplication:

```
output = W × input + b
```

If the input is a single number (σ = 2.7), then:
- `W` is a single weight
- The output is a single number
- There's essentially **one degree of freedom** to represent the noise level

That's not enough. The network needs to represent complex, non-linear
relationships between the noise level and its internal behavior. For example:
- At σ = 0.5, focus on texture restoration
- At σ = 2.0, focus on edge reconstruction
- At σ = 50, focus on global structure

A single scalar can't encode all these different "modes" simultaneously.

**Solution:** Convert the scalar into a **high-dimensional vector** (e.g., 128
dimensions). Each dimension can encode a different aspect of the noise level.
This is what sinusoidal embeddings do (Part 4).

## Problem 2: Non-Linear Relationships

The relationship between σ and the optimal denoising behavior is highly
non-linear. Consider what the network should do:

```
σ = 0.01:  Output ≈ Input (barely change anything)
σ = 0.1:   Smooth out fine grain, keep edges
σ = 1.0:   Reconstruct missing textures
σ = 10:    Rebuild shapes from blobs
σ = 80:    Generate from scratch using conditioning
```

A linear function `f(σ) = aσ + b` can't capture these qualitatively different
regimes. We need a representation that naturally supports non-linear,
multi-scale information.

## Problem 3: Scale Mismatch

Neural networks work best when all inputs have similar magnitudes. Typical good
input ranges are [-1, 1] or [-5, 5].

In CorrDiff, σ ranges from 0.002 to 80:

```
                 Neural network's "comfort zone"
                 ◄──────────────────────►
                 -1                    +1
                  │                     │
σ = 0.002 ●──────┼─────────────────────┼─────────────────────────● σ = 80
          │      │                     │                          │
          └──────┘                     └──────────────────────────┘
          This tiny                    All of this is way
          region has                   outside the network's
          important σ values           comfort zone
```

If we pass raw σ, the network receives inputs that vary by a factor of 40,000.
The weights and biases would need to handle both 0.002 and 80 simultaneously,
leading to:

- **Gradient issues:** Gradients from σ = 80 dominate; σ = 0.002 is invisible
- **Precision loss:** Float32 has limited precision; operations with 80 lose
  information about 0.002
- **Poor learning:** The network can't allocate equal capacity to all noise levels

## Problem 4: Uniform Spacing vs. Perceptual Spacing

With raw σ, the distance between σ = 0.01 and σ = 0.02 is 0.01.
The distance between σ = 40 and σ = 40.01 is also 0.01.

But perceptually, going from σ = 0.01 to 0.02 is a **huge** change (doubling
the noise!), while going from 40 to 40.01 is negligible (0.025% more noise).

```
Linear scale (raw σ):
├─●─●───────────────────────────────────────────────────────●─┤
0.01 0.02                                                    80
 ↑   ↑                                                       ↑
 These are crammed together         This gets tons of space

Log scale (log σ):
├────●────────────●────────────────────────────●──────────●───┤
  log(0.01)    log(0.02)                   log(40)    log(80)
     ↑            ↑                           ↑          ↑
  Equal spacing reflects equal perceptual importance
```

Log-space gives equal "width" to equal **ratios**, which matches how noise
perception actually works.

## The Solution: Embed, Don't Pass Raw

Instead of passing σ directly, the EDM framework:

1. **Compresses** via log: `log(σ)/4` → range [-1.5, 1.1]
2. **Expands** via sinusoidal embedding: scalar → 128-dim vector
3. **Projects** via MLP: 128-dim → 512-dim conditioning vector

```
Raw σ                    log(σ)/4              Sinusoidal           MLP
(bad input)              (good scalar)         (good vector)        (final embedding)

σ = 0.002  ──→  -1.55  ──→  [0.01, -0.99,    ──→  [0.23, -0.87,
                              0.84, ...]             0.11, ...]
                              (128 dims)             (512 dims)

σ = 80     ──→   1.10  ──→  [-0.81, 0.45,    ──→  [-0.44, 0.72,
                              -0.33, ...]            0.55, ...]
                              (128 dims)             (512 dims)
```

Each step fixes one of the problems:
- **log/4** fixes the scale mismatch and spacing (Problems 3 & 4)
- **Sinusoidal** fixes the scalar-vs-vector problem (Problem 1)
- **MLP** learns the optimal non-linear relationship (Problem 2)

## Where Learnable Noise Conditioning Fits In

The current pipeline uses a **fixed** formula for step 1 (`log(σ)/4`). The
learnable noise conditioning proposal replaces this with a small neural network
that can learn the optimal compression from data:

```
Current:    σ → log(σ)/4      → sinusoidal → MLP → UNet
                 ↑ fixed

Proposed:   σ → log(σ) → [MLP] → sinusoidal → MLP → UNet
                          ↑ learned
```

The intuition: maybe `log(σ)/4` isn't the perfect compression for weather data.
A tiny MLP can learn whatever transformation works best.

## Summary

| Problem | What goes wrong | How it's solved |
|---------|----------------|-----------------|
| Scalar input | One number can't encode complex behavior | Sinusoidal embedding → 128 dims |
| Non-linearity | Linear functions can't capture regime changes | MLP learns non-linear mapping |
| Scale mismatch | σ ranges 0.002–80, networks want [-1, 1] | log(σ)/4 compresses to [-1.5, 1.1] |
| Uniform spacing | Equal Δσ ≠ equal perceptual change | Log-space equalizes ratios |

---

**Previous:** [Part 2: σ — The Noise Level](./02-sigma-the-noise-level.md)
**Next:** [Part 4: Sinusoidal Embeddings Explained](./04-sinusoidal-embeddings-explained.md)
