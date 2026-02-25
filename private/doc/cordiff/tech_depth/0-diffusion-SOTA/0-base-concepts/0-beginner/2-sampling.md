# Sampling

## What is it?

**Sampling** means drawing a random value from a probability distribution. When we write:

$$x \sim p(x)$$

we mean: "generate a random value $x$ according to the rules of distribution $p$."

Each time you sample, you may get a **different value**. But over many samples, the histogram of values will match the shape of $p(x)$.

## Concrete Examples

### Sampling from a fair coin (discrete)
$$x \sim \text{Bernoulli}(0.5)$$
Each sample gives either 0 or 1, with equal probability.

### Sampling from a Gaussian (continuous)
$$x \sim \mathcal{N}(0, 1)$$
Each sample gives a real number. Values near 0 are most common; values far from 0 are rare. (See [1-gaussian-distribution.md](1-gaussian-distribution.md))

### Sampling in code (PyTorch)
```python
import torch

# Sample 5 values from standard Gaussian
samples = torch.randn(5)
# e.g., tensor([ 0.3421, -1.2038,  0.0512,  1.8834, -0.4217])

# Sample from Gaussian with mean=3, std=2
samples = 3 + 2 * torch.randn(5)
```

## Why Sampling Matters in Diffusion Models

Diffusion models are **generative models** — their entire purpose is to **sample** new data (images, weather fields, etc.) from a learned distribution.

There are two places where sampling appears:

### 1. During Training: Sampling Noise
We sample random noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ and add it to clean data to create noisy training examples.

### 2. During Generation (Inference): Sampling New Data
We start from pure noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$ and iteratively denoise it to produce a new, never-seen-before data sample.

> 💡 The quality of a generative model is judged by the quality of its **samples**: do the generated images/data look realistic?

## Sampling vs. Evaluating a Distribution

Don't confuse these two operations:

| Operation | What it does | Example |
|-----------|-------------|---------|
| **Evaluate** $p(x)$ | Compute the density at a specific point $x$ | "How likely is this particular image?" |
| **Sample** $x \sim p(x)$ | Generate a random value according to $p$ | "Give me a new random image" |

In diffusion models, we mostly care about **sampling** — generating new data. We rarely need to evaluate $p(x)$ directly.

## Important Terminology

| Term | Meaning |
|------|---------|
| **Sample** (noun) | A single value drawn from a distribution |
| **Sample** (verb) | The act of drawing a value |
| **i.i.d. samples** | "Independent and identically distributed" — multiple samples all drawn from the same distribution, independently of each other |
| **Ancestral sampling** | Sampling step-by-step through a chain of conditional distributions (used in DDPM — you'll see this later) |

## Knowledge Check ✅

1. What does $x \sim p(x)$ mean in plain English?
2. If you sample 10,000 values from $\mathcal{N}(0,1)$ and plot a histogram, what shape will it have?
3. Where does sampling appear during diffusion model **training**?
4. Where does sampling appear during diffusion model **generation/inference**?
5. What is the difference between evaluating $p(x)$ and sampling from $p(x)$?

---
*Previous: [1-gaussian-distribution.md](1-gaussian-distribution.md) · Next: [3-mean-variance-std.md](3-mean-variance-std.md)*
