# Noise (in the Context of Diffusion Models)

## What is it?

**Noise** is random, unstructured information that obscures or corrupts a signal. In everyday life, think of static on a TV screen or hiss on a phone call.

In diffusion models, noise is not a nuisance — it is the **central mechanism**. The entire framework is built around:
1. **Adding noise** to data (forward process)
2. **Learning to remove noise** from data (reverse process)

## Gaussian Noise (The Default)

In diffusion models, noise almost always means **Gaussian noise** (see [1-gaussian-distribution.md](1-gaussian-distribution.md)):

$$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This means:
- Each element of $\boldsymbol{\epsilon}$ is drawn independently from $\mathcal{N}(0, 1)$
- The noise has **zero mean** (no systematic bias in any direction)
- The noise has **unit variance** (standard deviation = 1) in every dimension
- Each dimension is **independent** (noise in one pixel doesn't affect another)

### What does Gaussian noise look like on an image?

If you sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ for a 64×64 image, you get a 64×64 grid of random values centered around 0. It looks like random gray static — no patterns, no structure.

## Adding Noise to Data

The simplest way to corrupt data with noise:

$$\mathbf{x}_{\text{noisy}} = \mathbf{x}_{\text{clean}} + \sigma \cdot \boldsymbol{\epsilon}$$

Where:
- $\mathbf{x}_{\text{clean}}$ is the original data
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is the noise
- $\sigma$ controls the **noise level** (how much noise)

Small $\sigma$ → barely visible corruption, image still recognizable.  
Large $\sigma$ → image completely destroyed, looks like pure static.

## The Noise Continuum

Diffusion models exploit a **continuum of noise levels**:

```
Clean data ←————————————————————————→ Pure noise
σ ≈ 0                                    σ → ∞
(all signal,                         (all noise,
 no noise)                            no signal)
```

- At low noise: the data is mostly intact, fine details are slightly blurred
- At medium noise: large-scale structure visible, details lost
- At high noise: data is unrecognizable, essentially random

## Two Ways to Think About Noise in Diffusion

### 1. The Noise That Was Added ($\boldsymbol{\epsilon}$)
This is the specific random perturbation drawn from $\mathcal{N}(\mathbf{0}, \mathbf{I})$. In DDPM, the neural network is trained to **predict this noise** given the noisy image:

$$\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \approx \boldsymbol{\epsilon}$$

### 2. The Noise Level ($\sigma$ or $t$)
This is a scalar that describes **how much** noise is present. It determines where we are on the clean-to-noisy continuum.

> 💡 Don't confuse these two: $\boldsymbol{\epsilon}$ is the actual noise (a high-dimensional vector, same size as the data), while $\sigma$ or $t$ is a single number describing the noise intensity.

## Why Noise is Central to Diffusion

The key insight of diffusion models:

1. **Destroying data is easy**: adding noise requires no learning — just sample $\boldsymbol{\epsilon}$ and add it
2. **Restoring data is hard**: removing noise requires understanding the structure of the data
3. **We can learn the hard part**: by training a network on many (clean, noisy) pairs at various noise levels

If we can learn to denoise perfectly at every noise level, we can start from pure noise and iteratively denoise step by step to generate brand new data.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{\epsilon}$ | A noise sample, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |
| $\sigma$ or $\sigma_t$ | Noise level (standard deviation) at step $t$ |
| $\sigma^2$ | Noise variance |
| $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | Neural network that predicts the noise |
| $\mathbf{n}$ | Sometimes used for noise instead of $\boldsymbol{\epsilon}$ |

## Knowledge Check ✅

1. What type of noise is used in diffusion models, and what are its properties?
2. What is the difference between $\boldsymbol{\epsilon}$ (the noise) and $\sigma$ (the noise level)?
3. What happens to an image as $\sigma$ increases from 0 to infinity?
4. Why is adding noise easy but removing noise hard?
5. What does the neural network learn to predict in the noise prediction ($\boldsymbol{\epsilon}$-prediction) formulation?

### Answers

1. **Gaussian noise** $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, with three key properties:
   - **Zero mean**: the noise has no systematic bias — it's equally likely to push pixel values up or down.
   - **Unit variance**: each component has standard deviation 1 (before scaling by $\sigma$).
   - **Independence**: noise in each dimension (pixel) is independent of every other — there's no spatial correlation or structure in the noise itself.
   
   Gaussian noise is used because it has beautiful mathematical properties (closed-form KL divergences, easy reparameterization, the central limit theorem ensures sums of many small perturbations converge to Gaussians).

2. **$\boldsymbol{\epsilon}$ is the actual noise vector** — it has the same dimensionality as the data (e.g., for a 64×64 image, $\boldsymbol{\epsilon} \in \mathbb{R}^{4096}$). It's the specific random perturbation drawn for a particular sample. **$\sigma$ is a scalar** that controls the *intensity* (magnitude) of the noise. The noisy data is $\mathbf{x}_{\text{noisy}} = \mathbf{x}_{\text{clean}} + \sigma \cdot \boldsymbol{\epsilon}$, so $\sigma$ scales how much of $\boldsymbol{\epsilon}$ is actually applied. Think of $\boldsymbol{\epsilon}$ as the "shape" of the noise and $\sigma$ as its "volume."

3. **The image transitions from perfectly clean to pure random static:**
   - $\sigma \approx 0$: the image is essentially unchanged — all fine details are intact.
   - Small $\sigma$: slight grain/fuzz appears, but the image is fully recognizable.
   - Medium $\sigma$: fine details (textures, edges) are lost, but large-scale structure (shapes, colors) is still visible.
   - Large $\sigma$: the image is mostly destroyed — only the faintest ghost of structure might remain.
   - $\sigma \to \infty$: the image is completely obliterated — the result is indistinguishable from pure Gaussian noise. No information about the original image survives.

4. **Adding noise is easy because it requires zero knowledge of the data** — just sample random numbers and add them. Any algorithm can do it; no learning needed. **Removing noise is hard because it requires understanding the *structure* of the data.** To denoise, you must know what "real" data looks like — what pixel patterns form valid images, what spatial correlations exist, what's physically plausible. This is implicit knowledge about the data distribution $p_{\text{data}}$, which can only be learned from examples. This asymmetry is exactly what diffusion models exploit.

5. **The neural network predicts the specific noise vector $\boldsymbol{\epsilon}$ that was added to create $\mathbf{x}_t$ from $\mathbf{x}_0$.** Given the noisy input $\mathbf{x}_t$ and the timestep $t$, the network outputs $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$, which should approximate the true $\boldsymbol{\epsilon}$ that was sampled during the forward process. Once you know the noise, you can subtract it (with appropriate scaling) to recover the clean data. The training loss is simply $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$.

---
*Previous: [9-neural-network-as-function-approximator.md](9-neural-network-as-function-approximator.md) · Next: [11-signal-to-noise-ratio.md](11-signal-to-noise-ratio.md)*
