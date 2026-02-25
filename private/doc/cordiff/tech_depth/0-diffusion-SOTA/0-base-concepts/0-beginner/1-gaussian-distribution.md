# Gaussian (Normal) Distribution

## What is it?

The **Gaussian distribution** (also called the **Normal distribution**) is the single most important distribution in diffusion models. It's the famous "bell curve."

A Gaussian distribution is **fully described by just two numbers**:
- **Mean** $\mu$ — the center of the bell curve
- **Variance** $\sigma^2$ — how wide/spread out the bell is

We write:
$$x \sim \mathcal{N}(\mu, \sigma^2)$$

This reads: "$x$ is sampled from a Gaussian with mean $\mu$ and variance $\sigma^2$."

## The PDF Formula

The probability density function of a 1D Gaussian is:

$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Let's break it down:
- $\frac{1}{\sqrt{2\pi\sigma^2}}$ — a normalization constant so the total area = 1
- $\exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ — the "bell shape." Values close to $\mu$ get high density; values far from $\mu$ get low density

## The Standard Gaussian

When $\mu = 0$ and $\sigma^2 = 1$, we get the **standard Gaussian**:

$$\mathcal{N}(0, 1)$$

$$p(x) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{x^2}{2}\right)$$

This is the default noise distribution used in almost all diffusion models. When papers say "sample noise," they almost always mean: $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

## Multivariate Gaussian

Images and data are high-dimensional (e.g., a 64×64 image has 4096 pixel values). We need a **multivariate Gaussian** — a Gaussian over vectors.

$$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

Where:
- $\boldsymbol{\mu}$ is a **vector** of means (one per dimension)
- $\boldsymbol{\Sigma}$ is the **covariance matrix** (describes spread and correlations between dimensions)

### The Isotropic Case (Most Common in Diffusion)

When all dimensions are **independent** and have the **same variance** $\sigma^2$:

$$\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$$

where $\mathbf{I}$ is the identity matrix. This is called **isotropic** Gaussian. The notation becomes:

$$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$$

> 💡 In diffusion models, the noise added at each step is almost always **isotropic Gaussian**: each pixel/dimension gets independent noise with the same variance.

## Key Properties (Used Constantly in Diffusion)

### Property 1: Scaling
If $x \sim \mathcal{N}(\mu, \sigma^2)$, then:
$$ax \sim \mathcal{N}(a\mu, a^2\sigma^2)$$

Multiplying a Gaussian random variable by a constant $a$ scales the mean by $a$ and the variance by $a^2$.

### Property 2: Addition
If $x \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $y \sim \mathcal{N}(\mu_2, \sigma_2^2)$ are **independent**, then:
$$x + y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$$

Means add. Variances add.

### Property 3: Linear Transform of Standard Gaussian
Any Gaussian can be constructed from a standard Gaussian:

$$x = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, 1)$$

then $x \sim \mathcal{N}(\mu, \sigma^2)$.

> 🔑 This is the basis of the **reparameterization trick** (see [16-reparameterization-trick.md](16-reparameterization-trick.md)), which is critical for training diffusion models.

## Why Gaussian for Diffusion?

Three reasons:
1. **Mathematical convenience**: Gaussians are closed under addition, scaling, and conditioning. All the math stays tractable.
2. **Central Limit Theorem**: Adding many small independent random perturbations converges to a Gaussian, regardless of the original distribution. This justifies that after enough noise steps, any data distribution becomes Gaussian.
3. **Easy to sample**: Computers can efficiently generate Gaussian random numbers.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian with mean $\mu$, variance $\sigma^2$ |
| $\mathcal{N}(0, 1)$ | Standard Gaussian |
| $\mathcal{N}(\mathbf{0}, \mathbf{I})$ | Standard multivariate Gaussian (zero mean, identity covariance) |
| $\mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ | Isotropic Gaussian with variance $\sigma^2$ in each dimension |
| $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ | A noise sample drawn from the standard Gaussian |

## Knowledge Check ✅

1. What two parameters fully describe a Gaussian distribution?
2. Write the notation for "x is sampled from a Gaussian with mean 3 and variance 4."
3. If $\epsilon \sim \mathcal{N}(0,1)$, what distribution does $5 + 2\epsilon$ follow?
4. What does "isotropic Gaussian" mean? Why is it important for images?
5. Why is the Gaussian distribution chosen for noise in diffusion models (give at least 2 reasons)?
6. What does the Central Limit Theorem tell us in the context of diffusion?

---
*Previous: [0-probability-distribution.md](0-probability-distribution.md) · Next: [2-sampling.md](2-sampling.md)*
