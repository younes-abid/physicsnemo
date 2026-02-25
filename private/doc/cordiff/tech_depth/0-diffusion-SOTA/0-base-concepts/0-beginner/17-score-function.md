# Score Function (∇ log p)

## What is it?

The **score function** of a probability distribution $p(\mathbf{x})$ is the **gradient of the log-density** with respect to the data:

$$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$

It is a **vector field**: at every point $\mathbf{x}$ in data space, the score tells you a direction and magnitude — specifically, **the direction in which the log-probability increases fastest**.

## Breaking Down the Notation

- $p(\mathbf{x})$ — the probability density at point $\mathbf{x}$ (a scalar)
- $\log p(\mathbf{x})$ — the log of the density (still a scalar)
- $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ — the gradient of that scalar w.r.t. $\mathbf{x}$ → a **vector** with the same dimensionality as $\mathbf{x}$

If $\mathbf{x}$ is a 64×64 image (4096 dimensions), the score is also a 4096-dimensional vector.

## Intuitive Meaning

Imagine the distribution $p(\mathbf{x})$ as a landscape of hills and valleys:
- **High-density regions** = hilltops (likely data)
- **Low-density regions** = valleys (unlikely data)

The score function at any point is like a **compass arrow pointing uphill** — toward higher probability. It tells you: "If you want to find more probable data, move in *this* direction."

```
     Low density ──→ ──→ ──→ High density (data)
                  score vectors point toward data
```

## Score of a Gaussian

For a Gaussian $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \sigma^2 \mathbf{I})$:

$$\log p(\mathbf{x}) = -\frac{\|\mathbf{x} - \boldsymbol{\mu}\|^2}{2\sigma^2} + \text{const}$$

$$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = -\frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2}$$

The score **points from $\mathbf{x}$ toward the mean $\boldsymbol{\mu}$**. The further you are from the mean, the stronger the score (larger magnitude).

> 💡 For noisy data $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, the conditional score is:
> $$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}$$
> This directly connects the **score** to the **noise** $\boldsymbol{\epsilon}$ — they are proportional!

## Score vs. Noise: The Key Relationship

This relationship is fundamental:

$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$$

Or in EDM notation where $\mathbf{x} = \mathbf{x}_0 + \sigma \boldsymbol{\epsilon}$:

$$\nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sigma}$$

This means:
- **Predicting noise** ($\boldsymbol{\epsilon}$-prediction in DDPM) and **predicting the score** ($\mathbf{s}$-prediction in score-based models) are **mathematically equivalent** — they differ only by a scaling factor
- The score points in the **opposite direction of the noise** — it points away from noise, toward clean data

> 🔑 This equivalence is one of the most important unifying insights in diffusion: DDPM (noise prediction) and score-based models (score prediction) are the same thing in different notation!

## Why the Score Function is So Important

### 1. You Don't Need to Know $p(\mathbf{x})$ Itself
Computing $p(\mathbf{x})$ requires knowing the normalization constant $Z = \int p^*(\mathbf{x}) d\mathbf{x}$, which is usually intractable. But the score **does not depend on $Z$**:

$$\nabla_{\mathbf{x}} \log \frac{p^*(\mathbf{x})}{Z} = \nabla_{\mathbf{x}} \log p^*(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z}_{= 0} = \nabla_{\mathbf{x}} \log p^*(\mathbf{x})$$

The normalization constant vanishes! This makes score estimation much easier than density estimation.

### 2. Score Enables Sampling via Langevin Dynamics
If we know the score, we can generate samples using **Langevin dynamics**:

$$\mathbf{x}_{i+1} = \mathbf{x}_i + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_i) + \sqrt{\eta}\,\mathbf{z}_i, \quad \mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Starting from random noise and iterating this update, $\mathbf{x}_i$ converges to a sample from $p(\mathbf{x})$. This is conceptually similar to what diffusion models do during generation.

### 3. Score Matching: Learning the Score
We can train a neural network $\mathbf{s}_\theta(\mathbf{x})$ to approximate the true score. The **denoising score matching** objective is:

$$L = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, \sigma}\left[\|\mathbf{s}_\theta(\mathbf{x}_0 + \sigma\boldsymbol{\epsilon}, \sigma) - (-\boldsymbol{\epsilon}/\sigma)\|^2\right]$$

This is equivalent to the DDPM noise prediction loss (up to a constant).

## The Score-Based Perspective on Diffusion

The **score-based generative modeling** framework (Song & Ermon, 2019-2021) views diffusion as:

1. **Forward**: data is gradually corrupted by noise at increasing levels $\sigma_1 < \sigma_2 < \ldots < \sigma_T$
2. **Learn**: train a score network $\mathbf{s}_\theta(\mathbf{x}, \sigma)$ to estimate $\nabla_\mathbf{x} \log p_\sigma(\mathbf{x})$ at each noise level
3. **Generate**: use the learned scores to run Langevin dynamics (or solve a reverse SDE/ODE) from high noise to low noise

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ | Score function of $p$ |
| $\mathbf{s}_\theta(\mathbf{x}, t)$ or $\mathbf{s}_\theta(\mathbf{x}, \sigma)$ | Neural network estimating the score |
| Score matching | Training objective for learning the score |
| Denoising score matching (DSM) | Score matching via denoising — equivalent to DDPM loss |

## Knowledge Check ✅

1. What is the score function? Write its formula.
2. What does the score vector "point toward" intuitively?
3. For a Gaussian $\mathcal{N}(\mu, \sigma^2)$, compute the score at point $x$.
4. What is the mathematical relationship between the score and the noise $\boldsymbol{\epsilon}$?
5. Why does the normalization constant $Z$ not matter for the score?
6. How does Langevin dynamics use the score to generate samples?
7. Why are DDPM (noise prediction) and score-based models (score prediction) equivalent?

---
*Previous: [16-reparameterization-trick.md](16-reparameterization-trick.md) · Next: [18-stochastic-differential-equation.md](18-stochastic-differential-equation.md)*
