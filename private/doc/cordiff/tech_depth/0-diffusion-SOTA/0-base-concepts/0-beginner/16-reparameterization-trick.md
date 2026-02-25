# Reparameterization Trick

## What is it?

The **reparameterization trick** is a technique that lets us **backpropagate gradients through a random sampling operation**. It rewrites a sample from a parameterized distribution as a **deterministic function** of the parameters plus some **fixed external noise**.

Without this trick, we cannot train models that involve sampling — and diffusion models involve sampling at every step.

## The Problem

Suppose we want to sample $\mathbf{x}$ from a Gaussian whose mean $\mu$ and variance $\sigma^2$ depend on learnable parameters $\theta$:

$$\mathbf{x} \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2)$$

We want to compute $\frac{\partial L}{\partial \theta}$ (the gradient of some loss $L$ with respect to $\theta$) so we can update $\theta$ via gradient descent.

**The problem**: sampling is a **stochastic, non-differentiable** operation. There is no formula connecting $\theta$ to the specific value $\mathbf{x}$ that was drawn. The "sample" operation is like a black box that breaks the gradient chain:

```
θ → [μ, σ] → ??? SAMPLE ??? → x → Loss
                    ↑
            Can't differentiate through this!
```

## The Solution

Rewrite the sampling as:

$$\mathbf{x} = \mu_\theta + \sigma_\theta \cdot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Now:
- $\boldsymbol{\epsilon}$ is sampled from a **fixed** distribution (no learnable parameters)
- $\mathbf{x}$ is a **deterministic, differentiable function** of $\mu_\theta$ and $\sigma_\theta$
- The randomness is "externalized" into $\boldsymbol{\epsilon}$

```
ε ~ N(0,I) ──────────────────────────┐
                                      ↓
θ → [μ_θ, σ_θ] → x = μ_θ + σ_θ · ε → Loss
       ↑                                  |
       └──────── gradient flows! ←────────┘
```

Gradients now flow cleanly through the deterministic computation:

$$\frac{\partial \mathbf{x}}{\partial \mu_\theta} = 1, \qquad \frac{\partial \mathbf{x}}{\partial \sigma_\theta} = \boldsymbol{\epsilon}$$

## Concrete Example

**Without reparameterization** (can't backprop):
```python
# This is NOT differentiable w.r.t. mu and sigma
x = torch.normal(mu, sigma)  # sampling blocks gradients
```

**With reparameterization** (can backprop):
```python
# This IS differentiable w.r.t. mu and sigma
eps = torch.randn_like(mu)   # external noise, detached from parameters
x = mu + sigma * eps         # deterministic function of parameters
```

## Where It Appears in Diffusion Models

### 1. The Forward Process
The noisy image at timestep $t$ is computed using the reparameterization trick:

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This is exactly the reparameterization of $\mathbf{x}_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$.

Without this trick, we couldn't construct $\mathbf{x}_t$ in a differentiable way during training.

### 2. The Reverse Process (Stochastic Sampling)
During generation with DDPM, each reverse step samples from:

$$\mathbf{x}_{t-1} = \mu_\theta(\mathbf{x}_t, t) + \sigma_t \, \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Again, reparameterization: the predicted mean $\mu_\theta$ plus scaled external noise.

### 3. VAEs (Historical Context)
The reparameterization trick was popularized by the **VAE paper** (Kingma & Welling, 2013). VAEs need to backpropagate through sampling from the latent distribution $q(\mathbf{z} \mid \mathbf{x})$. The same trick is used: $\mathbf{z} = \mu + \sigma \cdot \boldsymbol{\epsilon}$.

## The General Principle

For **any** distribution that can be written as a location-scale transform of a base distribution:

$$x = g(\theta, \epsilon), \quad \epsilon \sim p(\epsilon)$$

where $g$ is differentiable w.r.t. $\theta$, we can backpropagate through the sampling.

This works for:
- Gaussian: $x = \mu + \sigma \epsilon$, $\epsilon \sim \mathcal{N}(0,1)$
- Uniform: $x = a + (b-a)\epsilon$, $\epsilon \sim \text{Uniform}(0,1)$
- Many other distributions

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | External noise (not parameterized) |
| $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$ | Reparameterized forward process |
| $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ | External noise used during reverse sampling |

## Knowledge Check ✅

1. Why can't we directly backpropagate through a sampling operation?
2. How does the reparameterization trick solve this problem?
3. Rewrite $x \sim \mathcal{N}(3, 4)$ using the reparameterization trick.
4. In the diffusion forward process $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, identify the mean, the standard deviation, and the external noise.
5. Which landmark paper popularized the reparameterization trick?

---
*Previous: [15-loss-function.md](15-loss-function.md) · Next: [17-score-function.md](17-score-function.md)*
