# Reverse Process (Denoising Process)

## What is it?

The **reverse process** (also called the **denoising process** or **generative process**) is the procedure that **gradually recovers clean data from noise**. It goes backward through the forward process, starting from pure Gaussian noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and iteratively denoising to produce a clean sample $\mathbf{x}_0$.

```
x_T ──→ x_{T-1} ──→ x_{T-2} ──→ ··· ──→ x₀
(pure noise)                              (clean data!)
```

Unlike the forward process (which is fixed), the reverse process is **learned** — it contains the neural network $\boldsymbol{\epsilon}_\theta$ or $D_\theta$ that we train.

## Why Is the Reverse Process Hard?

The true reverse distribution $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ requires knowing the full data distribution $p_{\text{data}}$:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \, q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}$$

(This is Bayes' theorem — see [6-bayes-theorem.md](6-bayes-theorem.md))

The problem: $q(\mathbf{x}_{t-1})$ and $q(\mathbf{x}_t)$ are **marginal distributions** that require integrating over all possible data — intractable!

**Solution**: We train a neural network to **approximate** the reverse process:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \approx q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

## The DDPM Reverse Process

DDPM defines the learned reverse process as a Gaussian:

$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \sigma_t^2 \mathbf{I}\right)$$

The network predicts the **mean** $\boldsymbol{\mu}_\theta$, while the variance $\sigma_t^2$ is set to a fixed value (either $\beta_t$ or $\tilde{\beta}_t$).

### Computing the Mean via Noise Prediction

Instead of directly predicting $\boldsymbol{\mu}_\theta$, the network predicts the **noise** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$. The mean is then computed as:

$$\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right)$$

### The Sampling Step

Using the reparameterization trick (see [16-reparameterization-trick.md](16-reparameterization-trick.md)):

$$\boxed{\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

For the final step ($t = 1 \to t = 0$), no noise is added ($\mathbf{z} = \mathbf{0}$).

### Intuition Behind This Formula

1. **Start with noisy image** $\mathbf{x}_t$
2. **Predict noise** $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ — what noise is in this image?
3. **Subtract (scaled) predicted noise** — remove the estimated noise
4. **Add a small amount of fresh noise** $\sigma_t \mathbf{z}$ — this maintains stochasticity and diversity

> 💡 Step 4 might seem counterproductive (why add noise while denoising?), but it's essential! The fresh noise provides stochasticity: different runs produce different samples. Without it, generation becomes deterministic (which is what DDIM does — see [19-ordinary-differential-equation.md](19-ordinary-differential-equation.md)).

## The Tractable Posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$

While $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ is intractable, if we **also condition on** the clean data $\mathbf{x}_0$, the posterior becomes tractable:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1};\; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\; \tilde{\beta}_t \mathbf{I})$$

With closed-form expressions:

$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t$$

> 🔑 This tractable posterior is the **training target**. We train $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to match $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ by minimizing the KL divergence between them (see [7-kl-divergence.md](7-kl-divergence.md)).

Of course, during generation we don't have $\mathbf{x}_0$. But the network's noise prediction lets us **estimate** $\mathbf{x}_0$:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}$$

## The Full Generation Algorithm (DDPM)

```
Algorithm: DDPM Sampling
─────────────────────────
1. Sample x_T ~ N(0, I)                    # Start from pure noise
2. For t = T, T-1, ..., 1:
   a. If t > 1: sample z ~ N(0, I)         # Fresh noise
      Else: z = 0                           # No noise at final step
   b. ε̂ = ε_θ(x_t, t)                      # Predict noise
   c. x_{t-1} = (1/√α_t)(x_t - (β_t/√(1-ᾱ_t))·ε̂) + σ_t·z
3. Return x_0                               # Clean generated sample!
```

This requires $T$ sequential network evaluations (typically $T = 1000$), which is why DDPM generation is slow.

## Different Reverse Process Variants

| Method | Stochastic? | Steps needed | Key idea |
|--------|------------|--------------|----------|
| **DDPM** | Yes (adds noise $\mathbf{z}$ each step) | ~1000 | Original formulation |
| **DDIM** | No (deterministic, $\eta = 0$) | ~50-100 | Skip steps, same quality |
| **DDIM** ($\eta > 0$) | Partially | ~50-100 | Interpolate between DDPM and deterministic |
| **EDM** (Heun solver) | No (ODE solver) | ~35-80 | Optimal step schedule + 2nd-order solver |
| **DPM-Solver** | No (ODE solver) | ~10-20 | High-order solver for fast sampling |

## Stochastic vs. Deterministic Reverse

| Stochastic (SDE) | Deterministic (ODE) |
|-------------------|---------------------|
| Adds fresh noise each step | No noise — fully deterministic |
| Different output each run | Same noise seed → same output |
| Better sample diversity | Better for interpolation, editing |
| Needs more steps | Can use fewer steps |
| DDPM sampler | DDIM ($\eta=0$), EDM Heun, DPM-Solver |

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | Learned reverse step |
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ | Tractable posterior (training target) |
| $\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)$ | Predicted mean of the reverse step |
| $\sigma_t^2$ | Variance of the reverse step (usually fixed) |
| $\hat{\mathbf{x}}_0$ | Estimated clean data from noise prediction |

## Knowledge Check ✅

1. What does the reverse process do, and is it learned or fixed?
2. Why is $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ intractable?
3. What makes $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ tractable?
4. In the DDPM sampling formula, identify: (a) the noise removal term, (b) the fresh noise injection.
5. Why does DDPM add fresh noise $\mathbf{z}$ at each step? What happens if you don't?
6. What is the main disadvantage of the DDPM reverse process? How do DDIM and EDM address it?
7. Write out the 3 main steps of the DDPM generation algorithm.
8. How can the network estimate $\hat{\mathbf{x}}_0$ from $\mathbf{x}_t$ and the predicted noise?

---
*Previous: [22-forward-process.md](22-forward-process.md) · Next: [24-unet-architecture.md](24-unet-architecture.md)*
