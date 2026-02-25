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

### Answers

1. **The score function is the gradient of the log-probability density with respect to the data:**
   $$\mathbf{s}(\mathbf{x}) = \nabla_{\mathbf{x}} \log p(\mathbf{x})$$
   It is a **vector field** — at every point $\mathbf{x}$ in data space, the score assigns a vector with the same dimensionality as $\mathbf{x}$. This vector points in the direction of steepest increase of the log-density and has magnitude proportional to how steeply the log-density is changing. For a 64×64 image (4096 dimensions), the score is a 4096-dimensional vector.

2. **The score vector points toward higher-probability regions — "uphill" in the density landscape.** If you imagine the probability distribution as a terrain where peaks correspond to likely data and valleys to unlikely data, the score at any point is like a compass arrow pointing toward the nearest peak. At a data point far from the mode, the score has large magnitude pointing strongly toward the mode. At the mode itself, the score is zero (you're already at the top). This "pointing toward data" property is exactly what makes it useful for generation: following score vectors from random noise leads you toward realistic data.

3. For $p(x) = \mathcal{N}(\mu, \sigma^2)$:
   $$\log p(x) = -\frac{(x - \mu)^2}{2\sigma^2} + \text{const}$$
   $$\nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}$$
   
   The score points from $x$ toward the mean $\mu$. If $x > \mu$, the score is negative (pointing left toward $\mu$); if $x < \mu$, it's positive (pointing right toward $\mu$). The magnitude is proportional to the distance from the mean and inversely proportional to $\sigma^2$ — the tighter the distribution, the stronger the pull toward the center. At $x = \mu$, the score is zero.

4. **The score is proportional to the negative noise, scaled by the noise standard deviation:**
   $$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}}$$
   
   Or in EDM notation where $\mathbf{x} = \mathbf{x}_0 + \sigma\boldsymbol{\epsilon}$:
   $$\nabla_{\mathbf{x}} \log p(\mathbf{x} \mid \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sigma}$$
   
   The score points in the **opposite direction of the noise** — toward the clean data and away from the corruption. This makes intuitive sense: the noise displaced the data in direction $\boldsymbol{\epsilon}$, and the score says "go back the other way" ($-\boldsymbol{\epsilon}$), scaled by the noise level.

5. **Because the gradient of a constant is zero.** If $p(\mathbf{x}) = p^*(\mathbf{x})/Z$, then:
   $$\nabla_{\mathbf{x}} \log p(\mathbf{x}) = \nabla_{\mathbf{x}} \log p^*(\mathbf{x}) - \underbrace{\nabla_{\mathbf{x}} \log Z}_{= 0}$$
   
   $Z$ is a constant (it doesn't depend on $\mathbf{x}$), so its gradient w.r.t. $\mathbf{x}$ is zero. This is enormously important because computing $Z = \int p^*(\mathbf{x})\,d\mathbf{x}$ is intractable for complex distributions — it requires integrating over all of data space. The score sidesteps this entirely, making it possible to work with unnormalized densities. This is why score-based methods can handle distributions where direct density evaluation is impossible.

6. **Langevin dynamics iteratively updates a sample by taking small steps in the score direction plus random noise:**
   $$\mathbf{x}_{i+1} = \mathbf{x}_i + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_i) + \sqrt{\eta}\,\mathbf{z}_i, \quad \mathbf{z}_i \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   
   Starting from any initialization (e.g., random noise), this process converges to a sample from $p(\mathbf{x})$ as the step size $\eta \to 0$ and the number of steps $\to \infty$. The score term $\frac{\eta}{2}\nabla_{\mathbf{x}} \log p(\mathbf{x}_i)$ pushes the sample toward high-probability regions (like gradient ascent on log-density), while the noise term $\sqrt{\eta}\,\mathbf{z}_i$ ensures proper exploration so we sample the full distribution rather than just collapsing to the mode. This is conceptually what diffusion models do during the reverse/generation process.

7. **Because predicting noise and predicting the score differ only by a known scaling factor.** The relationship $\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\boldsymbol{\epsilon}/\sqrt{1-\bar{\alpha}_t}$ means:
   - A noise-prediction network $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ trained with $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2$
   - A score-prediction network $\mathbf{s}_\theta(\mathbf{x}_t, t)$ trained with $\|\nabla_{\mathbf{x}_t}\log q - \mathbf{s}_\theta\|^2$
   
   are related by $\mathbf{s}_\theta = -\boldsymbol{\epsilon}_\theta / \sqrt{1-\bar{\alpha}_t}$. The training objectives are identical up to a timestep-dependent constant. This is the key unifying insight: Ho et al.'s DDPM (2020) and Song & Ermon's score-based models (2019–2021) are the same framework expressed in different notation. DDPM asks "what noise was added?", score-based models ask "which direction leads to cleaner data?" — same question, different phrasing.

---
*Previous: [16-reparameterization-trick.md](16-reparameterization-trick.md) · Next: [18-stochastic-differential-equation.md](18-stochastic-differential-equation.md)*
