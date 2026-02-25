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

### How is the Mean Vector $\boldsymbol{\mu}$ Calculated?

Suppose you have $N$ data samples, each of dimension $d$ (e.g., for a 64×64 image, $d = 4096$). Write the $n$-th sample as $\mathbf{x}^{(n)} = (x_1^{(n)}, x_2^{(n)}, \dots, x_d^{(n)})$.

The mean vector $\boldsymbol{\mu}$ is simply the **element-wise average** across all $N$ samples:

$$\boldsymbol{\mu} = \frac{1}{N} \sum_{n=1}^{N} \mathbf{x}^{(n)}$$

Written component by component, the $i$-th entry of $\boldsymbol{\mu}$ is:

$$\mu_i = \frac{1}{N} \sum_{n=1}^{N} x_i^{(n)}, \qquad i = 1, \dots, d$$

**Intuition:** $\mu_i$ is just the ordinary average of the $i$-th dimension across all samples. If dimension $i$ represents pixel #42, then $\mu_i$ is the average value of pixel #42 over your entire dataset.

**Example (tiny 2D case):** Given three 2D data points $\mathbf{x}^{(1)}=(1, 4)$, $\mathbf{x}^{(2)}=(3, 2)$, $\mathbf{x}^{(3)}=(2, 6)$:

$$\boldsymbol{\mu} = \frac{1}{3}\begin{pmatrix}1+3+2\\4+2+6\end{pmatrix} = \begin{pmatrix}2\\4\end{pmatrix}$$

### How is the Covariance Matrix $\boldsymbol{\Sigma}$ Calculated?

The covariance matrix is a $d \times d$ matrix where entry $(i, j)$ measures how dimensions $i$ and $j$ **vary together**:

$$\Sigma_{ij} = \frac{1}{N} \sum_{n=1}^{N} (x_i^{(n)} - \mu_i)(x_j^{(n)} - \mu_j)$$

Or equivalently in matrix form:

$$\boldsymbol{\Sigma} = \frac{1}{N} \sum_{n=1}^{N} (\mathbf{x}^{(n)} - \boldsymbol{\mu})(\mathbf{x}^{(n)} - \boldsymbol{\mu})^\top$$

> 📝 **Note:** Some references use $\frac{1}{N-1}$ instead of $\frac{1}{N}$ (Bessel's correction for unbiased estimation from finite samples). Both are valid; the difference vanishes for large $N$.

**What each entry means:**

| Entry | Formula | Meaning |
|-------|---------|---------|
| $\Sigma_{ii}$ (diagonal) | $\frac{1}{N}\sum_n (x_i^{(n)} - \mu_i)^2$ | **Variance** of dimension $i$ — how much dimension $i$ spreads around its mean |
| $\Sigma_{ij}$ (off-diagonal, $i \neq j$) | $\frac{1}{N}\sum_n (x_i^{(n)} - \mu_i)(x_j^{(n)} - \mu_j)$ | **Covariance** between dimensions $i$ and $j$ — do they increase together (positive), move oppositely (negative), or behave independently (≈ 0)? |

**Key properties of $\boldsymbol{\Sigma}$:**
- It is **symmetric**: $\Sigma_{ij} = \Sigma_{ji}$
- It is **positive semi-definite**: variances and combined spreads are never negative
- Its size is $d \times d$, so for a 4096-dimensional image it would be a 4096 × 4096 matrix (≈16 million entries!) — this is one reason the isotropic simplification below is so important

**Example (continuing the 2D case above):** With $\boldsymbol{\mu} = (2, 4)$:

$$\boldsymbol{\Sigma} = \frac{1}{3}\begin{pmatrix}(1{-}2)^2 + (3{-}2)^2 + (2{-}2)^2 & (1{-}2)(4{-}4) + (3{-}2)(2{-}4) + (2{-}2)(6{-}4) \\ (1{-}2)(4{-}4) + (3{-}2)(2{-}4) + (2{-}2)(6{-}4) & (4{-}4)^2 + (2{-}4)^2 + (6{-}4)^2\end{pmatrix}$$

$$= \frac{1}{3}\begin{pmatrix}2 & -2 \\ -2 & 8\end{pmatrix} = \begin{pmatrix}0.67 & -0.67 \\ -0.67 & 2.67\end{pmatrix}$$

Reading this result: dimension 1 has variance 0.67, dimension 2 has variance 2.67 (more spread), and they have a **negative** covariance (−0.67), meaning when one goes up the other tends to go down.

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

### Answers

1. **Mean $\mu$ and variance $\sigma^2$.** The mean determines the center of the bell curve, and the variance determines how wide/spread out it is. That's it — no other parameters are needed.

2. $$x \sim \mathcal{N}(3, 4)$$
   Note: the second parameter is the **variance** (4), not the standard deviation (2). Some programming libraries use $\sigma$ instead of $\sigma^2$, so always check the convention.

3. **$5 + 2\epsilon \sim \mathcal{N}(5, 4)$.** Using Property 3 (linear transform): $x = \mu + \sigma \cdot \epsilon$ with $\mu = 5$ and $\sigma = 2$, so $x \sim \mathcal{N}(5, 2^2) = \mathcal{N}(5, 4)$. Equivalently, by Property 1, $2\epsilon \sim \mathcal{N}(0, 4)$, then shifting by 5 gives mean 5.

4. **"Isotropic Gaussian" means the covariance matrix is $\sigma^2 \mathbf{I}$** — all dimensions have the **same variance** and are **independent** of each other (zero correlation). For images, this means every pixel gets noise drawn independently with the same strength. This is important because: (a) it treats all pixels equally — no spatial bias, (b) it makes the math simple — no cross-terms between dimensions, and (c) it's trivially parallelizable to sample.

5. **Reasons the Gaussian is chosen:**
   - **Mathematical convenience / closure properties**: Gaussians remain Gaussian under addition, scaling, and conditioning. This keeps every step of the forward and reverse process analytically tractable.
   - **Central Limit Theorem**: The sum of many small independent perturbations converges to a Gaussian regardless of the original distribution, so the forward process naturally ends at a Gaussian.
   - **Easy to sample**: Efficient algorithms exist for generating Gaussian random numbers on CPUs and GPUs.
   - **Tractable KL divergence**: The KL divergence between two Gaussians has a closed-form solution, which is needed for the ELBO training objective.

6. **The CLT tells us that after adding many small independent noise perturbations, the result is approximately Gaussian — no matter what the original data distribution looked like.** This is exactly what the forward process does: it applies $T$ small noise steps. By the end ($t = T$), the data distribution has been pushed to something very close to $\mathcal{N}(\mathbf{0}, \mathbf{I})$, regardless of whether the data was images, audio, weather fields, etc. This guarantees a known, simple starting point for the reverse (generative) process.

---
*Previous: [0-probability-distribution.md](0-probability-distribution.md) · Next: [2-sampling.md](2-sampling.md)*
