# Mean, Variance, and Standard Deviation

## What Are They?

These three numbers summarize the **shape** of a probability distribution. They answer two questions:
- **Where** is the distribution centered? → **Mean**
- **How spread out** is the distribution? → **Variance** / **Standard Deviation**

## Mean (μ)

The **mean** (or **expected value**) is the "center of mass" of the distribution.

### For discrete distributions:
$$\mu = \mathbb{E}[X] = \sum_i x_i \, P(X = x_i)$$

### For continuous distributions:
$$\mu = \mathbb{E}[X] = \int_{-\infty}^{\infty} x \, p(x) \, dx$$

### Intuition
If you sample from the distribution millions of times and take the average, you get the mean.

### Example
For $X \sim \mathcal{N}(5, 2)$: the mean is $\mu = 5$. Most samples will cluster around 5.

## Variance (σ²)

The **variance** measures how far values typically spread from the mean.

$$\sigma^2 = \text{Var}(X) = \mathbb{E}\left[(X - \mu)^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

- **Small variance** → values tightly clustered around the mean
- **Large variance** → values widely spread out

### Units issue
If $X$ is measured in meters, variance is in meters². That's awkward, which is why we also use...

## Standard Deviation (σ)

The **standard deviation** is simply the square root of the variance:

$$\sigma = \sqrt{\sigma^2}$$

It has the **same units** as the original data, making it more interpretable.

### Example
For $X \sim \mathcal{N}(0, 4)$:
- Variance = $\sigma^2 = 4$
- Standard deviation = $\sigma = 2$
- About 68% of samples will fall within $[-2, 2]$ (one standard deviation from mean)
- About 95% of samples will fall within $[-4, 4]$ (two standard deviations from mean)

## Key Properties (Used in Diffusion Math)

### Variance of a scaled random variable
$$\text{Var}(aX) = a^2 \text{Var}(X)$$

> ⚠️ Note: the variance scales by $a^2$, not $a$! This is why when you see $\sqrt{\alpha_t}$ multiplying data in diffusion papers, the variance gets multiplied by $\alpha_t$.

### Variance of a sum of independent random variables
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) \quad \text{(if } X, Y \text{ independent)}$$

### Expectation is linear
$$\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$$

This holds even if $X$ and $Y$ are not independent.

## Why This Matters for Diffusion Models

In the forward process of diffusion, we carefully control the **mean** and **variance** at each noise step:

$$\mathbf{x}_t = \underbrace{\sqrt{\alpha_t}}_{\text{scales the mean}} \cdot \mathbf{x}_0 + \underbrace{\sqrt{1 - \alpha_t}}_{\text{controls the std of noise}} \cdot \boldsymbol{\epsilon}$$

- As $t$ increases, the mean contribution shrinks (signal fades)
- As $t$ increases, the variance contribution grows (noise dominates)

Understanding how mean and variance behave under scaling and addition is essential for deriving all the diffusion math.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\mu$ or $\mathbb{E}[X]$ | Mean / expected value |
| $\sigma^2$ or $\text{Var}(X)$ | Variance |
| $\sigma$ or $\text{Std}(X)$ | Standard deviation |
| $\mathbb{E}[\cdot]$ | Expectation operator |

## Knowledge Check ✅

1. If $X \sim \mathcal{N}(3, 9)$, what is the mean? The variance? The standard deviation?
2. If $X$ has variance 4, what is the variance of $3X$?
3. If $X$ and $Y$ are independent with variances 2 and 5, what is the variance of $X + Y$?
4. In the diffusion forward process formula $\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_0 + \sqrt{1-\alpha_t}\boldsymbol{\epsilon}$, why does the noise variance contribution equal $1 - \alpha_t$?
5. What is the 68-95-99.7 rule for Gaussians?

---
*Previous: [2-sampling.md](2-sampling.md) · Next: [4-random-variable.md](4-random-variable.md)*
