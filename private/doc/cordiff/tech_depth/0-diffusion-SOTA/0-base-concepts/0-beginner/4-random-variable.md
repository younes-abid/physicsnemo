# Random Variable

## What is it?

A **random variable** is a variable whose value is determined by a random process. It's not a fixed number — it's a placeholder for "whatever value comes out when we run the random experiment."

Formally, a random variable $X$ is a function that maps outcomes of a random experiment to numbers.

## Types

### Discrete Random Variable
Takes on countable values (e.g., 1, 2, 3, ...).
- Example: $X$ = number shown on a die roll. $X \in \{1, 2, 3, 4, 5, 6\}$.

### Continuous Random Variable
Takes on any real value in some range (possibly all of $\mathbb{R}$).
- Example: $X$ = height of a randomly selected person. $X \in (0, \infty)$.
- Example: $X$ = a pixel value in a generated image. $X \in \mathbb{R}$.

## Random Variable vs. Sample

This distinction is subtle but important:

| Concept | What it is | Notation |
|---------|-----------|----------|
| **Random variable** | An abstract object representing "a value that will be randomly drawn" | $X$ (capital letter) |
| **Sample / realization** | A specific concrete value that was actually drawn | $x$ (lowercase letter) |

When we write $X \sim \mathcal{N}(0,1)$, $X$ is the random variable.  
When we draw a specific value, say $x = 1.37$, that's a **realization** (or **sample**) of $X$.

> 💡 In many ML papers, this distinction is relaxed: people often use lowercase $x$ for both the random variable and its realization. Context tells you which is meant.

## Random Vectors

In diffusion models, we work with **high-dimensional data** (images, weather grids). An image is not a single number — it's a vector of thousands of pixel values.

A **random vector** $\mathbf{X} = (X_1, X_2, \ldots, X_d)$ is a collection of random variables. Each component $X_i$ might represent one pixel.

When we write:
$$\mathbf{x} \sim p(\mathbf{x})$$

we mean a random vector $\mathbf{x}$ (an entire image) is sampled from a joint distribution over all pixels simultaneously.

## Why This Matters for Diffusion

In diffusion models:
- $\mathbf{x}_0$ is a random variable representing a **clean data sample** (e.g., an image)
- $\mathbf{x}_t$ is a random variable representing the **noisy version** of that data at timestep $t$
- $\mathbf{x}_T$ is a random variable that is (approximately) **pure noise**
- $\boldsymbol{\epsilon}$ is a random variable representing the **noise** added during the forward process

All of the diffusion math describes relationships between these random variables.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $X$ or $\mathbf{X}$ | Random variable / random vector |
| $x$ or $\mathbf{x}$ | A realization (specific value) |
| $\mathbf{x}_0$ | A clean data sample |
| $\mathbf{x}_t$ | Data at noise level $t$ |
| $\boldsymbol{\epsilon}$ | Noise (random variable sampled from $\mathcal{N}(\mathbf{0}, \mathbf{I})$) |

## Knowledge Check ✅

1. What is the difference between a random variable $X$ and a realization $x$?
2. Why do we use random vectors $\mathbf{x}$ instead of scalar random variables $x$ in diffusion models?
3. In diffusion notation, what does $\mathbf{x}_0$ represent? What about $\mathbf{x}_T$?
4. Is $\boldsymbol{\epsilon}$ in diffusion models a fixed value or a random variable?
5. If $\mathbf{x}$ is a 64×64 grayscale image, how many dimensions does the random vector have?

### Answers

1. **A random variable $X$ is an abstract object** — it represents "a value that *will be* randomly drawn" but hasn't been drawn yet. It carries all the information about what values are possible and how likely each is. **A realization $x$ is a specific concrete number** that was actually obtained after performing the random draw. For example, $X \sim \mathcal{N}(0,1)$ is the random variable; after sampling, you might get $x = 1.37$, which is the realization. Think of $X$ as the *process* and $x$ as the *outcome*.

2. **Because the data we work with (images, weather fields) is high-dimensional.** A single image is not one number — it's a grid of thousands or millions of pixel values. We need a random vector $\mathbf{x} = (x_1, x_2, \ldots, x_d)$ where each component $x_i$ represents one pixel (or one channel of one pixel). The joint distribution $p(\mathbf{x})$ describes the probability over *entire images*, capturing not just individual pixel statistics but also the correlations between pixels that make images look coherent.

3. **$\mathbf{x}_0$ represents a clean data sample** — an original image (or weather field) from the dataset, with no noise added. **$\mathbf{x}_T$ represents the data after the full forward process** — by that point, so much noise has been added that the original signal is completely destroyed, and $\mathbf{x}_T$ is approximately pure Gaussian noise $\mathcal{N}(\mathbf{0}, \mathbf{I})$.

4. **$\boldsymbol{\epsilon}$ is a random variable.** Each time we run the forward process (during training or generation), we draw a *fresh* sample $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. It's different every time. The network's job is to predict *which particular* $\boldsymbol{\epsilon}$ was drawn, given the noisy result $\mathbf{x}_t$.

5. **4,096 dimensions.** A 64×64 grayscale image has $64 \times 64 = 4{,}096$ pixels, so the random vector $\mathbf{x} \in \mathbb{R}^{4096}$. For a color (RGB) image of the same size, it would be $64 \times 64 \times 3 = 12{,}288$ dimensions. This illustrates why the distributions in generative modeling are so complex — we're working in extremely high-dimensional spaces.

---
*Previous: [3-mean-variance-std.md](3-mean-variance-std.md) · Next: [5-conditional-probability.md](5-conditional-probability.md)*
