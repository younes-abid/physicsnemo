# Probability Distribution

## What is it?

A **probability distribution** is a mathematical description of how likely different outcomes are. It tells you: "If I pick a value at random, what are the chances it will be *this* value (or in *this* range)?"

Think of it as a **rule** that assigns a probability to every possible outcome.

## Two Flavors

### Discrete Distribution
When outcomes are countable (e.g., rolling a die):

| Outcome | 1 | 2 | 3 | 4 | 5 | 6 |
|---------|---|---|---|---|---|---|
| Probability | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 | 1/6 |

The **probability mass function (PMF)** gives the probability of each exact outcome:
$$P(X = x)$$

### Continuous Distribution
When outcomes can take any real value (e.g., the height of a person, the pixel intensity of an image):

Here we use a **probability density function (PDF)**, often written as $p(x)$.

> ⚠️ For continuous distributions, $p(x)$ is **not** a probability! It's a **density**. The probability of $x$ falling in a range $[a, b]$ is:
> $$P(a \leq X \leq b) = \int_a^b p(x) \, dx$$

The total area under the PDF always equals 1:
$$\int_{-\infty}^{+\infty} p(x) \, dx = 1$$

## Why Does This Matter for Diffusion Models?

In diffusion models, **everything is about distributions**:

- Your **training data** (e.g., images) comes from some unknown distribution $p_{\text{data}}(\mathbf{x})$. We never know its formula — we only have samples (the images themselves).
- **Noise** follows a known distribution (usually Gaussian — see [1-gaussian-distribution.md](1-gaussian-distribution.md)).
- The **goal** of a generative model is to learn a model distribution $p_\theta(\mathbf{x})$ that is as close as possible to $p_{\text{data}}(\mathbf{x})$, so that when we sample from $p_\theta$, we get realistic-looking data.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $p(x)$ | Probability density of $x$ |
| $p_{\text{data}}(x)$ | The true (unknown) data distribution |
| $p_\theta(x)$ | The model's learned distribution (parameterized by $\theta$) |
| $q(x)$ | Often used for a known or approximate distribution |
| $x \sim p(x)$ | "$x$ is sampled from the distribution $p$" |

## Intuitive Analogy

Imagine a jar of colored balls: 50 red, 30 blue, 20 green. The probability distribution is:
- P(red) = 0.5, P(blue) = 0.3, P(green) = 0.2

For images: imagine an impossibly large jar containing every possible image. Natural-looking photos of cats are very common balls; random static noise images are rare. The distribution $p_{\text{data}}$ tells you how common each image is in this jar.

## Knowledge Check ✅

1. What is the difference between a PMF and a PDF?
2. If $p(x)$ is a PDF, is $p(x) = 0.3$ at some point $x$ a probability? Why or why not?
3. What does the notation $x \sim p(x)$ mean?
4. In diffusion models, what are the two main distributions we care about?
5. Why can't we just write down the formula for $p_{\text{data}}(\mathbf{x})$ for natural images?

---
*Next: [1-gaussian-distribution.md](1-gaussian-distribution.md) — The specific distribution that makes diffusion models work.*
