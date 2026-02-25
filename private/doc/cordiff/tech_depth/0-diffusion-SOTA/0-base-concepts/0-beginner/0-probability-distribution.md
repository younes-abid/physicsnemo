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

### Answers

1. **PMF vs PDF.** A **PMF** (Probability Mass Function) applies to **discrete** distributions — it gives the exact probability of each countable outcome, e.g. $P(X = 3) = 1/6$. A **PDF** (Probability Density Function) applies to **continuous** distributions — it gives a *density*, not a probability. You must integrate the PDF over a range to get a probability: $P(a \le X \le b) = \int_a^b p(x)\,dx$.

2. **No, $p(x) = 0.3$ is not a probability.** For a continuous distribution, $p(x)$ is a *density*. The probability of any single exact point is zero ($P(X = x) = 0$). The value 0.3 tells you how "concentrated" the distribution is around that point. In fact, $p(x)$ can even exceed 1 (e.g., a Uniform(0, 0.5) distribution has $p(x) = 2$ for $x \in [0, 0.5]$). Only the integral over a range gives an actual probability.

3. **"$x$ is sampled from the distribution $p$."** It means we draw a random value $x$ according to the rules defined by $p(x)$ — values where $p(x)$ is high are more likely to be drawn, and values where $p(x)$ is low are less likely.

4. **The two main distributions are:**
   - $p_{\text{data}}(\mathbf{x})$ — the **true data distribution** (unknown). This is what real images/data follow.
   - $p_\theta(\mathbf{x})$ — the **model's learned distribution** (parameterized by neural network weights $\theta$). The goal of training is to make $p_\theta$ as close as possible to $p_{\text{data}}$.
   
   (A third important one is the noise distribution $\mathcal{N}(\mathbf{0}, \mathbf{I})$, which is known and fixed.)

5. **Because natural images live in an astronomically high-dimensional space with incredibly complex structure.** A 256×256 RGB image has ~196,000 dimensions. The set of "realistic-looking" images is a tiny, complicated manifold within this space. There is no closed-form mathematical formula that describes which pixel combinations look like real photographs vs. random noise. We only have *samples* from this distribution (our dataset of images), not the formula itself.

---
*Next: [1-gaussian-distribution.md](1-gaussian-distribution.md) — The specific distribution that makes diffusion models work.*
