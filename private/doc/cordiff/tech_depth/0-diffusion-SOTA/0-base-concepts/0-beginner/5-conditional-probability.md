# Conditional Probability

## What is it?

**Conditional probability** is the probability of an event **given that** another event has already occurred (or is known to be true).

We write:
$$P(A \mid B) \quad \text{or} \quad p(x \mid y)$$

Read as: "the probability of $A$ **given** $B$" or "the density of $x$ **given** $y$."

The vertical bar $\mid$ means "**given that**" or "**conditioned on**."

## The Formula (Discrete Case)

$$P(A \mid B) = \frac{P(A \cap B)}{P(B)}$$

Where:
- $P(A \cap B)$ = probability that **both** A and B happen
- $P(B)$ = probability that B happens (must be > 0)

### Example
A standard deck of 52 cards. What's the probability of drawing a King **given that** the card is a face card?

- $P(\text{King} \cap \text{Face}) = P(\text{King}) = 4/52$ (all Kings are face cards)
- $P(\text{Face}) = 12/52$
- $P(\text{King} \mid \text{Face}) = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}$

## Conditional Density (Continuous Case)

For continuous random variables, we use conditional **density**:

$$p(x \mid y) = \frac{p(x, y)}{p(y)}$$

Where:
- $p(x, y)$ is the **joint density** of $x$ and $y$ together
- $p(y)$ is the **marginal density** of $y$

## Why This is Everywhere in Diffusion Models

Diffusion models are **built on conditional probabilities**. Almost every equation you'll encounter is a conditional distribution.

### In the Forward Process
The noisy image at step $t$ is defined **conditioned on** the previous step:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$
"What is the distribution of the noisy image at step $t$, given that the image at step $t-1$ is known?"

### In the Reverse Process
The denoised image at step $t-1$ is predicted **conditioned on** the current noisy image:
$$p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$
"What is the distribution of the less-noisy image, given the current noisy image?"

### Conditional Generation
In conditional diffusion models (like CorrDiff for weather), we generate data **conditioned on** some input:
$$p_\theta(\mathbf{x} \mid \mathbf{c})$$
"Generate a high-resolution weather field $\mathbf{x}$ **given** a low-resolution input $\mathbf{c}$."

## Joint, Marginal, and Conditional — The Family

These three are related:

$$p(x, y) = p(x \mid y) \cdot p(y) = p(y \mid x) \cdot p(x)$$

| Type | Symbol | Meaning |
|------|--------|---------|
| **Joint** | $p(x, y)$ | Density of $x$ and $y$ together |
| **Marginal** | $p(x) = \int p(x, y) \, dy$ | Density of $x$ alone (integrate out $y$) |
| **Conditional** | $p(x \mid y)$ | Density of $x$ given $y$ is known |

> 💡 **Marginalization** (integrating out a variable) will appear when deriving the ELBO and other diffusion objectives. It's how we go from a joint distribution to a distribution over fewer variables.

## The Chain Rule of Probability

We can decompose any joint distribution into a product of conditionals:

$$p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2)$$

In diffusion, the entire forward process is written as a chain:
$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$

This is the chain rule applied to the sequence of noisy images $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $p(x \mid y)$ | Conditional density of $x$ given $y$ |
| $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ | Forward process: distribution of noisy data at step $t$ given step $t-1$ |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | Reverse process: learned denoising distribution |
| $p(\mathbf{x} \mid \mathbf{c})$ | Conditional generation given conditioning input $\mathbf{c}$ |

## Knowledge Check ✅

1. What does $p(x \mid y)$ mean in plain English?
2. Write the formula relating joint, marginal, and conditional densities.
3. In the diffusion forward process, what is $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ describing?
4. What does the chain rule of probability allow us to do with a joint distribution?
5. In conditional weather generation, what role does $\mathbf{c}$ play in $p_\theta(\mathbf{x} \mid \mathbf{c})$?

### Answers

1. **"The probability density of $x$ given that $y$ is known (or has already been observed)."** It tells you how the distribution of $x$ changes once you have information about $y$. For example, $p(\text{temperature} \mid \text{month=January})$ gives the distribution of temperature when you already know it's January — which is very different from the unconditional distribution $p(\text{temperature})$ over the whole year.

2. $$p(x, y) = p(x \mid y) \cdot p(y) = p(y \mid x) \cdot p(x)$$
   And the marginal is obtained by integrating out the other variable:
   $$p(x) = \int p(x, y)\,dy = \int p(x \mid y)\,p(y)\,dy$$
   These three relationships — joint, marginal, conditional — are the backbone of probabilistic reasoning and appear in nearly every derivation in diffusion model papers.

3. **It describes the distribution of the noisy data at step $t$, given that the data at step $t-1$ is known.** Specifically, it's a Gaussian that says: "take $\mathbf{x}_{t-1}$, slightly shrink it (multiply by $\sqrt{1-\beta_t}$), and add a small amount of Gaussian noise (with variance $\beta_t$)." It defines one step of the noise-adding process.

4. **The chain rule lets us decompose any joint distribution over many variables into a product of conditional distributions.** For example: $p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2)$. In diffusion, this is used to write the entire forward process as $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ — a product of simple one-step transitions. This is what makes the forward process tractable: instead of modeling a massive joint distribution over $T$ noisy images, we decompose it into $T$ simple Gaussian steps.

5. **$\mathbf{c}$ is the conditioning input — the information that guides generation.** In weather downscaling, $\mathbf{c}$ is typically the low-resolution weather field (e.g., from a coarse global model like ERA5). The conditional distribution $p_\theta(\mathbf{x} \mid \mathbf{c})$ says: "generate a plausible high-resolution weather field $\mathbf{x}$ that is consistent with the given low-resolution input $\mathbf{c}$." Without the conditioning, the model would generate random weather; with conditioning, it generates weather that matches the large-scale patterns in $\mathbf{c}$.

---
*Previous: [4-random-variable.md](4-random-variable.md) · Next: [6-bayes-theorem.md](6-bayes-theorem.md)*
