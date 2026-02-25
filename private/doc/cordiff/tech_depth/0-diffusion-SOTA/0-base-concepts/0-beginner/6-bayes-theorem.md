# Bayes' Theorem

## What is it?

**Bayes' theorem** is a formula that lets you **reverse** a conditional probability. If you know $p(B \mid A)$, Bayes' theorem tells you $p(A \mid B)$.

$$p(A \mid B) = \frac{p(B \mid A) \cdot p(A)}{p(B)}$$

This is one of the most important results in all of probability and statistics.

## Breaking It Down

Each piece has a name:

| Term | Symbol | Name | Meaning |
|------|--------|------|---------|
| What we want | $p(A \mid B)$ | **Posterior** | Probability of $A$ after observing $B$ |
| What we know | $p(B \mid A)$ | **Likelihood** | How likely is $B$ if $A$ is true? |
| Prior belief | $p(A)$ | **Prior** | Probability of $A$ before seeing any evidence |
| Normalizer | $p(B)$ | **Evidence** (or marginal likelihood) | Total probability of $B$ across all possibilities |

> 💡 **Posterior ∝ Likelihood × Prior** — this is the heart of Bayesian reasoning.

## Intuitive Example

**Scenario**: A medical test for a disease.
- 1% of people have the disease: $P(\text{disease}) = 0.01$
- The test is 99% accurate for sick people: $P(\text{positive} \mid \text{disease}) = 0.99$
- The test has a 5% false positive rate: $P(\text{positive} \mid \text{no disease}) = 0.05$

**Question**: You tested positive. What's the probability you actually have the disease?

Using Bayes:
$$P(\text{disease} \mid \text{positive}) = \frac{P(\text{positive} \mid \text{disease}) \cdot P(\text{disease})}{P(\text{positive})}$$

$$= \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = \frac{0.0099}{0.0099 + 0.0495} = \frac{0.0099}{0.0594} \approx 0.167$$

Only about 17%! The prior (disease is rare) strongly influences the result.

## The Continuous Version

For continuous random variables:

$$p(x \mid y) = \frac{p(y \mid x) \cdot p(x)}{p(y)}$$

Where:
$$p(y) = \int p(y \mid x) \cdot p(x) \, dx$$

## Why Bayes' Theorem Matters for Diffusion

### The Core Problem of Diffusion

In diffusion, we know the **forward process** — how to add noise:
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \quad \text{(easy to compute — it's just adding Gaussian noise)}$$

But we want the **reverse process** — how to remove noise:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) \quad \text{(this is what we need to generate data!)}$$

Bayes' theorem tells us these are related:
$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}$$

**The problem**: Computing $q(\mathbf{x}_{t-1})$ and $q(\mathbf{x}_t)$ requires integrating over all possible images — intractable!

**The solution**: We can't compute the exact reverse process, so we **train a neural network** $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to approximate it.

### The Posterior Trick (DDPM)

However, if we also condition on the **original clean image** $\mathbf{x}_0$, the reverse becomes tractable:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)}{q(\mathbf{x}_t \mid \mathbf{x}_0)}$$

All three terms on the right are Gaussian distributions that we know! This gives us a closed-form **posterior** that we use as the training target. (See [25-elbo.md](25-elbo.md) for more details.)

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | True reverse process (intractable) |
| $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ | Tractable posterior (conditioned on clean data) |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | Learned approximation to the reverse process |

## Knowledge Check ✅

1. State Bayes' theorem. What are the names of each term?
2. Why can't we directly compute $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ in diffusion models?
3. What trick makes the reverse posterior tractable? (Hint: what extra variable do we condition on?)
4. In Bayes' theorem, what role does the prior $p(A)$ play? Give the medical test example.
5. How does a diffusion model deal with the intractable reverse process?

---
*Previous: [5-conditional-probability.md](5-conditional-probability.md) · Next: [7-kl-divergence.md](7-kl-divergence.md)*
