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

### Answers

1. $$p(A \mid B) = \frac{p(B \mid A) \cdot p(A)}{p(B)}$$
   - $p(A \mid B)$ — **Posterior**: the probability of $A$ after observing $B$.
   - $p(B \mid A)$ — **Likelihood**: how probable the evidence $B$ is if $A$ were true.
   - $p(A)$ — **Prior**: our belief about $A$ before seeing any evidence.
   - $p(B)$ — **Evidence** (or marginal likelihood): the total probability of observing $B$ across all possibilities. It acts as a normalizing constant so the posterior integrates to 1.

2. **Because it requires knowing the marginals $q(\mathbf{x}_{t-1})$ and $q(\mathbf{x}_t)$, which are intractable.** By Bayes' theorem, $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \frac{q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) \cdot q(\mathbf{x}_{t-1})}{q(\mathbf{x}_t)}. The forward step $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ is a simple Gaussian — that's easy. But $q(\mathbf{x}_{t-1})$ and $q(\mathbf{x}_t)$ are marginal distributions that require integrating over *all possible images* weighted by the data distribution $p_{\text{data}}$. Since we don't know $p_{\text{data}}$ in closed form, these integrals are impossible to compute.

3. **We additionally condition on the clean data $\mathbf{x}_0$.** The posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ is tractable because all three terms in its Bayes' theorem expansion — $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}, \mathbf{x}_0)$, $q(\mathbf{x}_{t-1} \mid \mathbf{x}_0)$, and $q(\mathbf{x}_t \mid \mathbf{x}_0)$ — are known Gaussians (they come from the forward process, which we fully control). The result is a Gaussian with a closed-form mean and variance. During training, we *have* $\mathbf{x}_0$ (it's our training data), so we can use this tractable posterior as the training target. During generation, we don't have $\mathbf{x}_0$, but the trained network effectively estimates it.

4. **The prior $p(A)$ encodes what we believed *before* seeing any evidence.** It has a powerful effect on the posterior. In the medical test example: even though the test is 99% accurate, $P(\text{disease} \mid \text{positive})$ is only ~17% because the prior $P(\text{disease}) = 0.01$ is so low — the disease is very rare. The rare prior overwhelms the strong likelihood. This illustrates that evidence alone isn't enough; you must account for how likely the hypothesis was *a priori*.

5. **It trains a neural network $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to approximate the intractable true reverse $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$.** Specifically, the network is trained to match the *tractable* posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ by minimizing the KL divergence between them at every timestep. This reduces to predicting the noise $\boldsymbol{\epsilon}$ that was added, via a simple MSE loss. At generation time, the trained network replaces the unknown Bayes' theorem computation — it takes in $\mathbf{x}_t$ and outputs a prediction that lets us compute $\mathbf{x}_{t-1}$.

---
*Previous: [5-conditional-probability.md](5-conditional-probability.md) · Next: [7-kl-divergence.md](7-kl-divergence.md)*
