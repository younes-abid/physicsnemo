# Latent Variable

## What is it?

A **latent variable** is a variable that exists in the model but is **never directly observed** in the data. It's hidden — we can't measure it, but it plays a crucial role in explaining the data.

The word "latent" comes from Latin *latere*, meaning "to be hidden."

## Intuitive Analogy

### Example 1: Medical diagnosis
- **Observed**: symptoms (fever, cough, fatigue)
- **Latent**: the actual disease causing the symptoms

You can observe symptoms directly, but the disease itself is a hidden variable that you infer from the observations.

### Example 2: A document's topic
- **Observed**: the words in the document
- **Latent**: the topic (politics, sports, science)

The topic is never explicitly written — it's a hidden variable that explains the pattern of words.

## Latent Variable Models

A **latent variable model** assumes that the observed data $\mathbf{x}$ is generated through some hidden variable $\mathbf{z}$:

$$p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) \, p(\mathbf{z}) \, d\mathbf{z}$$

This reads: "The probability of observing $\mathbf{x}$ is obtained by considering all possible values of the hidden variable $\mathbf{z}$, weighing how likely each $\mathbf{z}$ is ($p(\mathbf{z})$, the **prior**) and how likely $\mathbf{x}$ is given that $\mathbf{z}$ ($p(\mathbf{x} \mid \mathbf{z})$, the **likelihood**)."

The integral "marginalizes out" the latent variable — summing over all its possible values.

## Latent Variables in Diffusion Models

In diffusion models, the latent variables are the **entire chain of noisy intermediate images**:

$$\mathbf{z} = \{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T\}$$

- **Observed**: $\mathbf{x}_0$ (the clean data)
- **Latent**: $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$ (all the intermediate noisy versions)

We never observe these intermediate steps in our training data — they are introduced by the model's design. The marginalization becomes:

$$p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}$$

This integral is **intractable** (too high-dimensional to compute), which is why we need the ELBO (see [25-elbo.md](25-elbo.md)).

## Latent Variables vs. Latent Space

Don't confuse these related but distinct concepts:

| Concept | Meaning | Example |
|---------|---------|---------|
| **Latent variable** | Any hidden variable in a probabilistic model | $\mathbf{x}_1, \ldots, \mathbf{x}_T$ in diffusion |
| **Latent space** | A lower-dimensional space where data is represented | The compressed space in a VAE or Stable Diffusion |

In **Stable Diffusion** (Latent Diffusion Model), the diffusion process happens in a **compressed latent space** rather than in pixel space. The encoder maps images to a small latent representation, diffusion runs there, and a decoder maps back to pixels. Here "latent" is used in both senses.

In standard diffusion (DDPM), the latent variables $\mathbf{x}_1, \ldots, \mathbf{x}_T$ live in the **same space** as the data (full pixel resolution) — they are just noisier versions.

## Why Latent Variables Matter

1. **Expressiveness**: Latent variables allow simple models to describe complex data distributions
2. **The ELBO connection**: Because we can't compute $p_\theta(\mathbf{x}_0)$ due to the intractable integral over latent variables, we derive a lower bound (ELBO) as our training objective
3. **Generation**: During sampling, we **sample** the latent variables (start from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$) and then decode them step by step into data

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\mathbf{z}$ | Generic latent variable |
| $\mathbf{x}_{1:T}$ | The latent variables in diffusion (all intermediate noisy steps) |
| $p(\mathbf{z})$ | Prior over latent variables |
| $p(\mathbf{x} \mid \mathbf{z})$ | Likelihood: how data is generated from latents |
| $q(\mathbf{z} \mid \mathbf{x})$ | Approximate posterior: inferring latents from data |

## Knowledge Check ✅

1. What makes a variable "latent"?
2. In a diffusion model, what are the latent variables?
3. Why is $p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T}$ intractable?
4. What is the difference between latent variables in DDPM vs. Stable Diffusion?
5. How does the existence of latent variables lead to the need for the ELBO?

### Answers

1. **A variable is "latent" if it is never directly observed in the data — it's hidden.** It exists only within the model's internal representation and must be inferred. You don't have ground-truth labels or measurements for it. In diffusion models, the intermediate noisy images $\mathbf{x}_1, \ldots, \mathbf{x}_T$ are latent because our training data only contains clean images $\mathbf{x}_0$ — the noisy intermediates are introduced by the model's design and never appear in the dataset.

2. **The entire chain of intermediate noisy images** $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$. These are the progressively noisier versions of the clean data $\mathbf{x}_0$. During training, we *construct* them artificially (using the forward process), but they are not part of the observed dataset. The clean data $\mathbf{x}_0$ is the only observed variable; all $T$ intermediate states are latent.

3. **Because the integral is over an astronomically high-dimensional space.** Each $\mathbf{x}_t$ has the same dimensionality as the data (e.g., $d = 64 \times 64 \times 3 = 12{,}288$ for a small image). With $T = 1000$ timesteps, the integral is over $T \times d = 12{,}288{,}000$ dimensions. No numerical integration method can handle this. Even Monte Carlo estimation would require an impossibly large number of samples to get a reasonable estimate in such a high-dimensional space.

4. **In DDPM**, the latent variables $\mathbf{x}_1, \ldots, \mathbf{x}_T$ live in the **same space** as the original data — they are full-resolution images, just progressively noisier. "Latent" here means "hidden/unobserved," not "compressed." **In Stable Diffusion (Latent Diffusion Model)**, the word "latent" is used in a second sense: the diffusion process operates in a **compressed latent space** (e.g., $64 \times 64 \times 4$ instead of $512 \times 512 \times 3$). A pretrained encoder maps images into this lower-dimensional space, diffusion runs there (much cheaper), and a decoder maps back to pixel space. So Stable Diffusion uses "latent" in *both* senses: hidden variables that also live in a compressed representation.

5. **Because the log-likelihood $\log p_\theta(\mathbf{x}_0)$ requires marginalizing over all latent variables, which is intractable.** To evaluate how well the model explains data, we'd need $\log p_\theta(\mathbf{x}_0) = \log \int p_\theta(\mathbf{x}_{0:T})\,d\mathbf{x}_{1:T}$, but this integral is impossible to compute (see answer 3). The ELBO (Evidence Lower Bound) provides a **computable lower bound** on $\log p_\theta(\mathbf{x}_0)$ by introducing an approximate posterior $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ and using Jensen's inequality. Maximizing the ELBO indirectly pushes up the true log-likelihood. In DDPM, the ELBO decomposes beautifully into per-timestep KL divergence terms that reduce to the simple MSE noise-prediction loss.

---
*Previous: [12-markov-chain.md](12-markov-chain.md) · Next: [14-generative-model.md](14-generative-model.md)*
