# Generative Model

## What is it?

A **generative model** is a model that can **create new data samples** that resemble the training data. It learns the underlying distribution $p_{\text{data}}(\mathbf{x})$ and can then draw new samples from it.

In contrast, a **discriminative model** learns to map inputs to labels (e.g., "is this a cat or a dog?"). A generative model learns to answer: "What does a cat look like?" and then produces a new cat image that never existed before.

## The Goal

Given a dataset of real samples $\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}$ drawn from an unknown distribution $p_{\text{data}}$, the goal of a generative model is to learn a model distribution $p_\theta$ such that:

$$p_\theta(\mathbf{x}) \approx p_{\text{data}}(\mathbf{x})$$

Once learned, we can **sample** $\mathbf{x}_{\text{new}} \sim p_\theta(\mathbf{x})$ to generate new data.

## Major Families of Generative Models

| Family | How it generates | Key idea | Year (landmark) |
|--------|-----------------|----------|-----------------|
| **GANs** (Generative Adversarial Networks) | Generator vs. discriminator game | Two networks compete: one generates, one judges | 2014 |
| **VAEs** (Variational Autoencoders) | Encode to latent space, decode back | Learn a compressed representation; use ELBO | 2013 |
| **Normalizing Flows** | Invertible transformations | Chain of bijective functions with exact likelihood | 2015 |
| **Autoregressive Models** | Generate one element at a time | Each pixel/token conditioned on all previous ones | 2016 |
| **Diffusion Models** | Iterative denoising | Gradually denoise pure noise into data | 2015/2020 |

## Where Diffusion Models Fit

Diffusion models are a type of **likelihood-based generative model** (like VAEs and flows, unlike GANs). They:

1. Define a **forward process** that gradually adds noise to data
2. Learn a **reverse process** that gradually removes noise
3. Generate new data by starting from pure noise and iteratively denoising

### Compared to other families:

| Aspect | GANs | VAEs | Diffusion |
|--------|------|------|-----------|
| **Sample quality** | High (but unstable) | Lower (blurry) | Very high |
| **Training stability** | Notoriously unstable | Stable | Stable |
| **Likelihood** | No access | Lower bound (ELBO) | Lower bound (ELBO) |
| **Diversity** | Mode collapse risk | Good diversity | Excellent diversity |
| **Speed** | Fast (one forward pass) | Fast | Slow (many steps) |
| **Mode coverage** | Often misses modes | Good | Excellent |

> 💡 Diffusion models emerged as the dominant generative model family around 2020-2021, surpassing GANs in image quality while being much more stable to train.

## Unconditional vs. Conditional Generation

### Unconditional Generation
Generate data from $p_\theta(\mathbf{x})$ — no additional input. Example: "Generate a random face."

### Conditional Generation
Generate data from $p_\theta(\mathbf{x} \mid \mathbf{c})$ — given some conditioning information $\mathbf{c}$. Examples:
- **Text-to-image**: $\mathbf{c}$ = text prompt → "a cat sitting on a rainbow"
- **Super-resolution**: $\mathbf{c}$ = low-resolution image → generate high-resolution version
- **Weather downscaling**: $\mathbf{c}$ = coarse weather field → generate fine-grained weather (this is what CorrDiff does!)
- **Inpainting**: $\mathbf{c}$ = image with missing region → fill in the gap

## Evaluation of Generative Models

How do we know if a generative model is good? Common metrics:

| Metric | What it measures | Used for |
|--------|-----------------|----------|
| **FID** (Fréchet Inception Distance) | Distance between real and generated image statistics | Image quality + diversity |
| **IS** (Inception Score) | Quality and diversity of generated images | Image quality |
| **NLL / BPD** | Negative log-likelihood / bits per dimension | How well the model explains data |
| **CRPS, RMSE** | Forecast accuracy metrics | Weather/scientific applications |

## The Two Phases

Every generative model has two phases:

### 1. Training Phase
- Input: a dataset of real samples
- Process: optimize model parameters $\theta$ to match $p_{\text{data}}$
- Output: a trained model $p_\theta$

### 2. Generation Phase (Inference / Sampling)
- Input: random noise (and optionally conditioning $\mathbf{c}$)
- Process: use the trained model to transform noise into data
- Output: new data samples

For diffusion models specifically:
- **Training**: learn to predict noise at various noise levels
- **Generation**: start from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, iteratively denoise through $T$ steps to get $\mathbf{x}_0$

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $p_{\text{data}}(\mathbf{x})$ | True data distribution (unknown) |
| $p_\theta(\mathbf{x})$ | Model's learned distribution |
| $p_\theta(\mathbf{x} \mid \mathbf{c})$ | Conditional model distribution |
| $G_\theta(\mathbf{z})$ | Generator function (GAN notation) |

## Knowledge Check ✅

1. What is the goal of a generative model in one sentence?
2. Name three families of generative models and one key property of each.
3. What advantage do diffusion models have over GANs? What disadvantage?
4. What is the difference between unconditional and conditional generation?
5. What are the two phases of using a generative model?
6. In weather applications like CorrDiff, what is the conditioning input $\mathbf{c}$?

### Answers

1. **Learn the data distribution $p_{\text{data}}(\mathbf{x})$ well enough to generate new, realistic samples that could plausibly have come from the same distribution.** The model approximates $p_{\text{data}}$ with $p_\theta$, and once trained, we can draw $\mathbf{x}_{\text{new}} \sim p_\theta$ to create data that never existed before but shares the same statistical properties as the training data.

2. Three families:
   - **GANs (Generative Adversarial Networks)**: use a minimax game between a generator and discriminator — the generator tries to fool the discriminator, which tries to distinguish real from fake. Key property: **very fast single-pass generation** but notoriously unstable training.
   - **VAEs (Variational Autoencoders)**: encode data into a compressed latent space and decode back, trained using the ELBO. Key property: **principled probabilistic framework** with a tractable lower bound on the likelihood.
   - **Diffusion Models**: iteratively denoise pure noise into data over many steps. Key property: **excellent sample quality and diversity** with stable training, at the cost of slow generation (many sequential steps).
   
   (Other valid answers: Normalizing Flows — exact likelihood via invertible transformations; Autoregressive Models — generate data one element at a time, each conditioned on all previous elements.)

3. **Advantage over GANs**: Diffusion models have **much more stable training** (no adversarial minimax game, just MSE regression), **better mode coverage** (they don't suffer from mode collapse — they capture the full diversity of the data distribution), and currently achieve **higher sample quality** on benchmarks like ImageNet. **Disadvantage**: Diffusion models are **much slower at generation** — they require many sequential denoising steps (typically 20–1000 forward passes through the network), whereas GANs generate a sample in a single forward pass. This makes diffusion models orders of magnitude slower at inference time.

4. **Unconditional generation** samples from $p_\theta(\mathbf{x})$ with no additional input — the model freely generates whatever it wants (e.g., "generate a random face"). **Conditional generation** samples from $p_\theta(\mathbf{x} \mid \mathbf{c})$ where $\mathbf{c}$ is some conditioning information that steers the output — the model generates data consistent with the condition (e.g., "generate a face matching this text description" or "generate a high-resolution weather field given this coarse input"). The conditioning can be text, a low-resolution image, a class label, a physical constraint, or any other side information.

5. The two phases:
   - **Training phase**: The model sees many real data samples and optimizes its parameters $\theta$ to learn $p_\theta \approx p_{\text{data}}$. For diffusion models, this means learning to predict noise at various noise levels — the training loss is typically $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$. This phase is computationally expensive but only done once.
   - **Generation phase (inference/sampling)**: The trained model is used to create new samples. For diffusion models, this means starting from pure noise $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and iteratively denoising through $T$ steps to produce $\mathbf{x}_0$. This phase is done every time you want a new sample.

6. **The conditioning input $\mathbf{c}$ is a coarse-resolution weather field** (e.g., from a global forecast model or reanalysis like ERA5). CorrDiff performs conditional generation: given a low-resolution weather state, it generates a plausible high-resolution (fine-grained) version. This is essentially **stochastic weather downscaling** — the diffusion model learns the conditional distribution $p_\theta(\mathbf{x}_{\text{high-res}} \mid \mathbf{x}_{\text{low-res}})$, producing multiple possible high-resolution realizations that are all physically consistent with the coarse input but differ in their fine-scale details.

---
*Previous: [13-latent-variable.md](13-latent-variable.md) · Next: [15-loss-function.md](15-loss-function.md)*
