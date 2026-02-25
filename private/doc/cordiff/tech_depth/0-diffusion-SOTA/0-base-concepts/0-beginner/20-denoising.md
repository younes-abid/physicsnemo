# Denoising

## What is it?

**Denoising** is the task of recovering a clean signal from a corrupted (noisy) version. Given a noisy observation $\mathbf{y}$, the goal is to estimate the original clean data $\mathbf{x}$.

$$\mathbf{y} = \mathbf{x} + \sigma\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

$$\text{Denoiser}: \quad \hat{\mathbf{x}} = D(\mathbf{y}, \sigma)$$

Denoising is one of the oldest problems in signal processing. What makes diffusion models revolutionary is the insight that **a good denoiser is all you need to build a generative model**.

## The Optimal Denoiser

Given a noisy observation $\mathbf{y} = \mathbf{x} + \sigma\boldsymbol{\epsilon}$, what is the best possible estimate of $\mathbf{x}$?

Under MSE loss, the optimal denoiser is the **posterior mean** (also called the **minimum mean squared error (MMSE) estimator**):

$$D^*(\mathbf{y}, \sigma) = \mathbb{E}[\mathbf{x} \mid \mathbf{y}] = \int \mathbf{x} \, p(\mathbf{x} \mid \mathbf{y}) \, d\mathbf{x}$$

This is the expected value of the clean data, given the noisy observation. It requires knowing the data distribution $p(\mathbf{x})$, which is why we need a neural network to approximate it.

## Tweedie's Formula: Connecting Denoising to Scores

A beautiful result called **Tweedie's formula** directly connects the optimal denoiser to the score function (see [17-score-function.md](17-score-function.md)):

$$\mathbb{E}[\mathbf{x} \mid \mathbf{y}] = \mathbf{y} + \sigma^2 \nabla_{\mathbf{y}} \log p(\mathbf{y})$$

In words: "The best denoised estimate equals the noisy input plus a correction term proportional to the score."

This means:
- If you know the **score** $\nabla_\mathbf{y} \log p(\mathbf{y})$, you can denoise optimally
- If you know the **optimal denoiser**, you can recover the score
- **Denoising and score estimation are two sides of the same coin**

> 🔑 This is why training a neural network to denoise (or predict noise) automatically gives you a score estimator, which in turn gives you a generative model.

## Denoising at Different Noise Levels

The key to diffusion is that we denoise at **many different noise levels**:

| Noise level $\sigma$ | What denoising looks like |
|----------------------|--------------------------|
| Very small $\sigma$ | Minor cleanup: remove faint grain, sharpen tiny details |
| Medium $\sigma$ | Reconstruct textures, edges, and mid-level features |
| Large $\sigma$ | Recover global structure (overall layout, shapes) from near-random noise |
| Very large $\sigma$ | Almost hallucinate — invent plausible content from nearly pure noise |

Each noise level requires different "skills":
- **High noise**: the network must understand **global statistics** (what does a face/weather field generally look like?)
- **Low noise**: the network must understand **fine-grained details** (exact textures, sharp edges)

## Denoising as the Core of Diffusion Models

### During Training
The neural network is trained as a denoiser:

1. Take clean data $\mathbf{x}_0$
2. Add noise at a random level: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$
3. Train the network to recover either:
   - The noise $\boldsymbol{\epsilon}$ (noise prediction — DDPM)
   - The clean data $\mathbf{x}_0$ (data prediction)
   - The score $\nabla \log p(\mathbf{x}_t)$ (score prediction)

All three are equivalent formulations of denoising.

### During Generation
The trained denoiser is applied **iteratively**:

```
Pure noise x_T
    ↓ denoise a little
x_{T-1} (slightly less noisy)
    ↓ denoise a little
x_{T-2}
    ↓ ...
    ↓ denoise a little
x_0 (clean generated data!)
```

Each step applies the denoiser at the current noise level to remove a small amount of noise. After many steps, we arrive at clean data.

> 💡 A single denoising step from pure noise would fail — the network can't guess the entire image at once. But **many small denoising steps** work beautifully because each step only needs to make a small correction.

## The EDM Perspective: Denoiser as the Central Object

The EDM paper (Karras et al., 2022) makes the denoiser the **primary object** of the framework. They define:

$$D_\theta(\mathbf{x}; \sigma) = c_{\text{skip}}(\sigma)\,\mathbf{x} + c_{\text{out}}(\sigma)\,F_\theta(c_{\text{in}}(\sigma)\,\mathbf{x}; c_{\text{noise}}(\sigma))$$

Where:
- $F_\theta$ is the raw neural network
- $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}$ are carefully designed preconditioning functions
- The skip connection $c_{\text{skip}}(\sigma)\,\mathbf{x}$ ensures the denoiser passes through the input at low noise levels

This formulation cleanly separates the network architecture from the noise level handling.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $D_\theta(\mathbf{x}, \sigma)$ | Learned denoiser (EDM notation) |
| $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ | Noise predictor (DDPM notation) — equivalent to denoising |
| $\hat{\mathbf{x}}_0$ | Denoised estimate of clean data |
| MMSE estimator | The optimal denoiser (posterior mean) |
| Tweedie's formula | Links optimal denoiser to the score |

## Knowledge Check ✅

1. What is the goal of denoising?
2. What is the optimal denoiser under MSE loss? (What quantity does it compute?)
3. State Tweedie's formula. How does it connect denoising to the score function?
4. Why does diffusion use **many small** denoising steps instead of **one big** step?
5. What different "skills" does the denoiser need at high vs. low noise levels?
6. How does the EDM framework treat the denoiser differently from DDPM?

### Answers

1. **Recovering a clean signal from a noisy observation.** Given $\mathbf{y} = \mathbf{x} + \sigma\boldsymbol{\epsilon}$, the goal is to estimate the original clean data $\mathbf{x}$ as accurately as possible.

2. **The posterior mean** (a.k.a. the **minimum mean squared error / MMSE estimator**):
$$D^*(\mathbf{y}, \sigma) = \mathbb{E}[\mathbf{x} \mid \mathbf{y}] = \int \mathbf{x}\, p(\mathbf{x}\mid\mathbf{y})\, d\mathbf{x}$$
It computes the **expected value of the clean data conditioned on the noisy observation**. This requires (implicitly) knowing the data distribution $p(\mathbf{x})$.

3. **Tweedie's formula:**
$$\mathbb{E}[\mathbf{x} \mid \mathbf{y}] = \mathbf{y} + \sigma^2 \nabla_{\mathbf{y}} \log p(\mathbf{y})$$
It says the optimal denoised estimate equals the noisy input plus a correction proportional to the **score** of the noisy distribution. This means **denoising and score estimation are equivalent**: knowing one immediately gives you the other. Training a network to denoise therefore also trains it to estimate the score, which is exactly what is needed to run the reverse diffusion process.

4. **A single big step from pure noise would fail** because the network cannot reconstruct all global structure and fine details at once from nearly random input. Many small steps work because each step only needs to make a **small, manageable correction** at the current noise level. The iterative refinement lets the model first establish coarse structure (at high noise) and progressively sharpen details (at low noise).

5. At **high noise levels**, the denoiser must understand **global statistics** — what the overall layout, shape, and large-scale structure of the data look like (e.g., the general arrangement of a face, or large-scale weather patterns). At **low noise levels**, the denoiser must understand **fine-grained details** — exact textures, sharp edges, and subtle local features.

6. **EDM makes the denoiser $D_\theta$ the primary / central object**, with explicit preconditioning functions ($c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}$) that cleanly separate noise-level handling from the raw network $F_\theta$. DDPM instead frames everything around a **noise predictor** $\boldsymbol{\epsilon}_\theta$. The EDM formulation includes a **skip connection** ($c_{\text{skip}}(\sigma)\,\mathbf{x}$) so that at low noise levels the denoiser approximately passes the input through unchanged, making training more stable. Both are mathematically equivalent but EDM's parameterization is more principled and easier to analyze.

---
*Previous: [19-ordinary-differential-equation.md](19-ordinary-differential-equation.md) · Next: [21-timestep-and-noise-schedule.md](21-timestep-and-noise-schedule.md)*
