# Evidence Lower Bound (ELBO)

## What is it?

The **Evidence Lower Bound (ELBO)** is a tractable **lower bound** on the log-likelihood $\log p_\theta(\mathbf{x}_0)$ of the data. Since computing the exact log-likelihood of a diffusion model is intractable (see [8-log-likelihood.md](8-log-likelihood.md)), we instead maximize the ELBO — which indirectly pushes up the true log-likelihood.

$$\log p_\theta(\mathbf{x}_0) \geq \text{ELBO}(\theta)$$

By maximizing the ELBO (or equivalently, minimizing the negative ELBO), we train the model to assign high probability to real data.

## Why Is $\log p_\theta(\mathbf{x}_0)$ Intractable?

The model defines a joint distribution over the clean data and all noisy intermediates:

$$p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}$$

This integral marginalizes out all the latent variables $\mathbf{x}_1, \ldots, \mathbf{x}_T$ (see [13-latent-variable.md](13-latent-variable.md)). For images, each $\mathbf{x}_t$ has thousands of dimensions, and there are $T = 1000$ of them. The integral is over an astronomically high-dimensional space — impossible to compute.

## Deriving the ELBO

The ELBO comes from a simple inequality. Starting with:

$$\log p_\theta(\mathbf{x}_0) = \log \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}$$

We introduce the forward process $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ as a proposal distribution:

$$= \log \int \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)} \, q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \, d\mathbf{x}_{1:T}$$

$$= \log \, \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\left[\frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right]$$

By **Jensen's inequality** ($\log \mathbb{E}[\cdot] \geq \mathbb{E}[\log \cdot]$ since log is concave):

$$\geq \mathbb{E}_{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\left[\log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)}\right] = \text{ELBO}$$

> 💡 The gap between $\log p_\theta(\mathbf{x}_0)$ and the ELBO is exactly $D_{\text{KL}}(q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{1:T} \mid \mathbf{x}_0))$. This KL divergence is always $\geq 0$, confirming the ELBO is indeed a lower bound.

## Decomposing the ELBO into KL Terms

The ELBO can be decomposed into a sum of simpler terms. After expanding and regrouping:

$$\text{ELBO} = \underbrace{-D_{\text{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T))}_{L_T \text{ (prior matching)}} + \sum_{t=2}^{T} \underbrace{-D_{\text{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))}_{L_{t-1} \text{ (denoising matching)}} + \underbrace{\log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)}_{L_0 \text{ (reconstruction)}}$$

Let's understand each term:

### $L_T$: Prior Matching Term
$$L_T = -D_{\text{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, p(\mathbf{x}_T))$$

Measures how close the end of the forward process is to pure Gaussian noise $\mathcal{N}(\mathbf{0}, \mathbf{I})$. If the noise schedule is designed properly, this is nearly zero. **This term has no learnable parameters** — it's determined by the noise schedule.

### $L_{t-1}$ (for $t = 2, \ldots, T$): Denoising Matching Terms
$$L_{t-1} = -D_{\text{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))$$

This is the **core of the training objective**. Each term asks: "How close is the learned reverse step $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ to the true reverse step $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$?"

Both are Gaussians (see [23-reverse-process.md](23-reverse-process.md)), so the KL divergence has a closed-form solution (see [7-kl-divergence.md](7-kl-divergence.md)).

### $L_0$: Reconstruction Term
$$L_0 = \log p_\theta(\mathbf{x}_0 \mid \mathbf{x}_1)$$

Measures how well the model reconstructs the clean data from the least noisy intermediate $\mathbf{x}_1$. Often treated as an MSE or discretized Gaussian likelihood.

## From ELBO to $L_{\text{simple}}$

The DDPM paper (Ho et al., 2020) showed that the KL terms in the ELBO simplify to MSE losses between the true and predicted noise. After working through the math:

$$L_{t-1} \propto \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$$

They then proposed the **simplified loss** $L_{\text{simple}}$ — dropping the timestep-dependent weighting:

$$\boxed{L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]}$$

> 🔑 This is the loss function used in practice. It's simply "predict the noise, measure MSE." The elegant ELBO derivation justifies *why* this simple loss works — but in practice, the simple version trains better than the theoretically optimal weighted version.

### Why Does $L_{\text{simple}}$ Work Better?

The ELBO-derived weighting $\lambda(t)$ puts very high weight on small $t$ (low noise) and low weight on large $t$ (high noise). This causes the network to focus almost entirely on fine details and ignore global structure.

$L_{\text{simple}}$ uses uniform weighting ($\lambda(t) = 1$), which gives balanced attention to all noise levels. In practice, this produces significantly better samples, even though it's not the "correct" ELBO objective.

## The Big Picture

```
Goal: Maximize log p_θ(x₀)              ← intractable
         ↓
Maximize ELBO                            ← tractable lower bound
         ↓
Minimize sum of KL divergences           ← closed-form for Gaussians
         ↓
Minimize sum of MSE terms               ← simplify the KL
         ↓
Minimize E[‖ε - ε_θ(x_t, t)‖²]         ← L_simple (drop weights)
```

Each step is either an equality or a simplification that works well in practice.

## ELBO in Other Generative Models

The ELBO is not unique to diffusion — it's used in:
- **VAEs**: ELBO = reconstruction loss + KL regularization
- **Diffusion models**: ELBO = sum of denoising KL terms
- **Hierarchical VAEs**: ELBO = sum of KL terms across hierarchy levels

Diffusion models can be viewed as a special kind of hierarchical VAE with $T$ levels, where the encoder (forward process) is fixed and the latent variables have the same dimension as the data.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| ELBO | Evidence Lower Bound |
| $L_{\text{vlb}}$ | Variational lower bound loss (= negative ELBO, to be minimized) |
| $L_{\text{simple}}$ | Simplified loss (unweighted MSE) |
| $L_T$ | Prior matching term |
| $L_{t-1}$ | Denoising matching term at step $t$ |
| $L_0$ | Reconstruction term |
| Jensen's inequality | The mathematical inequality that gives us the lower bound |

## Knowledge Check ✅

1. Why can't we compute $\log p_\theta(\mathbf{x}_0)$ directly?
2. What mathematical inequality gives us the ELBO? Why does it apply here?
3. Name the three types of terms in the ELBO decomposition and explain what each measures.
4. Which term in the ELBO contains the learnable parameters?
5. How does the ELBO simplify to $L_{\text{simple}} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2]$?
6. Why does $L_{\text{simple}}$ (with uniform weighting) work better in practice than the "correct" ELBO weighting?
7. What is the relationship between the ELBO gap and KL divergence?
8. How does a diffusion model relate to a hierarchical VAE?

---
*Previous: [24-unet-architecture.md](24-unet-architecture.md)*

---

## 🎓 Congratulations!

You have completed all **26 beginner-level base concepts** (0–25). You now have the mathematical and conceptual vocabulary to understand the core ideas behind any diffusion model paper. 

**Next steps**: Proceed to the paper-specific deep dives in `1-papers/`, starting with DDPM (Ho et al., 2020), where all these concepts come together into a complete generative model.
