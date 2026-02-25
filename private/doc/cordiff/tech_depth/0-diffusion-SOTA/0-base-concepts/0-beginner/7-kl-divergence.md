# KL Divergence (Kullback-Leibler Divergence)

## What is it?

**KL divergence** is a measure of how **different** two probability distributions are. It answers the question: "How much information do I lose if I use distribution $q$ to approximate the true distribution $p$?"

$$D_{\text{KL}}(p \,\|\, q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

Or equivalently:
$$D_{\text{KL}}(p \,\|\, q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right]$$

## Key Properties

### 1. Always Non-Negative
$$D_{\text{KL}}(p \,\|\, q) \geq 0$$

It equals zero **only** when $p$ and $q$ are the exact same distribution.

### 2. NOT Symmetric
$$D_{\text{KL}}(p \,\|\, q) \neq D_{\text{KL}}(q \,\|\, p) \quad \text{(in general)}$$

This means KL divergence is **not a true distance**. The order matters!

- $D_{\text{KL}}(p \,\|\, q)$: "How well does $q$ cover $p$?" (forward KL)
- $D_{\text{KL}}(q \,\|\, p)$: "How well does $p$ cover $q$?" (reverse KL)

### 3. Units
KL divergence is measured in **nats** (when using natural log $\ln$) or **bits** (when using $\log_2$). In ML, we almost always use natural log.

## Intuitive Understanding

Imagine $p$ is the true distribution of cat images, and $q$ is your model's distribution:

- **$D_{\text{KL}}(p \,\|\, q)$ is large** → your model $q$ assigns low probability to images that are common under $p$ (it's missing important cats!)
- **$D_{\text{KL}}(p \,\|\, q)$ is small** → your model $q$ assigns similar probabilities to all images as the true distribution $p$
- **$D_{\text{KL}}(p \,\|\, q) = 0$** → your model perfectly matches reality

## KL Divergence Between Two Gaussians

This is a formula you will see constantly in diffusion papers. For two univariate Gaussians:

$$D_{\text{KL}}\big(\mathcal{N}(\mu_1, \sigma_1^2) \,\|\, \mathcal{N}(\mu_2, \sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

For multivariate Gaussians with diagonal covariance:
$$D_{\text{KL}}\big(\mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1) \,\|\, \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)\big) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_2|}{|\boldsymbol{\Sigma}_1|} - d + \text{tr}(\boldsymbol{\Sigma}_2^{-1}\boldsymbol{\Sigma}_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^T \boldsymbol{\Sigma}_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)\right]$$

where $d$ is the dimensionality and $|\cdot|$ is the determinant.

> 💡 The key takeaway: KL divergence between Gaussians has a **closed-form formula** — no need for numerical integration. This is one reason Gaussians are so convenient in diffusion models.

## Why KL Divergence Matters for Diffusion

### 1. Training Objective (ELBO)
The training loss of diffusion models (DDPM) is derived from the **Evidence Lower Bound (ELBO)**, which decomposes into a sum of KL divergences:

$$L = \sum_{t=1}^{T} D_{\text{KL}}\big(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)\big)$$

Each term measures: "How close is our learned reverse step to the true reverse step?"

### 2. Comparing Distributions
When we want to know if our generative model's distribution $p_\theta$ matches the data distribution $p_{\text{data}}$, KL divergence is the natural measure.

### 3. The Final Step
At the end of the forward process, $q(\mathbf{x}_T)$ should be close to $\mathcal{N}(\mathbf{0}, \mathbf{I})$. We measure this with:
$$D_{\text{KL}}\big(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I})\big)$$

If the noise schedule is designed well, this is nearly zero.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $D_{\text{KL}}(p \,\|\, q)$ | KL divergence from $p$ to $q$ |
| $\text{KL}(p \,\|\, q)$ | Same thing, shorter notation |
| $D_{\text{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \,\|\, p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t))$ | Per-step loss term in DDPM |

## Knowledge Check ✅

1. What does $D_{\text{KL}}(p \,\|\, q) = 0$ mean?
2. Is KL divergence symmetric? Why does this matter?
3. Why is the closed-form KL between Gaussians useful for diffusion models?
4. In the ELBO decomposition, what are we comparing with each KL term?
5. What should $D_{\text{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \,\|\, \mathcal{N}(\mathbf{0}, \mathbf{I}))$ be, and why?

### Answers

1. **It means $p$ and $q$ are the exact same distribution.** There is zero information loss when using $q$ to approximate $p$. Since KL divergence is always $\geq 0$, a value of 0 is the best possible — it means the two distributions assign identical probabilities to every possible outcome.

2. **No, KL divergence is NOT symmetric:** $D_{\text{KL}}(p \,\|\, q) \neq D_{\text{KL}}(q \,\|\, p)$ in general. This matters because the two directions have different behaviors:
   - **Forward KL** $D_{\text{KL}}(p \,\|\, q)$: penalizes $q$ heavily wherever $p$ has mass but $q$ doesn't → encourages $q$ to be **mode-covering** (spread out to cover all of $p$).
   - **Reverse KL** $D_{\text{KL}}(q \,\|\, p)$: penalizes $q$ heavily wherever $q$ has mass but $p$ doesn't → encourages $q$ to be **mode-seeking** (concentrate on the highest-density regions of $p$).
   
   In diffusion models, the ELBO uses a specific direction at each step, and which direction matters for the behavior of the trained model.

3. **Because both the true reverse posterior $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ and the learned reverse $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ are Gaussians.** The KL divergence between two Gaussians has a closed-form formula — no sampling or numerical integration needed. This makes each term in the ELBO loss exactly computable. It further simplifies to an MSE between the predicted and true means (since the variances are often fixed), which is why the final training loss is just $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$.

4. **Each KL term compares the true (tractable) reverse step $q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0)$ against the learned reverse step $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$.** In other words: "At timestep $t$, how close is the network's predicted denoising distribution to the actual optimal denoising distribution?" Minimizing these KL terms trains the network to denoise correctly at every noise level.

5. **It should be approximately zero.** This term measures how close the fully noised data $q(\mathbf{x}_T \mid \mathbf{x}_0)$ is to pure Gaussian noise $\mathcal{N}(\mathbf{0}, \mathbf{I})$. If the noise schedule is designed properly (enough steps, large enough total noise), the forward process completely destroys the signal by step $T$, so $q(\mathbf{x}_T \mid \mathbf{x}_0) \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$ and the KL is nearly zero. This is important because the reverse process *starts* from $\mathcal{N}(\mathbf{0}, \mathbf{I})$ — if the forward process doesn't actually reach pure noise, there's a mismatch at the starting point of generation.

---
*Previous: [6-bayes-theorem.md](6-bayes-theorem.md) · Next: [8-log-likelihood.md](8-log-likelihood.md)*
