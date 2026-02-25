# Log-Likelihood

## What is it?

**Likelihood** measures how well a model explains observed data. **Log-likelihood** is simply the logarithm of the likelihood — we use the log version because it's mathematically much easier to work with.

## Starting with Likelihood

Given a model with parameters $\theta$ and observed data $\mathbf{x}$, the **likelihood** is:

$$\mathcal{L}(\theta) = p_\theta(\mathbf{x})$$

This reads: "Under model parameters $\theta$, how probable is the data $\mathbf{x}$?"

> ⚠️ Important distinction:
> - $p(\mathbf{x} \mid \theta)$ as a function of $\mathbf{x}$ → this is a **probability distribution** (over data)
> - $p(\mathbf{x} \mid \theta)$ as a function of $\theta$ → this is the **likelihood function** (over parameters)
> 
> Same formula, different perspective!

## Why Take the Log?

### Problem: Products of tiny numbers
If we have $N$ independent data points $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N$, the joint likelihood is:

$$\mathcal{L}(\theta) = \prod_{i=1}^{N} p_\theta(\mathbf{x}_i)$$

Each $p_\theta(\mathbf{x}_i)$ is a small number (e.g., $10^{-5}$). Multiplying thousands of them → the product becomes astronomically small, causing **numerical underflow** (the computer rounds to 0).

### Solution: Use logarithms
$$\log \mathcal{L}(\theta) = \sum_{i=1}^{N} \log p_\theta(\mathbf{x}_i)$$

The log turns **products into sums** — much more numerically stable and easier to optimize.

### Two bonus properties of log:
1. **Log is monotonic**: maximizing $\log \mathcal{L}$ is equivalent to maximizing $\mathcal{L}$ (the optimal $\theta$ is the same)
2. **Log of exponentials simplifies**: $\log(\exp(\cdot)) = \cdot$ — this is extremely useful since Gaussians involve $\exp(\cdot)$

## Maximum Likelihood Estimation (MLE)

The standard approach to training a model: find parameters $\theta$ that **maximize** the log-likelihood:

$$\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log p_\theta(\mathbf{x}_i)$$

Or equivalently, **minimize the negative log-likelihood (NLL)**:

$$\theta^* = \arg\min_\theta \left[-\sum_{i=1}^{N} \log p_\theta(\mathbf{x}_i)\right]$$

> 💡 In deep learning, we always **minimize** losses. So we minimize the **negative** log-likelihood.

## Connection to KL Divergence

Minimizing KL divergence between the data distribution and the model is equivalent to maximizing log-likelihood:

$$\min_\theta D_{\text{KL}}(p_{\text{data}} \,\|\, p_\theta) \iff \max_\theta \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_\theta(\mathbf{x})]$$

This is why log-likelihood and KL divergence appear interchangeably in diffusion papers — they lead to the same optimization.

## Why Log-Likelihood Matters for Diffusion

### The Problem
For diffusion models, computing $\log p_\theta(\mathbf{x})$ directly is **intractable** — it requires integrating over all possible noise trajectories:

$$p_\theta(\mathbf{x}_0) = \int p_\theta(\mathbf{x}_{0:T}) \, d\mathbf{x}_{1:T}$$

This integral is over an astronomically high-dimensional space (all intermediate noisy images).

### The Solution: ELBO
Instead of maximizing $\log p_\theta(\mathbf{x})$ directly, we maximize a **lower bound** on it — the **Evidence Lower Bound (ELBO)**:

$$\log p_\theta(\mathbf{x}) \geq \text{ELBO}(\theta)$$

By maximizing the ELBO, we indirectly push up the log-likelihood. (See [25-elbo.md](25-elbo.md) for details.)

### Log-likelihood as an Evaluation Metric
Some papers report **bits per dimension (BPD)** — a normalized version of negative log-likelihood — to compare generative models:

$$\text{BPD} = \frac{-\log_2 p_\theta(\mathbf{x})}{d}$$

where $d$ is the number of dimensions (e.g., pixels). Lower BPD = better model.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\log p_\theta(\mathbf{x})$ | Log-likelihood of data $\mathbf{x}$ under model $\theta$ |
| $-\log p_\theta(\mathbf{x})$ | Negative log-likelihood (NLL) — what we minimize |
| $\mathbb{E}_{p_{\text{data}}}[\log p_\theta(\mathbf{x})]$ | Expected log-likelihood over the data distribution |
| NLL | Negative log-likelihood |
| BPD | Bits per dimension |

## Knowledge Check ✅

1. Why do we use log-likelihood instead of likelihood directly?
2. What is the relationship between maximizing log-likelihood and minimizing KL divergence?
3. Why can't we compute $\log p_\theta(\mathbf{x})$ directly for diffusion models?
4. What is the ELBO, and how does it relate to log-likelihood?
5. What does "bits per dimension" measure?

---
*Previous: [7-kl-divergence.md](7-kl-divergence.md) · Next: [9-neural-network-as-function-approximator.md](9-neural-network-as-function-approximator.md)*
