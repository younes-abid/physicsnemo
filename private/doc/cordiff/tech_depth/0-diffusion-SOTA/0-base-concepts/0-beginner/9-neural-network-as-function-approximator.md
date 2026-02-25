# Neural Network as Function Approximator

## What is it?

A **neural network** is a parametric function $f_\theta(\cdot)$ that can learn to approximate virtually any mapping from inputs to outputs, given enough data and capacity.

In mathematical terms:
$$y = f_\theta(x)$$

Where:
- $x$ is the input (e.g., a noisy image)
- $y$ is the output (e.g., predicted noise, or predicted clean image)
- $\theta$ represents all the learnable **parameters** (weights and biases) of the network

## Why Do We Need Function Approximators?

In many problems, we know a function **exists** but we **don't know its formula**. For example:

| Problem | Unknown Function | Input | Output |
|---------|-----------------|-------|--------|
| Image classification | "What object is in this image?" | Image pixels | Class label |
| Translation | "What is this sentence in French?" | English sentence | French sentence |
| **Denoising** | "What was the original clean image?" | Noisy image | Clean image (or the noise) |

For diffusion models specifically, the unknown function is:
> "Given a noisy image $\mathbf{x}_t$ at noise level $t$, predict the noise $\boldsymbol{\epsilon}$ that was added."

We don't have a formula for this — we **learn** it from data.

## The Universal Approximation Theorem

A foundational result in ML states that a neural network with enough neurons can approximate **any continuous function** to arbitrary accuracy.

This is why neural networks are the go-to tool: whatever function we need, a sufficiently large network can learn to approximate it.

## How Learning Works (In Brief)

1. **Initialize** $\theta$ randomly
2. **Forward pass**: compute $\hat{y} = f_\theta(x)$ (the prediction)
3. **Compute loss**: measure how far $\hat{y}$ is from the true answer $y$
   - e.g., $L = \|\hat{y} - y\|^2$ (mean squared error — see [15-loss-function.md](15-loss-function.md))
4. **Backward pass**: compute gradients $\nabla_\theta L$ (how to adjust each parameter to reduce the loss)
5. **Update**: $\theta \leftarrow \theta - \eta \nabla_\theta L$ (gradient descent, where $\eta$ is the learning rate)
6. **Repeat** for many data samples until the loss is small

## The $\theta$ Notation

Everywhere in diffusion papers, you'll see subscript $\theta$:
- $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ — a neural network that predicts noise, parameterized by $\theta$
- $\mathbf{s}_\theta(\mathbf{x}_t, t)$ — a neural network that predicts the score function
- $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ — a probability distribution whose parameters come from a neural network
- $D_\theta(\mathbf{x}_t, t)$ — a denoiser network (EDM notation)

The subscript $\theta$ is a reminder that **these are learned functions**, not fixed formulas.

## What the Network Learns in Diffusion Models

Depending on the paper/formulation, the network is trained to predict different things:

| Parameterization | Network predicts | Used in |
|-----------------|-----------------|---------|
| **Noise prediction** ($\boldsymbol{\epsilon}$-prediction) | The noise $\boldsymbol{\epsilon}$ that was added | DDPM |
| **Score prediction** ($\mathbf{s}$-prediction) | The score $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ | Score-based models (SMLD) |
| **Data prediction** ($\mathbf{x}_0$-prediction) | The clean data $\mathbf{x}_0$ | Some formulations |
| **Denoiser** ($D_\theta$) | The denoised data directly | EDM |
| **Velocity prediction** ($\mathbf{v}$-prediction) | A combination of noise and data | Progressive Distillation |

> 💡 All these parameterizations are **mathematically equivalent** — you can convert between them. They just affect training stability and practical performance. This will be explained when we cover each paper.

## Why This Matters for Diffusion

The neural network is the **only learned component** in a diffusion model. Everything else (the forward process, the noise schedule, the sampling algorithm) is **fixed by design**. The network's job is to learn the one thing we can't compute analytically: how to reverse the noise corruption.

## Knowledge Check ✅

1. What does the subscript $\theta$ mean in $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$?
2. Why can't we use a fixed formula instead of a neural network for denoising?
3. What is the Universal Approximation Theorem, and why is it relevant?
4. Name three different things a diffusion network can be trained to predict.
5. In a diffusion model, what is learned and what is fixed by design?

### Answers

1. **The subscript $\theta$ indicates that this is a learned (parametric) function, not a fixed formula.** $\theta$ represents all the trainable weights and biases of the neural network. Writing $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ means: "a function that takes noisy data $\mathbf{x}_t$ and timestep $t$ as input, and outputs a noise prediction, where the function's behavior is determined by learned parameters $\theta$." Different values of $\theta$ give different functions — training finds the $\theta$ that makes the best predictions.

2. **Because the mapping from noisy images to noise (or clean images) is incredibly complex and data-dependent.** There's no closed-form formula that says "given these specific noisy pixel values at this noise level, the original image was X." The relationship depends on the entire structure of natural images (or weather fields) — edges, textures, spatial correlations, physical constraints — which can only be captured by learning from thousands of examples. A neural network learns these patterns from data.

3. **The Universal Approximation Theorem states that a neural network with sufficient capacity (enough neurons) can approximate any continuous function to arbitrary accuracy.** This is relevant because it gives us confidence that whatever the true denoising function looks like — no matter how complex — a sufficiently large neural network *can* in principle learn it. It's the theoretical justification for using neural networks as general-purpose function approximators in diffusion models (and ML broadly).

4. Three different prediction targets:
   - **Noise prediction** ($\boldsymbol{\epsilon}$-prediction): predict the Gaussian noise $\boldsymbol{\epsilon}$ that was added to the clean data (used in DDPM).
   - **Score prediction** ($\mathbf{s}$-prediction): predict the score function $\nabla_{\mathbf{x}} \log p(\mathbf{x})$, which points toward higher-density regions (used in score-based/SMLD models).
   - **Data prediction** ($\mathbf{x}_0$-prediction): directly predict the clean data $\mathbf{x}_0$ from the noisy input.
   
   (Others include velocity prediction and direct denoiser output in EDM.) All these parameterizations are mathematically equivalent — you can convert between them — but they differ in training stability and practical performance.

5. **Learned: the neural network** $f_\theta$ (the denoiser/noise predictor) — this is the *only* component that is trained from data. **Fixed by design: everything else** — the forward noising process $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$, the noise schedule ($\beta_t$ or $\sigma_t$), and the sampling/generation algorithm (e.g., DDPM ancestral sampling, DDIM, or ODE solvers). The entire framework is carefully designed so that the network only needs to learn one thing: how to reverse the noise corruption at each step.

---
*Previous: [8-log-likelihood.md](8-log-likelihood.md) · Next: [10-noise.md](10-noise.md)*
