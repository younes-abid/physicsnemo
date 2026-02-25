# Stochastic Differential Equation (SDE)

## What is it?

A **Stochastic Differential Equation (SDE)** is a differential equation that includes a **random noise term**. It describes how a system evolves over time when subject to both deterministic forces and random fluctuations.

The general form of an SDE is:

$$d\mathbf{x} = \underbrace{f(\mathbf{x}, t)}_{\text{drift}} \, dt + \underbrace{g(t)}_{\text{diffusion coefficient}} \, d\mathbf{w}$$

Where:
- $\mathbf{x}$ — the state (e.g., an image)
- $t$ — continuous time
- $f(\mathbf{x}, t)$ — the **drift** term: a deterministic force pushing $\mathbf{x}$ in a particular direction
- $g(t)$ — the **diffusion coefficient**: controls how much randomness is injected
- $d\mathbf{w}$ — an infinitesimal **Wiener process** (Brownian motion) increment: the source of randomness

## Breaking It Down Piece by Piece

### The Drift: $f(\mathbf{x}, t) \, dt$
This is the "predictable" part. If there were no noise ($g = 0$), the SDE would reduce to an ordinary differential equation (ODE), and $\mathbf{x}$ would follow a smooth, deterministic path.

Think of it as a river current: it pushes the state in a systematic direction.

### The Diffusion: $g(t) \, d\mathbf{w}$
This is the "random" part. The Wiener process $\mathbf{w}$ is a continuous-time random walk — at each infinitesimal time step, a small random Gaussian perturbation is added.

Think of it as turbulence on top of the river current: unpredictable jitter.

### The Wiener Process ($\mathbf{w}$)
Also called **Brownian motion**. Key properties:
- $\mathbf{w}(0) = \mathbf{0}$
- Increments $\mathbf{w}(t + \Delta t) - \mathbf{w}(t) \sim \mathcal{N}(\mathbf{0}, \Delta t \, \mathbf{I})$
- Increments at non-overlapping time intervals are independent

> 💡 The Wiener process is the continuous-time analog of repeatedly adding small Gaussian noise — exactly what diffusion models do!

## From Discrete Steps to Continuous Time

The DDPM forward process uses **discrete** timesteps $t = 0, 1, 2, \ldots, T$:

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t$$

As the number of steps $T \to \infty$ and each step becomes infinitesimally small, this discrete chain converges to a **continuous-time SDE**. The SDE is the limiting case of the discrete Markov chain.

This is the key insight of Song et al. (2021) — "Score-Based Generative Modeling through Stochastic Differential Equations."

## SDEs in Diffusion Models

### The Forward SDE (Adding Noise)
The forward process (data → noise) is described by an SDE:

$$d\mathbf{x} = f(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$$

Two important special cases:

#### VP-SDE (Variance Preserving) — corresponds to DDPM
$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}$$

- The drift $-\frac{1}{2}\beta(t)\mathbf{x}$ shrinks the data toward zero
- The diffusion $\sqrt{\beta(t)}$ adds noise
- Together, they keep the total variance bounded (hence "variance preserving")

#### VE-SDE (Variance Exploding) — corresponds to SMLD/NCSN
$$d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,d\mathbf{w}$$

- No drift term (data is not scaled)
- Only noise is added, so the variance grows ("explodes") over time

### The Reverse SDE (Removing Noise)
A remarkable result by Anderson (1982) shows that **every forward SDE has a corresponding reverse SDE**:

$$d\mathbf{x} = \left[f(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t)\,d\bar{\mathbf{w}}$$

Where:
- $d\bar{\mathbf{w}}$ is a reverse-time Wiener process
- $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ is the **score function** (see [17-score-function.md](17-score-function.md)) at time $t$

> 🔑 The reverse SDE requires knowing the **score** $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ at every noise level. This is what the neural network learns! Once we have an estimate $\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$, we can solve the reverse SDE to generate data.

## Why SDEs Matter for Diffusion

### 1. Unification
The SDE framework by Song et al. (2021) unified two previously separate lines of work:
- **DDPM** (Ho et al., 2020) → discrete-time, noise prediction → VP-SDE in the continuous limit
- **NCSN/SMLD** (Song & Ermon, 2019) → discrete noise levels, score matching → VE-SDE in the continuous limit

Both are special cases of the same SDE framework!

### 2. Flexible Sampling
With the SDE framework, you can choose **different numerical solvers** to solve the reverse SDE:
- **SDE solvers** (e.g., Euler-Maruyama): stochastic, give diverse samples
- **ODE solvers** (using the probability flow ODE — see [19-ordinary-differential-equation.md](19-ordinary-differential-equation.md)): deterministic, faster
- **Predictor-corrector methods**: combine both for better quality

### 3. Exact Likelihood Computation
The probability flow ODE (derived from the SDE) allows exact log-likelihood computation via the change-of-variables formula — something not possible with the discrete DDPM.

## Discrete Approximation: Euler-Maruyama

To actually solve an SDE on a computer, we discretize time into small steps $\Delta t$:

$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + f(\mathbf{x}_t, t)\Delta t + g(t)\sqrt{\Delta t}\,\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This is called the **Euler-Maruyama method** — the simplest SDE solver. More sophisticated solvers (Heun, RK methods) give better accuracy with fewer steps.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $d\mathbf{x} = f\,dt + g\,d\mathbf{w}$ | Forward SDE |
| $f(\mathbf{x}, t)$ | Drift coefficient |
| $g(t)$ | Diffusion coefficient |
| $d\mathbf{w}$ | Wiener process increment |
| VP-SDE | Variance Preserving SDE (DDPM family) |
| VE-SDE | Variance Exploding SDE (SMLD/NCSN family) |
| Reverse SDE | The time-reversed SDE used for generation |

## Knowledge Check ✅

1. What are the two main components of an SDE? What does each represent?
2. What is the Wiener process, and what distribution do its increments follow?
3. How does the DDPM discrete forward process relate to the VP-SDE?
4. What does the reverse SDE need to know in order to run backward in time?
5. What are the VP-SDE and VE-SDE, and which prior methods do they correspond to?
6. Why was the SDE framework (Song et al., 2021) such an important contribution?
7. What is the Euler-Maruyama method?

---
*Previous: [17-score-function.md](17-score-function.md) · Next: [19-ordinary-differential-equation.md](19-ordinary-differential-equation.md)*
