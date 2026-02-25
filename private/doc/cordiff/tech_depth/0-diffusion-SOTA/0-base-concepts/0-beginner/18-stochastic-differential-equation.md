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

### Answers

1. **The two components are the drift and the diffusion:**
   - **Drift** $f(\mathbf{x}, t)\,dt$ — the deterministic force that systematically pushes the state in a particular direction. If the diffusion coefficient were zero, the SDE would reduce to an ODE and the state would follow a smooth, predictable trajectory. In diffusion models, the drift typically shrinks the data toward zero (e.g., $-\frac{1}{2}\beta(t)\mathbf{x}$ in VP-SDE).
   - **Diffusion** $g(t)\,d\mathbf{w}$ — the stochastic component that injects random Gaussian noise at each infinitesimal time step. The coefficient $g(t)$ controls the magnitude of the randomness, and $d\mathbf{w}$ is the Wiener process increment providing the actual source of randomness. In diffusion models, this is what progressively corrupts the data with noise.
   
   Together, $d\mathbf{x} = f(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$ describes a system evolving under both systematic forces and random fluctuations — like a leaf on a river (current = drift, turbulence = diffusion).

2. **The Wiener process $\mathbf{w}(t)$ (also called Brownian motion) is a continuous-time random process.** Its key properties are:
   - $\mathbf{w}(0) = \mathbf{0}$ (starts at the origin)
   - **Increments are Gaussian**: $\mathbf{w}(t + \Delta t) - \mathbf{w}(t) \sim \mathcal{N}(\mathbf{0}, \Delta t\,\mathbf{I})$ — the variance scales linearly with the time interval
   - **Independent increments**: non-overlapping time intervals produce independent random values
   - **Continuous but nowhere differentiable** paths (infinitely jagged)
   
   It is the continuous-time analog of a random walk. In diffusion models, the Wiener process is the mathematical formalization of "keep adding small Gaussian noise at every step" — taken to the limit of infinitely many infinitesimally small steps.

3. **The DDPM discrete forward process is a discretization of the VP-SDE; as the number of steps $T \to \infty$, the discrete chain converges to the continuous VP-SDE.** The DDPM step:
   $$\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t$$
   simultaneously scales down the signal (by $\sqrt{1 - \beta_t}$) and adds noise (scaled by $\sqrt{\beta_t}$). In the continuous limit, these two effects become the drift $-\frac{1}{2}\beta(t)\mathbf{x}\,dt$ (shrinking the data) and diffusion $\sqrt{\beta(t)}\,d\mathbf{w}$ (adding noise), giving the VP-SDE:
   $$d\mathbf{x} = -\tfrac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}$$
   The "variance preserving" name comes from the fact that the drift and diffusion are balanced so the total variance remains bounded throughout the process, rather than growing without limit.

4. **The reverse SDE requires the score function $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ at every noise level $t$.** Anderson's 1982 result shows the reverse SDE is:
   $$d\mathbf{x} = \left[f(\mathbf{x}, t) - g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt + g(t)\,d\bar{\mathbf{w}}$$
   
   The drift and diffusion coefficient $f$ and $g$ are known from the forward process design, but the score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$ — the gradient of the log marginal density of the noisy data at time $t$ — is unknown. This is precisely what the neural network learns: $\mathbf{s}_\theta(\mathbf{x}, t) \approx \nabla_{\mathbf{x}} \log p_t(\mathbf{x})$. Once trained, plugging the network's score estimate into the reverse SDE lets us solve it from $t = T$ (pure noise) back to $t = 0$ (clean data), generating samples.

5. **VP-SDE (Variance Preserving)** has both drift and diffusion: $d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{w}$. The drift shrinks the data while noise is added, keeping total variance bounded. It corresponds to **DDPM** (Ho et al., 2020).
   
   **VE-SDE (Variance Exploding)** has no drift — only diffusion: $d\mathbf{x} = \sqrt{\frac{d[\sigma^2(t)]}{dt}}\,d\mathbf{w}$. The data is not scaled, so noise simply accumulates and the variance grows without bound ("explodes"). It corresponds to **SMLD/NCSN** (Song & Ermon, 2019) — Score Matching with Langevin Dynamics / Noise Conditional Score Networks.
   
   The key difference: VP-SDE preserves variance by balancing signal decay with noise injection; VE-SDE lets variance grow by only adding noise. Both approaches work for generation, just with different noise schedule behaviors.

6. **The SDE framework was a landmark because it unified two previously separate research threads into a single mathematical formalism and enabled powerful new capabilities:**
   - **Unification**: DDPM (discrete, noise prediction) and NCSN (discrete noise levels, score matching) were shown to be special cases of the same continuous-time SDE — VP-SDE and VE-SDE respectively. This connected two communities and clarified that they were solving the same problem with different notation.
   - **Flexible sampling**: The continuous-time view opened up a rich toolkit of numerical solvers — SDE solvers (stochastic, diverse samples), ODE solvers (deterministic, faster), and predictor-corrector methods (combining both). This was impossible in the fixed-step discrete framework.
   - **Exact likelihood**: The probability flow ODE derived from the SDE enables exact log-likelihood computation via the continuous change-of-variables formula, something the discrete DDPM could not do.
   - **New noise schedules**: Working in continuous time makes it easy to design and interpolate between different noise schedules without committing to a fixed number of discrete steps.

7. **The Euler-Maruyama method is the simplest numerical scheme for solving SDEs — the stochastic analog of the Euler method for ODEs.** It discretizes time into finite steps $\Delta t$ and approximates:
   $$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + f(\mathbf{x}_t, t)\,\Delta t + g(t)\sqrt{\Delta t}\,\mathbf{z}, \quad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   
   The drift is approximated as $f \cdot \Delta t$ (forward Euler), and the Wiener increment $d\mathbf{w}$ is approximated as $\sqrt{\Delta t}\,\mathbf{z}$ (since Wiener increments have variance $\Delta t$). It's first-order accurate: halving $\Delta t$ roughly halves the error. More sophisticated solvers like the Heun method (used in EDM/Karras et al., 2022) achieve better accuracy with fewer steps, which is critical for fast sampling in diffusion models.

---
*Previous: [17-score-function.md](17-score-function.md) · Next: [19-ordinary-differential-equation.md](19-ordinary-differential-equation.md)*
