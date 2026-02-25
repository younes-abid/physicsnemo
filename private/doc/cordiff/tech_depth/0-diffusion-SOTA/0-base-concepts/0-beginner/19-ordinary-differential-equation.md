# Ordinary Differential Equation (ODE)

## What is it?

An **Ordinary Differential Equation (ODE)** describes how a quantity changes over time in a **deterministic** (non-random) way. Given the current state and the rate of change, the future is completely determined — no randomness involved.

$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t)$$

This reads: "The rate of change of $\mathbf{x}$ at time $t$ is given by the function $f$."

Given an initial condition $\mathbf{x}(0) = \mathbf{x}_0$, the solution $\mathbf{x}(t)$ is a **unique, smooth trajectory** through time.

## ODE vs. SDE

| Property | ODE | SDE |
|----------|-----|-----|
| **Randomness** | None — fully deterministic | Has a random noise term |
| **Same input → same output?** | Yes, always | No, different each time |
| **Trajectory** | One smooth path | A different noisy path every run |
| **Formula** | $d\mathbf{x} = f(\mathbf{x},t)\,dt$ | $d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{w}$ |

An ODE is simply an SDE with the noise term set to zero ($g = 0$). See [18-stochastic-differential-equation.md](18-stochastic-differential-equation.md).

## The Probability Flow ODE

This is where ODEs become crucial for diffusion models. Song et al. (2021) showed that for **any** forward SDE:

$$d\mathbf{x} = f(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{w}$$

there exists a corresponding **deterministic ODE** that produces the **exact same marginal distributions** $p_t(\mathbf{x})$ at every time $t$:

$$d\mathbf{x} = \left[f(\mathbf{x}, t) - \frac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$$

This is called the **Probability Flow ODE** (PF-ODE).

> 🔑 Key insight: The probability flow ODE and the reverse SDE produce the **same distribution** of outputs, but the ODE does it **deterministically**. Same noise input → same output, every time.

## Why the Probability Flow ODE Matters

### 1. Deterministic Sampling
With the SDE, each run produces a different sample (because of the noise $d\mathbf{w}$). With the PF-ODE, given the same starting noise $\mathbf{x}_T$, you always get the **same output** $\mathbf{x}_0$.

This enables:
- **Reproducibility**: same seed → same generated sample
- **Interpolation**: smoothly interpolate between two noise inputs to get smooth transitions between generated images
- **Encoding**: run the ODE forward to map a data point to its latent representation (invertible!)

### 2. Faster Sampling
ODE solvers can take **larger steps** than SDE solvers because there's no noise to fight against. Advanced ODE solvers (Heun, RK45, DPM-Solver) can generate good samples in as few as **10-50 steps**, compared to 1000 steps for the original DDPM SDE sampler.

This is one of the most practically important benefits — it directly addresses the main weakness of diffusion models (slow generation).

### 3. Exact Likelihood Computation
The PF-ODE is a **continuous normalizing flow** — an invertible mapping between data and noise. Using the instantaneous change-of-variables formula, we can compute exact log-likelihoods:

$$\log p_0(\mathbf{x}_0) = \log p_T(\mathbf{x}_T) + \int_0^T \nabla \cdot \tilde{f}(\mathbf{x}_t, t)\,dt$$

This is not possible with the discrete DDPM formulation (which only gives a lower bound via the ELBO).

## Numerical ODE Solvers

Since we can rarely solve ODEs analytically, we use **numerical solvers**:

### Euler Method (Simplest)
$$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + f(\mathbf{x}_t, t) \cdot \Delta t$$

Simple but needs many small steps for accuracy.

### Heun's Method (2nd Order)
Used in EDM (Karras et al., 2022):
1. Predict: $\hat{\mathbf{x}} = \mathbf{x}_t + f(\mathbf{x}_t, t) \cdot \Delta t$
2. Correct: $\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \frac{\Delta t}{2}\left[f(\mathbf{x}_t, t) + f(\hat{\mathbf{x}}, t+\Delta t)\right]$

Better accuracy with the same step size — EDM uses this as its default sampler.

### Higher-Order Solvers
- **RK45** (Runge-Kutta): adaptive step size, very accurate
- **DPM-Solver** (Lu et al., 2022): specifically designed for diffusion PF-ODEs, achieves great quality in 10-20 steps

## DDIM as an ODE Solver

**DDIM** (Song et al., 2020) can be understood as an **Euler solver** for the probability flow ODE in disguise! The DDIM update rule:

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{\mathbf{x}}_0 + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \hat{\boldsymbol{\epsilon}}_\theta$$

is equivalent to taking an Euler step along the PF-ODE trajectory when the stochasticity parameter $\eta = 0$.

This connection was made explicit by the Score SDE paper (Song et al., 2021).

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t)$ | General ODE |
| Probability Flow ODE (PF-ODE) | The deterministic ODE equivalent to the reverse SDE |
| $\tilde{f}(\mathbf{x}, t)$ | The drift of the PF-ODE (includes the score term) |
| Euler, Heun, RK45 | Numerical ODE solvers of increasing sophistication |
| NFE | Number of Function Evaluations — how many times the network is called during sampling |

## Knowledge Check ✅

1. What is the key difference between an ODE and an SDE?
2. What is the Probability Flow ODE, and how does it relate to the reverse SDE?
3. Name three advantages of using the PF-ODE over the reverse SDE for generation.
4. Why can ODE solvers take fewer steps than SDE solvers?
5. How does DDIM relate to the Probability Flow ODE?
6. What ODE solver does EDM (Karras et al., 2022) use by default?
7. What does "NFE" measure, and why does it matter?

### Answers

1. **Determinism vs. randomness.** An ODE is fully **deterministic** — given the same initial condition, the trajectory is always identical. An SDE includes a **random noise term** ($g(t)\,d\mathbf{w}$), so every run produces a different trajectory even from the same starting point. Mathematically, an ODE is an SDE with $g = 0$.

2. **The Probability Flow ODE** is a deterministic ODE whose solution traces out the **same marginal distributions** $p_t(\mathbf{x})$ at every time $t$ as the corresponding reverse SDE. It is derived by replacing the stochastic noise term in the reverse SDE with an additional deterministic drift that depends on the score $\nabla_{\mathbf{x}} \log p_t(\mathbf{x})$:
$$d\mathbf{x} = \left[f(\mathbf{x}, t) - \tfrac{1}{2}g(t)^2 \nabla_{\mathbf{x}} \log p_t(\mathbf{x})\right] dt$$
The distributions match, but individual trajectories differ (deterministic vs. stochastic).

3. **Three advantages of the PF-ODE:**
   - **Deterministic sampling / reproducibility:** same starting noise always yields the same output, enabling reproducibility, smooth interpolation in latent space, and invertible encoding.
   - **Faster sampling:** ODE solvers can take larger steps (no noise to fight against), so high-quality samples can be produced in 10–50 steps instead of ~1000.
   - **Exact likelihood computation:** the PF-ODE defines a continuous normalizing flow, allowing exact log-likelihood computation via the instantaneous change-of-variables formula (rather than only a lower bound via the ELBO).

4. **Because there is no stochastic noise to accumulate errors.** SDE solvers must take small steps to keep the injected noise from compounding into large errors. ODE trajectories are smooth and predictable, so higher-order solvers (Heun, RK45, DPM-Solver) can take much larger steps while maintaining accuracy.

5. **DDIM is an Euler solver for the Probability Flow ODE.** When the stochasticity parameter $\eta = 0$, the DDIM update rule is mathematically equivalent to taking a single Euler discretization step along the PF-ODE trajectory. This connection was made explicit by Song et al. (2021) in the Score SDE paper.

6. **Heun's method (a 2nd-order solver).** It performs a predict step and then a correction step, giving better accuracy per step than the simple Euler method.

7. **NFE = Number of Function Evaluations** — the number of times the neural network (denoiser / score model) is called during sampling. It matters because each function evaluation is the dominant computational cost of generation. Lower NFE means faster generation; Heun's method uses 2 NFEs per step, so $N$ Heun steps = $2N$ NFEs.

---
*Previous: [18-stochastic-differential-equation.md](18-stochastic-differential-equation.md) · Next: [20-denoising.md](20-denoising.md)*
