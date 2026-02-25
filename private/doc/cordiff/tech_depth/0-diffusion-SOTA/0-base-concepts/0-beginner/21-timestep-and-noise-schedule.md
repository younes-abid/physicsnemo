# Timestep & Noise Schedule

## What is it?

The **noise schedule** defines **how much noise** is added at each step of the forward process. It is a sequence (or continuous function) that controls the transition from clean data to pure noise.

The **timestep** $t$ is an index that tells you where you are in this process:
- $t = 0$: clean data (no noise)
- $t = T$: (approximately) pure Gaussian noise
- Intermediate $t$: partially noisy data

The noise schedule is a **design choice** — it is not learned. Different schedules lead to different quality, speed, and stability.

## DDPM Notation: $\beta_t$ and $\bar{\alpha}_t$

### Step-level noise: $\beta_t$

In DDPM, each forward step adds noise controlled by $\beta_t \in (0, 1)$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

$\beta_t$ is small (typically $10^{-4}$ to $0.02$) — each step adds just a tiny bit of noise.

### Cumulative noise: $\alpha_t$ and $\bar{\alpha}_t$

To avoid computing $T$ sequential steps, we define:

$$\alpha_t = 1 - \beta_t$$

$$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s = \prod_{s=1}^{t} (1 - \beta_s)$$

$\bar{\alpha}_t$ is the **cumulative product** — it tells you how much of the original signal survives after $t$ steps.

This gives us the crucial **closed-form** for jumping from $\mathbf{x}_0$ directly to $\mathbf{x}_t$:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

### The schedule curve

As $t$ goes from $0$ to $T$:
- $\bar{\alpha}_t$ decreases from $\approx 1$ to $\approx 0$
- Signal coefficient $\sqrt{\bar{\alpha}_t}$ decreases from $\approx 1$ to $\approx 0$
- Noise coefficient $\sqrt{1-\bar{\alpha}_t}$ increases from $\approx 0$ to $\approx 1$
- SNR $= \bar{\alpha}_t / (1-\bar{\alpha}_t)$ decreases from $\infty$ to $0$

```
Signal strength    Noise strength
     1 ┐                    ┌ 1
       │\                  /│
       │  \              /  │
       │    \          /    │
       │      \      /      │
     0 ┘        ────        └ 0
       t=0              t=T
```

## Common Noise Schedules

### 1. Linear Schedule (DDPM, Ho et al. 2020)
$$\beta_t = \beta_{\min} + \frac{t-1}{T-1}(\beta_{\max} - \beta_{\min})$$

- Original DDPM: $\beta_1 = 10^{-4}$, $\beta_T = 0.02$, $T = 1000$
- Simple but not optimal — too much noise is added too quickly in the middle of the process

### 2. Cosine Schedule (Nichol & Dhariwal, 2021 — Improved DDPM)
$$\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2$$

- Designed to have a more gradual transition, especially at high noise levels
- Prevents the signal from being destroyed too quickly
- $s = 0.008$ is a small offset to prevent $\beta_t$ from being too small near $t = 0$

### 3. EDM Schedule (Karras et al., 2022)
Instead of $\beta_t$, EDM works directly with the noise level $\sigma(t)$:

$$\sigma(t) = t \quad \text{(or more generally, some monotonic function of } t\text{)}$$

The timesteps for sampling are chosen as:

$$\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho$$

with $\rho = 7$ giving more steps at low noise (where fine details are generated).

### 4. Continuous Schedules (Score SDE)
In the SDE framework, $\beta(t)$ becomes a **continuous function** of time $t \in [0, 1]$:
- VP-SDE: $\beta(t) = \beta_{\min} + t(\beta_{\max} - \beta_{\min})$
- VE-SDE: $\sigma(t)$ is a continuous increasing function

## Why the Schedule Matters

### 1. Sample Quality
A bad schedule can waste steps on noise levels where nothing interesting happens, or rush through levels where fine details are generated.

### 2. Training Stability
If too much noise is added too early, the network struggles to learn meaningful denoising at intermediate steps. The cosine schedule was specifically designed to fix this issue.

### 3. Sampling Efficiency
During generation, you don't need to use all $T$ timesteps. You can select a **subset** of timesteps (e.g., 50 out of 1000). How you space these steps — uniform, concentrated at high/low noise — dramatically affects quality.

EDM's insight: put **more steps near low noise** (where details are generated) and **fewer steps at high noise** (where only coarse structure changes).

### 4. SNR Distribution
The schedule determines the distribution of SNR values during training. Some papers argue that sampling $t$ uniformly leads to an unbalanced emphasis — too many training examples at extreme noise levels, not enough in the middle.

## Discrete vs. Continuous Time

| Aspect | Discrete (DDPM) | Continuous (Score SDE, EDM) |
|--------|-----------------|---------------------------|
| Time | $t \in \{0, 1, \ldots, T\}$ | $t \in [0, T]$ or $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ |
| Schedule | Sequence $\beta_1, \ldots, \beta_T$ | Function $\beta(t)$ or $\sigma(t)$ |
| Typical $T$ | 1000 | Arbitrary (continuous) |
| Sampling steps | Subset of $\{0, \ldots, T\}$ | Any set of values in $[\sigma_{\min}, \sigma_{\max}]$ |

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $\beta_t$ | Noise variance added at step $t$ (DDPM) |
| $\alpha_t = 1 - \beta_t$ | Signal retention at step $t$ |
| $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ | Cumulative signal retention through step $t$ |
| $\sigma(t)$ or $\sigma_t$ | Noise standard deviation at time $t$ (Score SDE / EDM) |
| $T$ | Total number of timesteps (discrete) or final time (continuous) |
| $t \sim \text{Uniform}(1, T)$ | Random timestep sampling during training |

## Knowledge Check ✅

1. What is the purpose of the noise schedule?
2. What do $\beta_t$, $\alpha_t$, and $\bar{\alpha}_t$ represent in DDPM?
3. Write the formula for sampling $\mathbf{x}_t$ directly from $\mathbf{x}_0$ (without going through intermediate steps).
4. Why was the cosine schedule introduced? What problem does it fix?
5. How does EDM's approach to the noise schedule differ from DDPM's?
6. Why should sampling steps be concentrated near low noise levels during generation?
7. What is the difference between the discrete-time and continuous-time treatment of the schedule?

### Answers

1. The noise schedule defines **how much noise is added at each step** of the forward process, controlling the transition from clean data to pure Gaussian noise. It determines the signal-to-noise ratio at every point in the diffusion process and is a critical design choice that affects sample quality, training stability, and sampling efficiency.

2. - **$\beta_t$**: The noise variance added at a single forward step $t$ (small values, typically $10^{-4}$ to $0.02$).
   - **$\alpha_t = 1 - \beta_t$**: The fraction of signal retained at step $t$.
   - **$\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$**: The **cumulative signal retention** — the fraction of the original signal that survives after all $t$ steps. It decreases monotonically from $\approx 1$ (at $t=0$) to $\approx 0$ (at $t=T$).

3. $$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
   Equivalently: $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\, (1-\bar{\alpha}_t)\mathbf{I})$. This closed-form avoids running $t$ sequential noising steps.

4. The cosine schedule (Nichol & Dhariwal, 2021) was introduced because the **linear schedule destroys the signal too quickly** in the middle of the process. With a linear schedule, $\bar{\alpha}_t$ drops rapidly, meaning the model wastes many timesteps in a regime that is essentially pure noise. The cosine schedule provides a **more gradual, symmetric transition**, ensuring meaningful signal content persists longer and that the model gets useful training signal across a wider range of timesteps.

5. EDM **works directly with the noise standard deviation $\sigma(t)$** rather than defining a sequence of $\beta_t$ values. Instead of specifying how much noise to add at each step, EDM treats $\sigma$ as the fundamental quantity. For sampling, EDM spaces timesteps using $\sigma_i = \left(\sigma_{\max}^{1/\rho} + \frac{i}{N-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho})\right)^\rho$ with $\rho=7$, which is explicitly designed to allocate more steps where they matter most. This is cleaner and more flexible than the $\beta_t$-based formulation.

6. **Fine details are generated at low noise levels.** At high noise, only coarse, large-scale structure changes between steps — these transitions are smooth and easy. At low noise, the model must resolve sharp edges, textures, and subtle features, which requires more precise, smaller steps. Concentrating sampling steps near low noise (as EDM does with $\rho=7$) gives the solver more resolution where the denoising trajectory changes most rapidly, improving final sample quality.

7. | Aspect | Discrete (DDPM) | Continuous (Score SDE / EDM) |
   |--------|----------------|------------------------------|
   | **Time variable** | $t \in \{0, 1, \ldots, T\}$ (integer index) | $t \in [0, T]$ or $\sigma \in [\sigma_{\min}, \sigma_{\max}]$ (real-valued) |
   | **Schedule** | A fixed sequence $\beta_1, \ldots, \beta_T$ | A continuous function $\beta(t)$ or $\sigma(t)$ |
   | **Flexibility** | Tied to a specific $T$ (e.g., 1000) | Can choose any number of sampling steps from the continuous curve |
   | **Theory** | Markov chain with fixed transitions | Stochastic/ordinary differential equations |
   
   The continuous formulation is more general — discrete schedules can be seen as discretizations of a continuous schedule.

---
*Previous: [20-denoising.md](20-denoising.md) · Next: [22-forward-process.md](22-forward-process.md)*
