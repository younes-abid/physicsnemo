# Signal-to-Noise Ratio (SNR)

## What is it?

The **Signal-to-Noise Ratio (SNR)** quantifies how much useful information (signal) is present relative to the amount of random corruption (noise). It's a single number that tells you: "Is this mostly signal or mostly noise?"

$$\text{SNR} = \frac{\text{Power of signal}}{\text{Power of noise}}$$

In diffusion models, we typically define it as the ratio of variances:

$$\text{SNR}(t) = \frac{\alpha_t}{1 - \alpha_t}$$

where $\alpha_t$ (or $\bar{\alpha}_t$ in DDPM notation) controls how much original signal remains at timestep $t$.

## Intuitive Understanding

| SNR Value | What it means | In diffusion |
|-----------|--------------|--------------|
| $\text{SNR} \gg 1$ (high) | Mostly signal, very little noise | Early steps: image nearly clean |
| $\text{SNR} \approx 1$ | Equal parts signal and noise | Middle steps: partially corrupted |
| $\text{SNR} \ll 1$ (low) | Mostly noise, very little signal | Late steps: image nearly destroyed |
| $\text{SNR} = 0$ | Pure noise, no signal at all | Final step: $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ |

## Log-SNR

Papers often use **log-SNR** because it provides a more balanced view:

$$\log \text{SNR}(t) = \log \frac{\alpha_t}{1 - \alpha_t}$$

- $\log \text{SNR} > 0$ → more signal than noise
- $\log \text{SNR} = 0$ → equal signal and noise
- $\log \text{SNR} < 0$ → more noise than signal

Log-SNR ranges from $+\infty$ (clean) to $-\infty$ (pure noise), which is a more natural and symmetric scale.

## SNR in the Diffusion Forward Process

Recall the forward process (see [22-forward-process.md](22-forward-process.md)):

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t} \, \mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}$$

The signal component has variance $\bar{\alpha}_t \cdot \text{Var}(\mathbf{x}_0)$ and the noise component has variance $1 - \bar{\alpha}_t$.

Assuming data is normalized so $\text{Var}(\mathbf{x}_0) = 1$:

$$\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}$$

As $t$ goes from $0$ to $T$:
- $\bar{\alpha}_t$ decreases from $\approx 1$ to $\approx 0$
- SNR decreases from $\approx \infty$ to $\approx 0$
- The image transitions from clean to pure noise

## Why SNR Matters for Diffusion Models

### 1. Understanding the Noise Schedule
The noise schedule (see [21-timestep-and-noise-schedule.md](21-timestep-and-noise-schedule.md)) defines how $\bar{\alpha}_t$ changes over time. Different schedules produce different SNR curves. The SNR perspective makes it easier to compare schedules across different papers and formulations.

### 2. Loss Weighting
The importance of each timestep in the training loss can be understood through SNR. Some papers (notably EDM by Karras et al., 2022) argue that the loss should be weighted based on the SNR to give balanced attention to all noise levels:

$$L = \mathbb{E}_{t}\left[w(\text{SNR}(t)) \cdot \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

The weighting function $w(\text{SNR}(t))$ determines how much each noise level contributes to the total loss.

### 3. Unifying Different Formulations
The EDM framework shows that many seemingly different diffusion formulations (DDPM, SMLD, etc.) can be understood as the **same process** with different SNR schedules. SNR becomes the unifying language.

## Different Notations Across Papers

Different papers parameterize the same concept differently:

| Paper | Signal coefficient | Noise coefficient | SNR expression |
|-------|-------------------|-------------------|----------------|
| DDPM | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | $\bar{\alpha}_t / (1-\bar{\alpha}_t)$ |
| Score SDE (VP) | $e^{-\frac{1}{2}\int_0^t \beta(s)ds}$ | $\sqrt{1 - e^{-\int_0^t \beta(s)ds}}$ | Similar ratio |
| EDM | $1$ (data not scaled) | $\sigma(t)$ | $1/\sigma(t)^2$ |

> 💡 Despite different notation, they all describe the same idea: the ratio of signal power to noise power at each point in the process.

## Knowledge Check ✅

1. What does SNR measure in plain English?
2. If $\text{SNR}(t) = 100$, is the image mostly clean or mostly noisy?
3. What is the SNR at the very beginning ($t=0$) and the very end ($t=T$) of the forward process?
4. Why do papers sometimes use log-SNR instead of SNR?
5. How does the EDM framework use SNR to unify different diffusion formulations?
6. In the formula $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$, identify the signal and noise components.

### Answers

1. **SNR measures how much useful information (signal) remains relative to the amount of random corruption (noise).** A high SNR means the data is mostly intact with little noise; a low SNR means noise dominates and the original data is hard to discern. It's a single number summarizing "how clean vs. how corrupted" the data is at a given point in the diffusion process.

2. **Mostly clean.** $\text{SNR} = 100$ means there is 100× more signal power than noise power. The original image is almost perfectly preserved with only a tiny amount of noise — you'd barely see any corruption. This corresponds to very early timesteps in the forward process.

3. - **At $t = 0$**: $\bar{\alpha}_0 \approx 1$, so $\text{SNR}(0) = \frac{\bar{\alpha}_0}{1 - \bar{\alpha}_0} \approx \frac{1}{0} \to +\infty$. The data is clean — essentially infinite signal relative to noise.
   - **At $t = T$**: $\bar{\alpha}_T \approx 0$, so $\text{SNR}(T) = \frac{\bar{\alpha}_T}{1 - \bar{\alpha}_T} \approx \frac{0}{1} = 0$. The data is pure noise — no signal remains.
   
   The entire forward process is a monotonic journey from $\text{SNR} = \infty$ to $\text{SNR} = 0$.

4. **Because log-SNR provides a more balanced and symmetric scale.** Raw SNR ranges from $+\infty$ to $0$, which is lopsided — high-signal regimes get huge values while high-noise regimes are compressed near zero. Log-SNR ranges from $+\infty$ to $-\infty$, with $\log \text{SNR} = 0$ marking the exact midpoint where signal and noise are equal. This makes it easier to visualize noise schedules, compare them across papers, and design loss weightings that treat all noise levels fairly.

5. **EDM shows that DDPM, SMLD/Score SDE, and other formulations are all instances of the same generative process — they just use different SNR schedules.** By reparameterizing everything in terms of the noise level $\sigma(t)$ (with $\text{SNR} = 1/\sigma^2$ in EDM's convention), Karras et al. demonstrate that the choice of noise schedule, loss weighting, and network preconditioning can be decoupled and studied independently. SNR becomes the common language: instead of comparing formulations through their different $\alpha_t$, $\beta_t$, or $\sigma_t$ parameterizations, you just compare their SNR curves over time.

6. - **Signal component**: $\sqrt{\bar{\alpha}_t} \, \mathbf{x}_0$ — this is the original clean data, scaled down by $\sqrt{\bar{\alpha}_t}$. Its power (variance) is $\bar{\alpha}_t \cdot \text{Var}(\mathbf{x}_0)$.
   - **Noise component**: $\sqrt{1 - \bar{\alpha}_t} \, \boldsymbol{\epsilon}$ — this is standard Gaussian noise, scaled by $\sqrt{1 - \bar{\alpha}_t}$. Its power (variance) is $1 - \bar{\alpha}_t$.
   
   As $t$ increases, $\bar{\alpha}_t$ shrinks toward 0: the signal coefficient decreases (signal fades) while the noise coefficient increases (noise grows). The SNR is the ratio of their variances: $\text{SNR}(t) = \bar{\alpha}_t / (1 - \bar{\alpha}_t)$.

---
*Previous: [10-noise.md](10-noise.md) · Next: [12-markov-chain.md](12-markov-chain.md)*
