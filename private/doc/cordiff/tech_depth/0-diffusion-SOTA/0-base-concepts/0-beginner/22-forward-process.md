# Forward Process (Diffusion Process)

## What is it?

The **forward process** (also called the **diffusion process** or **noising process**) is the procedure that **gradually destroys data by adding noise**. It takes a clean data sample $\mathbf{x}_0$ and produces a sequence of increasingly noisy versions $\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T$, ending at (approximately) pure Gaussian noise.

```
x₀ ──→ x₁ ──→ x₂ ──→ ··· ──→ x_T
(clean)   (a little noisy)        (pure noise)
```

The forward process is **fixed** — it has no learnable parameters. It is entirely defined by the noise schedule (see [21-timestep-and-noise-schedule.md](21-timestep-and-noise-schedule.md)).

## Why Do We Need the Forward Process?

The forward process serves two purposes:

1. **Creates training data**: During training, we need (noisy input, target) pairs. The forward process generates the noisy inputs at any noise level.
2. **Defines the generative task**: The reverse process (generation) undoes the forward process. To know what to undo, we must first define what was done.

## The Step-by-Step Definition (DDPM)

Each step adds a small amount of Gaussian noise, controlled by $\beta_t$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\left(\mathbf{x}_t;\; \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1},\; \beta_t\,\mathbf{I}\right)$$

Breaking this down:
- The mean is $\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}$ — the previous image is **slightly shrunk** (scaled toward zero)
- The variance is $\beta_t\,\mathbf{I}$ — a small amount of noise is **added**

Using the reparameterization trick (see [16-reparameterization-trick.md](16-reparameterization-trick.md)):

$$\mathbf{x}_t = \sqrt{1 - \beta_t}\,\mathbf{x}_{t-1} + \sqrt{\beta_t}\,\boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

This is a Markov chain (see [12-markov-chain.md](12-markov-chain.md)): each step depends only on the previous step.

## The Closed-Form Shortcut

We don't need to simulate all $t$ steps sequentially! Thanks to the Gaussian properties (see [1-gaussian-distribution.md](1-gaussian-distribution.md)), we can jump directly from $\mathbf{x}_0$ to any $\mathbf{x}_t$:

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\left(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0,\; (1 - \bar{\alpha}_t)\,\mathbf{I}\right)$$

$$\boxed{\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

Where $\bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$ is the cumulative signal retention.

> 🔑 This closed-form formula is **essential for efficient training**. At each training step, we pick a random $t$, sample $\boldsymbol{\epsilon}$, compute $\mathbf{x}_t$ in one shot, and train the network. No need to simulate all previous steps.

### Derivation Sketch (Why It Works)

Starting from $\mathbf{x}_0$, apply two successive steps:

$$\mathbf{x}_1 = \sqrt{\alpha_1}\,\mathbf{x}_0 + \sqrt{1-\alpha_1}\,\boldsymbol{\epsilon}_1$$
$$\mathbf{x}_2 = \sqrt{\alpha_2}\,\mathbf{x}_1 + \sqrt{1-\alpha_2}\,\boldsymbol{\epsilon}_2$$

Substituting $\mathbf{x}_1$ into $\mathbf{x}_2$:

$$\mathbf{x}_2 = \sqrt{\alpha_2}\left(\sqrt{\alpha_1}\,\mathbf{x}_0 + \sqrt{1-\alpha_1}\,\boldsymbol{\epsilon}_1\right) + \sqrt{1-\alpha_2}\,\boldsymbol{\epsilon}_2$$

$$= \sqrt{\alpha_1 \alpha_2}\,\mathbf{x}_0 + \underbrace{\sqrt{\alpha_2(1-\alpha_1)}\,\boldsymbol{\epsilon}_1 + \sqrt{1-\alpha_2}\,\boldsymbol{\epsilon}_2}_{\text{sum of two independent Gaussians}}$$

By the variance addition property (see [3-mean-variance-std.md](3-mean-variance-std.md)), the noise term has variance:
$$\alpha_2(1-\alpha_1) + (1-\alpha_2) = 1 - \alpha_1\alpha_2 = 1 - \bar{\alpha}_2$$

So: $\mathbf{x}_2 = \sqrt{\bar{\alpha}_2}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_2}\,\boldsymbol{\epsilon}$. This generalizes to any $t$ by induction.

## What Happens at Each End?

### At $t = 0$: Clean Data
$$\bar{\alpha}_0 = 1, \quad \mathbf{x}_0 = 1 \cdot \mathbf{x}_0 + 0 \cdot \boldsymbol{\epsilon} = \mathbf{x}_0$$

### At $t = T$: Pure Noise
$$\bar{\alpha}_T \approx 0, \quad \mathbf{x}_T \approx 0 \cdot \mathbf{x}_0 + 1 \cdot \boldsymbol{\epsilon} = \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The schedule is designed so that by the final step, the original data is essentially gone.

## The Joint Forward Distribution

The entire forward chain is:

$$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$

This is a product of Gaussian transitions — a Markov chain.

## Alternative Formulations

### VE (Variance Exploding) — Score-Based Models
$$\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$$

- No scaling of the data — just add noise directly
- Variance grows as $\sigma_t$ increases (hence "exploding")

### VP (Variance Preserving) — DDPM Family
$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$$

- Data is scaled down and noise is scaled up
- Total variance stays bounded at $\approx 1$ (hence "preserving")

### EDM Formulation
$$\mathbf{x}_t = \mathbf{x}_0 + \sigma(t)\,\boldsymbol{\epsilon}$$

- Equivalent to VE but with a carefully chosen $\sigma(t)$

> 💡 All three describe the **same fundamental idea** (gradually adding noise) with different notation and slightly different mathematical properties. The EDM paper shows how to translate between them.

## The Forward Process in Practice (Training)

During training, the forward process is used like this:

```python
# 1. Get a clean sample from the dataset
x_0 = dataset[i]

# 2. Sample a random timestep
t = torch.randint(1, T+1, (1,))

# 3. Sample noise
eps = torch.randn_like(x_0)

# 4. Create noisy sample (closed-form, one shot!)
alpha_bar_t = alpha_bar_schedule[t]
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

# 5. Predict noise with the network
eps_pred = model(x_t, t)

# 6. Compute loss
loss = MSE(eps, eps_pred)
```

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ | One forward step |
| $q(\mathbf{x}_t \mid \mathbf{x}_0)$ | Direct jump from clean to noisy (closed-form) |
| $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ | Full forward chain |
| $\sqrt{\bar{\alpha}_t}$ | Signal coefficient at step $t$ |
| $\sqrt{1-\bar{\alpha}_t}$ | Noise coefficient at step $t$ |

## Knowledge Check ✅

1. What does the forward process do, and is it learned or fixed?
2. Write the one-step forward transition $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$.
3. Write the closed-form $q(\mathbf{x}_t \mid \mathbf{x}_0)$ and explain why it's so useful.
4. Sketch the derivation for why the closed-form works (hint: Gaussian addition property).
5. What are the values of $\bar{\alpha}_t$ at $t=0$ and $t=T$? What does this mean physically?
6. What is the difference between VP and VE forward processes?
7. During training, do we ever need to run all $T$ steps sequentially? Why or why not?

---
*Previous: [21-timestep-and-noise-schedule.md](21-timestep-and-noise-schedule.md) · Next: [23-reverse-process.md](23-reverse-process.md)*
