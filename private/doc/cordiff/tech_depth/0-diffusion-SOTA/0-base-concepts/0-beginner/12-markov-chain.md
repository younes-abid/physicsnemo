# Markov Chain

## What is it?

A **Markov chain** is a sequence of random variables $X_0, X_1, X_2, \ldots$ where the distribution of each state depends **only on the immediately preceding state** — not on any earlier history.

This is called the **Markov property** (or "memorylessness"):

$$P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} \mid X_t)$$

In words: "The future depends only on the present, not the past."

## Intuitive Analogy

Imagine a board game where you roll a die to move:
- Where you land next depends only on **where you are now** and the die roll
- It doesn't matter **how** you got to your current position
- Your full history of moves is irrelevant — only your current square matters

That's a Markov chain.

## A Simple Example

### Weather as a Markov chain

Suppose weather is either Sunny (S) or Rainy (R), and:
- If today is Sunny: 80% chance tomorrow is Sunny, 20% chance Rainy
- If today is Rainy: 40% chance tomorrow is Sunny, 60% chance Rainy

```
        0.8
    S ──────→ S
    │         ↑
0.2 │         │ 0.4
    ↓         │
    R ──────→ R
        0.6
```

This is a Markov chain because tomorrow's weather depends only on today's weather, not yesterday's.

## The Transition Kernel

The rule that defines how we go from one state to the next is called the **transition kernel** (or **transition probability**):

$$T(x_{t+1} \mid x_t) = P(X_{t+1} = x_{t+1} \mid X_t = x_t)$$

For continuous state spaces (like images), this is a conditional density:
$$p(x_{t+1} \mid x_t)$$

## Why Markov Chains Matter for Diffusion

### The Forward Process is a Markov Chain

The DDPM forward process is defined as a Markov chain:

$$q(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_T \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$

Each noisy image $\mathbf{x}_t$ depends only on the previous image $\mathbf{x}_{t-1}$:

$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t \mathbf{I})$$

```
x₀ → x₁ → x₂ → ... → x_T
(clean)                 (pure noise)
```

Each arrow represents one step of adding a small amount of Gaussian noise.

### The Reverse Process is Also a Markov Chain

The learned reverse process is also defined as a Markov chain:

$$p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod_{t=1}^{T} p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$$

```
x_T → x_{T-1} → ... → x₁ → x₀
(pure noise)              (clean)
```

Each reverse step only needs the current noisy image — no memory of earlier states.

### Why is the Markov Property Important?

1. **Simplicity**: We only need to define **one-step transitions**, not the full joint distribution over all steps
2. **Factorization**: The joint probability factors into a product of simple conditional terms (the chain rule of the Markov chain)
3. **Tractability**: Each step involves a simple Gaussian transition — manageable math
4. **The "nice" closed form**: Even though the forward process is a chain of $T$ small steps, the Markov property plus Gaussian transitions allows us to compute $q(\mathbf{x}_t \mid \mathbf{x}_0)$ directly (skipping all intermediate steps)

## Beyond DDPM: Non-Markovian Processes

**DDIM** (Denoising Diffusion Implicit Models, Song et al. 2020) breaks the Markov assumption! The DDIM reverse process looks at both $\mathbf{x}_t$ and $\mathbf{x}_0$ (predicted), making it **non-Markovian**. This allows:
- Fewer sampling steps (skip steps)
- Deterministic sampling (same noise → same output)

This is an important innovation we'll cover later in the paper-specific files.

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ | One forward step of the Markov chain |
| $p_\theta(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$ | One reverse step of the Markov chain |
| $\prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ | Full forward chain (product of transitions) |

## Knowledge Check ✅

1. What is the Markov property? State it in one sentence.
2. In the diffusion forward process, does $\mathbf{x}_t$ depend on $\mathbf{x}_0$ directly, or only on $\mathbf{x}_{t-1}$?
3. Why does the Markov property make the math simpler?
4. Write the factored form of the joint forward distribution $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$.
5. Which paper introduces a non-Markovian reverse process, and why?

### Answers

1. **The future state depends only on the present state, not on any past history:** $P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0) = P(X_{t+1} \mid X_t)$. In other words, knowing where you are right now tells you everything you need to predict where you'll be next — how you got here is irrelevant.

2. **Both, depending on context.** By *definition*, the forward process is a Markov chain where each step $q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$ depends only on $\mathbf{x}_{t-1}$. However, because the transitions are Gaussian and the Markov chain is linear-Gaussian, we can derive a *closed-form marginal* $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$ that lets us jump directly from $\mathbf{x}_0$ to any $\mathbf{x}_t$ without computing intermediate steps. The Markov structure is in the *definition*; the direct dependence on $\mathbf{x}_0$ is a *derived shortcut* that's critical for efficient training.

3. **Because it lets us factorize the full joint distribution into a product of simple one-step transitions.** Without the Markov property, $q(\mathbf{x}_{1:T} \mid \mathbf{x}_0)$ would require specifying the full $T$-dimensional joint distribution — intractable for large $T$. With the Markov property, it factors as $\prod_{t=1}^T q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$, where each factor is a simple Gaussian. This also means we only need to define *one* transition rule that's reused at every step, and it enables the closed-form marginal $q(\mathbf{x}_t \mid \mathbf{x}_0)$ via cumulative products of the $\alpha_t$ coefficients.

4. $$q(\mathbf{x}_{1:T} \mid \mathbf{x}_0) = \prod_{t=1}^{T} q(\mathbf{x}_t \mid \mathbf{x}_{t-1})$$
   where each factor is $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t \mathbf{I})$. This is the chain rule of probability *simplified* by the Markov property — each term conditions only on the immediately preceding state, not all previous states.

5. **DDIM (Denoising Diffusion Implicit Models, Song et al. 2020)** introduces a non-Markovian reverse process. In DDIM, the reverse step uses both $\mathbf{x}_t$ and a prediction of $\mathbf{x}_0$ (derived from the noise prediction), making each reverse step depend on more than just the current state. The motivation is twofold:
   - **Fewer sampling steps**: because the process is non-Markovian, you can skip timesteps during generation (e.g., go from step 1000 → 900 → 800 instead of every single step), dramatically reducing inference cost.
   - **Deterministic sampling**: DDIM can set the stochasticity to zero, making generation fully deterministic — the same initial noise always produces the same output. This enables meaningful interpolation in the latent space and reproducible generation.

---
*Previous: [11-signal-to-noise-ratio.md](11-signal-to-noise-ratio.md) · Next: [13-latent-variable.md](13-latent-variable.md)*
