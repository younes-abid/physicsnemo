# Exponential Moving Average (EMA) — Background

## 1. The Problem: Why Raw Training Weights Are Noisy

During neural network training, the optimizer (e.g., Adam, SGD) updates model
weights at every step based on **mini-batch gradients**. These gradients are
inherently noisy because:

1. **Mini-batch sampling:** Each gradient is computed on a small random subset of
   the data, not the full dataset. Different mini-batches produce different
   gradient directions.

2. **Stochastic noise:** The loss landscape is complex. Even with the same data,
   the gradient signal is mixed with noise from the optimization dynamics.

3. **Learning rate:** Larger learning rates amplify the noise in weight updates.

The result: model weights **oscillate** around a good solution rather than
converging smoothly to it. At any given training step, the current weights are a
noisy estimate of the "true" optimal weights.

```
                    ┌─ Raw weights oscillate
    Loss            │
     │    ╭─╮ ╭─╮  │  ╭─╮
     │   ╱  ╰╮│ ╰──╯╭╯  ╰─╮  ╭──
     │  ╱    ╰╯      ╰     ╰──╯
     │ ╱
     │╱                ← True optimum (smooth)
     └────────────────────────────── Training steps
```

## 2. The Solution: Exponential Moving Average

**EMA** maintains a separate copy of the model weights that is a smoothed
(time-averaged) version of the training weights.

### The Update Rule

At each training step `t`, after the optimizer updates the training weights `θ_t`:

```
θ_ema(t) = β × θ_ema(t-1) + (1 - β) × θ_t
```

Where:
- `θ_t` = current training weights (after optimizer step)
- `θ_ema(t)` = EMA weights at step t
- `β` = decay rate (typically 0.999 to 0.9999)
- `(1 - β)` = how much of the new weights to incorporate

### Intuition

Think of it like a **low-pass filter** on the weight trajectory:
- `β = 0.999` means "keep 99.9% of the old average, add 0.1% of the new weights"
- This smooths out the oscillations while tracking the long-term trend
- The EMA weights represent a "consensus" of recent training states

### Effective Window

The EMA with decay `β` is approximately equivalent to averaging over the last
`1/(1-β)` steps:

| Decay (β) | Effective window | Use case |
|-----------|-----------------|----------|
| 0.99      | ~100 steps      | Fast adaptation, more noise |
| 0.999     | ~1,000 steps    | Good balance |
| 0.9999    | ~10,000 steps   | Very smooth, slow to adapt |
| 0.99999   | ~100,000 steps  | Extremely smooth |

## 3. Why EMA Works Better for Inference

### Training weights vs EMA weights

```
Training weights at step 1000: θ_1000  (noisy, one specific point)
EMA weights at step 1000:      average of θ_900 to θ_1000 (smoothed)
```

The EMA weights are better because:

1. **Noise reduction:** Averaging reduces the variance of the weight estimate by
   ~1/√N where N is the effective window size.

2. **Implicit ensemble:** The EMA weights act like an ensemble of models from
   recent training history, which generalizes better.

3. **Flatter minima:** EMA weights tend to sit in flatter regions of the loss
   landscape, which correlate with better generalization.

### Empirical evidence

In virtually every diffusion model paper:
- **Training** uses the raw optimizer-updated weights
- **Inference/sampling** uses the EMA weights
- The quality difference is substantial (often 10-30% better FID scores)

## 4. Implementation Pattern

The standard implementation in PyTorch:

```python
import copy

# 1. Create EMA model as a deep copy of the training model
ema_model = copy.deepcopy(model)
ema_model.eval()
ema_model.requires_grad_(False)  # EMA model is never trained directly

# 2. After each optimizer.step(), update EMA weights
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for p_ema, p_train in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p_train.data, alpha=1.0 - decay)

# 3. Training loop
for batch in dataloader:
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    update_ema(ema_model, model, decay=0.9999)  # ← This is the only addition

# 4. Use EMA model for inference
samples = generate(ema_model, ...)  # NOT model
```

### Key points:
- The EMA model is **never** trained with gradients
- It's updated via the simple weighted average formula
- It **doubles GPU memory** (two copies of all parameters)
- The training model is used for gradient computation
- The EMA model is used for inference/evaluation

## 5. EMA in Diffusion Models — Why It's Critical

Diffusion models are **especially** sensitive to weight noise because:

1. **Multi-step sampling:** During inference, the model is called 20-1000 times
   in sequence. Small weight noise compounds across steps.

2. **Score estimation:** The model predicts a denoising direction (score). Noisy
   weights → noisy scores → noisy samples.

3. **High-frequency details:** Fine details in generated images are the first to
   degrade from weight noise.

### Standard practice in all major diffusion implementations:

| Implementation | Uses EMA? | Decay |
|---------------|-----------|-------|
| EDM (Karras et al., 2022) | ✅ Yes | 0.9999 |
| DDPM (Ho et al., 2020) | ✅ Yes | 0.9999 |
| Stable Diffusion | ✅ Yes | 0.9999 |
| CorrDiff paper (Mardani et al., 2023) | ✅ Yes | 0.9999 |
| Our current train.py | ❌ **NO** | — |

## 6. Common Pitfalls

### Pitfall 1: EMA warmup
At the start of training, the EMA weights are just a copy of the (random) initial
weights. As training progresses, the EMA "warms up" and becomes meaningful. Some
implementations skip the first N steps or use a lower decay initially.

### Pitfall 2: DDP wrapper
When using `DistributedDataParallel`, the model is wrapped. The EMA should track
the **unwrapped** model parameters (accessed via `model.module`).

### Pitfall 3: Forgetting to save/load EMA
The EMA model state must be checkpointed alongside the training model. Otherwise,
resuming training loses the accumulated EMA history.

### Pitfall 4: Using EMA for training
The EMA model should **never** be used in the training forward pass. It's only for
inference/evaluation.

---

## References

1. Polyak, B.T. and Juditsky, A.B., 1992. "Acceleration of stochastic
   approximation by averaging." SIAM Journal on Control and Optimization.
   — *Original theoretical foundation for weight averaging*

2. Tarvainen, A. and Valpola, H., 2017. "Mean teachers are better role models."
   NeurIPS. — *EMA in semi-supervised learning (Mean Teacher)*

3. Karras, T. et al., 2022. "Elucidating the Design Space of Diffusion-Based
   Generative Models." NeurIPS. — *EDM framework used in CorrDiff, uses EMA*

4. Ho, J. et al., 2020. "Denoising Diffusion Probabilistic Models." NeurIPS.
   — *DDPM, uses EMA with decay 0.9999*
