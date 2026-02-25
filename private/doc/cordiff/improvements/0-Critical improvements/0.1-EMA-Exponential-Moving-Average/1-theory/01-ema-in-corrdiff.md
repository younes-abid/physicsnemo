# EMA in CorrDiff — Where and How to Intervene

## 1. Current CorrDiff Training Flow (What Exists)

The current training script (`examples/weather/corrdiff/train.py`) follows this flow:

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT FLOW (NO EMA)                    │
│                                                             │
│  1. Create model (EDMPrecondSuperResolution)                │
│  2. Wrap with DDP                                           │
│  3. Create optimizer (Adam)                                 │
│  4. Load checkpoint (if resuming)                           │
│                                                             │
│  Training loop:                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  a. Zero gradients                                   │   │
│  │  b. Forward pass → compute loss (ResidualLoss)       │   │
│  │  c. Backward pass → compute gradients                │   │
│  │  d. Clip gradients                                   │   │
│  │  e. optimizer.step()  ← weights updated              │   │
│  │  f. (nothing else — NO EMA update)                   │   │
│  │  g. Save checkpoint periodically                     │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  The SAME noisy weights are used for:                       │
│  - Training (forward/backward)                              │
│  - Validation                                               │
│  - Checkpoint saving                                        │
│  - Inference (generate.py loads these checkpoints)          │
└─────────────────────────────────────────────────────────────┘
```

### The key problem

When you run `generate.py` to produce weather predictions, it loads the
checkpoint saved by `train.py`. These are the **raw training weights** — the
noisy, oscillating weights from the last optimizer step. The diffusion sampling
process (20-100 denoising steps) amplifies this noise.

## 2. Modified Flow (With EMA)

```
┌─────────────────────────────────────────────────────────────┐
│                   NEW FLOW (WITH EMA)                       │
│                                                             │
│  1. Create model (EDMPrecondSuperResolution)                │
│  2. Wrap with DDP                                           │
│  3. Create EMA copy of model  ← NEW                        │
│  4. Create optimizer (Adam)                                 │
│  5. Load checkpoint (if resuming)                           │
│  5b. Load EMA checkpoint (if resuming)  ← NEW              │
│                                                             │
│  Training loop:                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  a. Zero gradients                                   │   │
│  │  b. Forward pass → compute loss (ResidualLoss)       │   │
│  │  c. Backward pass → compute gradients                │   │
│  │  d. Clip gradients                                   │   │
│  │  e. optimizer.step()  ← training weights updated     │   │
│  │  f. update_ema(ema_model, model)  ← NEW              │   │
│  │  g. Save checkpoint (training + EMA) periodically    │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  Training weights → used for forward/backward               │
│  EMA weights → used for validation & saved for inference    │
└─────────────────────────────────────────────────────────────┘
```

## 3. Exact Intervention Points in train.py

Below are the **exact locations** in `train.py` where code changes are needed,
mapped to the function names in our refactored training script.

### 3.1 — Create the EMA model (in `main()`, after model creation)

**Location:** After `model = setup_distributed_data_parallel(model, dist)` and
after loading the model checkpoint.

**What to add:**
- Deep copy the model to create `ema_model`
- Set EMA model to eval mode, no gradients
- Load EMA checkpoint if resuming

**Why here:** The EMA must be created AFTER the model is fully initialized,
wrapped with DDP, and loaded from checkpoint (so the EMA starts from the correct
weights when resuming).

### 3.2 — Update EMA after optimizer step (in `training_iteration_block()`)

**Location:** Inside `training_iteration_block()`, immediately after
`optimizer.step()`.

**What to add:** A single call to `update_ema(ema_model, model, decay)`.

**Why here:** The EMA update must happen after every optimizer step, before
anything else. This is the core of the EMA mechanism.

### 3.3 — Save EMA in checkpoints (in `checkpoint_block()`)

**Location:** Inside `checkpoint_block()`, alongside the existing
`save_checkpoint()` call.

**What to add:** Save the EMA model state dict to a separate file.

**Why here:** The EMA weights must be persisted so they survive training
restarts. Without this, resuming training resets the EMA to the training weights,
losing all accumulated smoothing.

### 3.4 — Log EMA vs training weight divergence (optional, in logging)

**Location:** In `log_training_metrics()` or `log_progress_block()`.

**What to add:** Log the L2 distance between EMA and training weights. This is
useful for monitoring — if the distance is too large, the decay might be too high.

## 4. Interaction with Existing Components

### 4.1 — DistributedDataParallel (DDP)

CorrDiff uses DDP when `dist.world_size > 1`. The training model is wrapped:
```python
model = DistributedDataParallel(model, ...)
```

The EMA model should track the **unwrapped** parameters. When accessing
parameters for the EMA update:
```python
# Correct: unwrap DDP to get actual parameters
train_params = model.module.parameters() if hasattr(model, 'module') else model.parameters()
ema_params = ema_model.parameters()
```

The EMA model itself does NOT need DDP wrapping since it's never used for
training (no gradient synchronization needed).

### 4.2 — torch.compile

If `use_torch_compile` is enabled, the model is compiled:
```python
model = torch.compile(model)
```

The EMA model should be created BEFORE compilation. If you compile the EMA model,
you may need to unwrap `_orig_mod` to access parameters. The simpler approach:
don't compile the EMA model (it's never used in the training loop).

### 4.3 — Gradient checkpointing

The UNet uses gradient checkpointing (`checkpoint_level > 0`) to save memory.
This only affects the training forward pass — the EMA model is not affected since
it's never used for forward/backward during training.

### 4.4 — ResidualLoss and regression_net

The loss function `ResidualLoss` uses a frozen `regression_net` for computing
residuals. The regression network is completely separate from the diffusion model
and does NOT need EMA — it's already frozen and loaded from a pre-trained
checkpoint.

### 4.5 — Memory impact

The EMA model doubles the parameter memory:

| Component | Memory |
|-----------|--------|
| Training model parameters | ~X GB |
| EMA model parameters | ~X GB (same size) |
| Optimizer states (Adam) | ~2X GB (momentum + variance) |
| **Total with EMA** | **~4X GB** (vs 3X without EMA) |

For a typical CorrDiff model, this is ~200-400 MB additional GPU memory — well
within the capacity of modern GPUs (A100 80GB, H100 80GB).

## 5. Configuration

The EMA is controlled by a single new config parameter added to the training
config YAML:

```yaml
training:
  hp:
    # ... existing parameters ...
    ema_decay: 0.9999
    # EMA decay rate. Set to 0 or null to disable EMA.
    # Recommended: 0.9999 (standard for diffusion models)
```

## 6. Validation with EMA

Optionally, the `validation_block()` could use the EMA model instead of the
training model for validation metrics. This gives a more accurate picture of
inference quality during training. This is a minor change (pass `ema_model`
instead of `model` to the loss function during validation).

However, for the v-1 implementation we keep it simple: the EMA model is only
saved in checkpoints. Validation still uses the training model. This can be
enhanced in a future version.

## 7. Files Modified

| File | Change |
|------|--------|
| `examples/weather/corrdiff/train.py` | Add EMA creation, update, save/load |

That's it — **one file only**. No changes to the model architecture, loss
function, dataset, or any PhysicsNeMo library code.
