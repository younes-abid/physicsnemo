# Part 7: What Is an MLP?

## MLP = Multi-Layer Perceptron

An MLP is the simplest type of neural network. It's a stack of **linear layers**
with **nonlinear activations** between them. That's it — no convolutions, no
attention, no recurrence. Just matrix multiplications and element-wise functions.

## The Building Blocks

### Linear Layer

A linear layer computes:

```
output = input × W^T + b
```

Where:
- `input`: the data coming in (a vector)
- `W`: a matrix of **weights** (learned during training)
- `b`: a vector of **biases** (also learned)
- `output`: the transformed data

**Example:** Linear(1 → 64) takes a 1-dimensional input and produces a
64-dimensional output:

```
input:  [x]           (1 number)
W:      [[w₁],        (64 rows × 1 column = 64 weights)
         [w₂],
         ...
         [w₆₄]]
b:      [b₁, b₂, ..., b₆₄]   (64 biases)

output: [x×w₁+b₁, x×w₂+b₂, ..., x×w₆₄+b₆₄]   (64 numbers)
```

Each output dimension is a **different linear function** of the input.
With 64 outputs, we get 64 different "views" of the single input number.

### Activation Function (SiLU)

A nonlinear activation is applied element-wise after each linear layer.
CorrDiff uses **SiLU** (also called "Swish"):

```
SiLU(x) = x × sigmoid(x) = x × (1 / (1 + e^(-x)))
```

Plotted:
```
output
  │         ╱
  │        ╱
  │       ╱
  │      ╱
  │    ╱
──┼──╱─────── input
  │╱
  │
  │
```

Key properties:
- **Smooth:** No sharp corners (good for gradient flow)
- **Non-monotonic:** Has a slight dip for negative inputs (richer than ReLU)
- **Passes through zero:** SiLU(0) = 0
- **Approximately linear for large positive x:** SiLU(x) ≈ x when x >> 0
- **Suppresses large negative x:** SiLU(x) ≈ 0 when x << 0

**Why do we need nonlinearity?** Without it, stacking two linear layers is
equivalent to a single linear layer (matrix multiplication is associative):

```
Linear₂(Linear₁(x)) = (W₂ × W₁) × x + (W₂ × b₁ + b₂) = W_combined × x + b_combined
```

The activation function breaks this equivalence, allowing the network to learn
**non-linear** relationships.

## Our Specific MLP: 1 → 64 → 1

The noise conditioning MLP has three layers:

```
                        ┌─────────────────────────────────┐
Input: log(σ)           │          THE MLP                │
(1 number)              │                                 │
    │                   │  Layer 0: Linear(1 → 64)        │
    ▼                   │  ┌──────────────────────────┐   │
┌───────┐               │  │ 64 weights + 64 biases   │   │
│ log(σ)│──────────────►│  │ = 128 parameters         │   │
└───────┘               │  └────────────┬─────────────┘   │
                        │               │                  │
                        │               ▼                  │
                        │  Layer 1: SiLU (no parameters)  │
                        │  ┌──────────────────────────┐   │
                        │  │ x × sigmoid(x)            │   │
                        │  │ applied to each of 64 dims│   │
                        │  └────────────┬─────────────┘   │
                        │               │                  │
                        │               ▼                  │
                        │  Layer 2: Linear(64 → 1)        │
                        │  ┌──────────────────────────┐   │
                        │  │ 64 weights + 1 bias       │   │
                        │  │ = 65 parameters           │   │
                        │  └────────────┬─────────────┘   │
                        │               │                  │
                        └───────────────┼─────────────────┘
                                        │
                                        ▼
                                   Output: δ(σ)
                                   (1 number)

Total: 128 + 65 = 193 parameters
```

### What Each Layer Does

**Layer 0 (Linear 1→64):** Takes the single number `log(σ)` and creates 64
different linear combinations of it. Think of it as creating 64 "features":

```
feature_1 = w₁ × log(σ) + b₁    (maybe this one responds to low noise)
feature_2 = w₂ × log(σ) + b₂    (maybe this one responds to high noise)
...
feature_64 = w₆₄ × log(σ) + b₆₄
```

**Layer 1 (SiLU):** Applies the nonlinearity to each feature independently.
This allows each feature to "activate" in a specific range of log(σ) and be
"off" in other ranges:

```
feature_1 after SiLU: active for log(σ) > -2  (σ > 0.14)
feature_2 after SiLU: active for log(σ) > 0   (σ > 1.0)
feature_17 after SiLU: active for log(σ) > 3  (σ > 20)
... etc
```

**Layer 2 (Linear 64→1):** Combines all 64 activated features into a single
output number — the correction δ(σ):

```
δ(σ) = Σᵢ wᵢ × SiLU(vᵢ × log(σ) + cᵢ) + d
```

This is a **weighted sum of 64 SiLU bumps**, each centered at a different
location along the log(σ) axis. Together they can approximate any smooth
function of log(σ).

## What the MLP Can Learn

With 64 hidden units and SiLU activation, the MLP can approximate essentially
any smooth function `f: ℝ → ℝ`. Here are examples of what δ(σ) might look like:

### Example 1: Linear Correction (rescaling)
```
δ(σ) ≈ 0.05 × log(σ)
Effect: c_noise = log(σ)/4 + 0.05×log(σ) = log(σ) × (1/4 + 0.05) = log(σ)/3.64
Meaning: The optimal scaling is /3.64, not /4
```

### Example 2: Constant Shift
```
δ(σ) ≈ 0.3
Effect: c_noise = log(σ)/4 + 0.3
Meaning: The sinusoidal embedding works better with a shifted center
```

### Example 3: Bump at Low Noise
```
δ(σ) ≈ 0 for most σ, but δ ≈ 0.5 for σ ∈ [0.01, 0.1]
Effect: Low-noise labels are pushed further apart
Meaning: Weather data needs finer discrimination at low noise levels
```

### Example 4: Near-Zero (identity)
```
δ(σ) ≈ 0 everywhere
Effect: c_noise = log(σ)/4 (unchanged)
Meaning: EDM's formula is already optimal for this data
```

## Why 64 Hidden Units?

This is a design choice. Here's the tradeoff:

| Hidden size | Parameters | Expressivity | Risk of overfitting |
|-------------|-----------|--------------|-------------------|
| 8 | 25 | Can learn simple curves | Very low |
| **64** | **193** | **Can learn complex shapes** | **Very low** |
| 256 | 769 | Can learn anything | Still very low |
| 1024 | 3,073 | Overkill | Still very low |

Even 1024 hidden units would add only 3,073 parameters to a 100M+ parameter
model. The risk of overfitting is essentially zero because:

1. The MLP outputs a single scalar (1D output = very constrained)
2. The input is 1D (log(σ)), so the function being learned is 1D → 1D
3. 193 parameters for a 1D → 1D function is already very expressive

We chose 64 as a reasonable middle ground: expressive enough to learn any
smooth correction, small enough to be truly negligible.

## The Zero Initialization Trick

The last layer is initialized to **all zeros**:

```python
nn.init.zeros_(self.noise_embed[2].weight)   # W₂ = 0
nn.init.zeros_(self.noise_embed[2].bias)     # b₂ = 0
```

This means at the start of training:

```
δ(σ) = W₂ × SiLU(W₁ × log(σ) + b₁) + b₂
     = 0  × SiLU(anything)           + 0
     = 0

∴ c_noise = log(σ)/4 + 0 = log(σ)/4    ← exact EDM baseline!
```

No matter what the first layer's random weights are, the output is zero because
the second layer's weights are zero. Training then gradually "opens up" these
weights as the optimizer finds useful corrections.

This is a common trick in deep learning called **"zero initialization for
residual branches"** — used in GPT-2, ViT, and many other architectures.

## Summary

| Concept | What it means |
|---------|---------------|
| MLP | Stack of linear layers with nonlinear activations |
| Linear layer | Matrix multiplication: y = Wx + b |
| SiLU activation | x × sigmoid(x); allows non-linear learning |
| Our MLP | 1 → 64 → 1; takes log(σ), outputs correction δ(σ) |
| 193 parameters | 0.0001% of the full model — truly negligible |
| 64 hidden units | Can approximate any smooth 1D function |
| Zero init | Last layer starts at 0 → MLP outputs 0 → starts at baseline |

---

**Previous:** [Part 6: The Full Pipeline and What We Want to Change](./06-the-full-pipeline-and-what-we-want-to-change.md)
**Next:** [Part 8: Glossary and Cheat Sheet](./08-glossary-and-cheat-sheet.md)
