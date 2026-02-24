# Part 1: What Is a Diffusion Model?

## The Simplest Possible Explanation

A diffusion model is a neural network that learns to **remove noise from images**.

That's it. The entire magic of diffusion-based generation — photorealistic images,
weather forecasting, molecular design — comes down to one skill: given a noisy
image, predict the clean version.

## The Two Phases

### Phase 1: Adding Noise (Forward Process — No Learning)

Take a clean image and gradually destroy it by adding Gaussian noise:

```
Step 0          Step 1          Step 2          ...    Step T
(clean)         (slightly       (more           ...    (pure
                 noisy)          noisy)                 noise)

┌──────────┐   ┌──────────┐   ┌──────────┐         ┌──────────┐
│ ████████ │   │ ███▒██▒█ │   │ █▒▒▒█▒▒▒ │         │ ░▒░▒░▒░▒ │
│ ████████ │ → │ ██▒████▒ │ → │ ▒█▒▒▒▒█▒ │ → ... → │ ▒░▒░▒░▒░ │
│ ████████ │   │ █████▒██ │   │ ▒▒█▒▒▒▒█ │         │ ░▒░▒░▒░▒ │
└──────────┘   └──────────┘   └──────────┘         └──────────┘
```

This is pure math — no neural network involved. At each step, we just add a
little bit of random noise:

```
x_noisy = x_clean + σ × noise    where noise ~ N(0, I)
```

The parameter **σ** (sigma) controls how much noise is added. Small σ means
barely any noise; large σ means the image is almost entirely random static.

### Phase 2: Removing Noise (Reverse Process — The Neural Network)

Now train a neural network to reverse the process — given a noisy image and the
noise level σ, predict the clean image:

```
Step T          Step T-1        Step T-2        ...    Step 0
(pure           (slightly       (less           ...    (clean!)
 noise)          structured)     noisy)

┌──────────┐   ┌──────────┐   ┌──────────┐         ┌──────────┐
│ ░▒░▒░▒░▒ │   │ ▒▒▒█▒▒▒▒ │   │ ▒██▒██▒▒ │         │ ████████ │
│ ▒░▒░▒░▒░ │ → │ ▒▒▒▒█▒▒▒ │ → │ █▒████▒█ │ → ... → │ ████████ │
│ ░▒░▒░▒░▒ │   │ ▒▒▒▒▒▒█▒ │   │ ██▒███▒█ │         │ ████████ │
└──────────┘   └──────────┘   └──────────┘         └──────────┘
                  ↑               ↑                      ↑
              Neural net      Neural net             Neural net
              removes some    removes more           final cleanup
              noise           noise
```

**Key insight:** The neural network is called **many times** in sequence during
generation (typically 20–1000 times). Each call removes a little noise. The
calls are chained together to go from pure noise → clean output.

## Why This Matters for Weather

CorrDiff uses this exact process for weather super-resolution:

1. **Start** with a low-resolution weather forecast (e.g., ERA5 at 25km)
2. **Add noise** to a high-resolution target (e.g., WRF at 2km)
3. **Train** the neural network to denoise — given noisy HR + LR input, predict clean HR
4. **Generate** by starting from noise and iteratively denoising, conditioned on the LR input

The result: realistic high-resolution weather fields that are physically consistent
with the low-resolution input.

## The Training Recipe

For each training step:

```python
# 1. Sample a clean image
x_clean = dataset.get_random_sample()

# 2. Sample a random noise level
σ = sample_random_sigma()           # e.g., σ = 2.7

# 3. Add noise
noise = torch.randn_like(x_clean)
x_noisy = x_clean + σ * noise

# 4. Ask the model to denoise
x_predicted = model(x_noisy, σ)     # ← The model needs to know σ!

# 5. Compute loss
loss = (x_predicted - x_clean) ** 2

# 6. Update model weights
loss.backward()
optimizer.step()
```

**Notice step 4:** The model receives **two inputs** — the noisy image AND the
noise level σ. The model *must* know σ to decide how aggressively to denoise.
This is **noise conditioning**, and it's the topic of this document series.

## Summary

| Concept | What it means |
|---------|---------------|
| Diffusion model | Neural network that removes noise from images |
| Forward process | Adding noise (math, no learning) |
| Reverse process | Removing noise (neural network, learned) |
| σ (sigma) | How much noise was added — the model MUST know this |
| Noise conditioning | The mechanism for telling the model what σ is |
| Multi-step generation | The model is called 20-1000× in sequence; noise compounds |

---

**Next:** [Part 2: σ — The Noise Level](./02-sigma-the-noise-level.md)
