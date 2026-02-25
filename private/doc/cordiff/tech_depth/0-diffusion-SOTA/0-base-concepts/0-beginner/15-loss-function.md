# Loss Function & Mean Squared Error (MSE)

## What is a Loss Function?

A **loss function** (also called **cost function** or **objective function**) is a mathematical formula that measures **how wrong** a model's prediction is compared to the true answer. Training a neural network means finding parameters $\theta$ that **minimize** the loss.

$$\theta^* = \arg\min_\theta \; L(\theta)$$

A loss of 0 means perfect prediction. Higher loss = worse prediction.

## Mean Squared Error (MSE)

The **MSE** is the most common loss function in diffusion models. It measures the average squared difference between prediction and target:

$$L_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} \|y_i - \hat{y}_i\|^2$$

Where:
- $y_i$ = true value (target / ground truth)
- $\hat{y}_i$ = predicted value
- $N$ = number of data points
- $\|\cdot\|^2$ = squared Euclidean norm (sum of squared differences across all dimensions)

### For a single data point (vector-valued):

$$L = \|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \sum_{j=1}^{d} (y_j - \hat{y}_j)^2$$

This sums the squared error across all $d$ dimensions (e.g., all pixels of an image).

## Why MSE?

### 1. Connection to Gaussian Likelihood
MSE loss is mathematically equivalent to **maximum likelihood estimation** under a Gaussian noise model (see [8-log-likelihood.md](8-log-likelihood.md)).

If we assume:
$$y = f_\theta(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$

Then maximizing the log-likelihood of the data is equivalent to minimizing:
$$-\log p(y \mid x, \theta) \propto \|y - f_\theta(x)\|^2$$

> 💡 This is why MSE appears naturally in diffusion models — the noise is Gaussian, so MSE is the principled loss.

### 2. Simplicity and Differentiability
- Easy to compute
- Smooth everywhere → well-behaved gradients for backpropagation
- Penalizes large errors more than small ones (quadratic penalty)

## MSE in Diffusion Models

### The DDPM Loss (Simplified)

The training loss for DDPM (and most diffusion models) is an MSE between the **true noise** $\boldsymbol{\epsilon}$ and the **predicted noise** $\boldsymbol{\epsilon}_\theta$:

$$L_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}} \left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

Breaking this down:
- Sample a clean image $\mathbf{x}_0$ from the dataset
- Sample a random timestep $t \sim \text{Uniform}(1, T)$
- Sample noise $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- Create noisy image: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$
- Predict noise: $\hat{\boldsymbol{\epsilon}} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$
- Compute loss: $L = \|\boldsymbol{\epsilon} - \hat{\boldsymbol{\epsilon}}\|^2$

### Different Prediction Targets, Same MSE Structure

| What the network predicts | Loss |
|--------------------------|------|
| Noise ($\boldsymbol{\epsilon}$-prediction) | $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$ |
| Clean data ($\mathbf{x}_0$-prediction) | $\|\mathbf{x}_0 - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, t)\|^2$ |
| Score ($\mathbf{s}$-prediction) | $\|\nabla_{\mathbf{x}} \log q(\mathbf{x}_t \mid \mathbf{x}_0) - \mathbf{s}_\theta(\mathbf{x}_t, t)\|^2$ |

All use MSE — the only difference is **what** we're comparing.

## Weighted Loss

In practice, the loss is often **weighted** by a factor $\lambda(t)$ that depends on the timestep:

$$L = \mathbb{E}_{t}\left[\lambda(t) \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]$$

Different choices of $\lambda(t)$ give different emphasis to different noise levels:
- $\lambda(t) = 1$ (DDPM "simple" loss) — equal weight to all timesteps
- $\lambda(t)$ from the ELBO derivation — theory-optimal but noisier in practice
- $\lambda(t)$ from EDM — carefully designed based on SNR analysis

> 💡 The choice of $\lambda(t)$ is one of the key design decisions that differs across papers. It significantly affects sample quality.

## Other Loss Functions (Brief Mentions)

| Loss | Formula | When used |
|------|---------|-----------|
| **L1 (MAE)** | $\|y - \hat{y}\|_1 = \sum |y_j - \hat{y}_j|$ | Sometimes used; less sensitive to outliers |
| **Perceptual Loss** | Distance in feature space of a pretrained network | When visual quality matters more than pixel accuracy |
| **Adversarial Loss** | From a discriminator network | Used in GANs and some hybrid diffusion models |

## Notation You Will See in Papers

| Symbol | Meaning |
|--------|---------|
| $L$ or $\mathcal{L}$ | Loss function |
| $L_{\text{simple}}$ | The simplified DDPM loss |
| $L_{\text{vlb}}$ | Variational lower bound loss (from ELBO) |
| $\lambda(t)$ or $w(t)$ | Timestep-dependent loss weight |
| $\|\cdot\|^2$ | Squared L2 norm (sum of squares) |
| $\mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}[\cdot]$ | Expectation over timesteps, data, and noise |

## Knowledge Check ✅

1. What does a loss function measure?
2. Write the MSE formula for comparing two vectors $\mathbf{y}$ and $\hat{\mathbf{y}}$.
3. Why is MSE the natural loss for diffusion models? (Hint: connection to Gaussian noise)
4. In the DDPM training loss, what are the two quantities being compared?
5. What role does the weight $\lambda(t)$ play in the loss, and why do different papers choose it differently?
6. Name the five steps of computing the DDPM training loss for one sample.

### Answers

1. **A loss function measures how wrong a model's prediction is compared to the true answer.** It assigns a non-negative scalar value where 0 means perfect prediction and larger values mean worse predictions. Training is the process of finding parameters $\theta^* = \arg\min_\theta L(\theta)$ that minimize this measure of error. The loss function defines what "good" means for the model — different losses encode different notions of quality.

2. $$L_{\text{MSE}} = \|\mathbf{y} - \hat{\mathbf{y}}\|^2 = \sum_{j=1}^{d} (y_j - \hat{y}_j)^2$$
   This sums the squared difference between each corresponding element of the two vectors across all $d$ dimensions. For images, $d$ is the total number of pixel values (height × width × channels). The squaring ensures all errors are positive and penalizes large errors disproportionately more than small ones.

3. **Because MSE is mathematically equivalent to maximum likelihood estimation under Gaussian noise.** Diffusion models add Gaussian noise, so the forward process noise is $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. If we model the prediction error as Gaussian — $\boldsymbol{\epsilon} = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \text{error}$, where $\text{error} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$ — then maximizing the log-likelihood of the true noise given the prediction is equivalent to minimizing $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$. MSE isn't just a convenient choice — it's the *principled* loss that falls out of the probabilistic derivation.

4. **The true noise $\boldsymbol{\epsilon}$ and the predicted noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$.** Specifically:
   - $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ is the actual Gaussian noise that was added to create $\mathbf{x}_t$
   - $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$ is the neural network's prediction of what that noise was, given only the noisy image $\mathbf{x}_t$ and the timestep $t`
   
   The loss is $\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2`. The network essentially learns to answer: "Given this noisy image at this noise level, what noise was added?"

5. **$\lambda(t)$ controls how much the model prioritizes learning to denoise at different noise levels.** Different timesteps correspond to different SNR regimes: early timesteps (low $t$) have little noise, late timesteps (high $t$) have lots of noise. The weight $\lambda(t)$ determines the relative importance of each regime in the total loss.
   
   Papers choose it differently because the optimal weighting depends on the application:
   - **DDPM ($\lambda(t) = 1$)**: Equal weight everywhere — simple and works well in practice, though it overweights high-noise timesteps relative to the ELBO-derived weighting.
   - **ELBO-derived $\lambda(t)$**: Theoretically optimal for maximizing the evidence lower bound, but produces noisier gradient estimates in practice.
   - **EDM $\lambda(t)$**: Carefully designed based on SNR analysis to balance contribution from all noise levels, leading to better sample quality.
   
   This is one of the most impactful hyperparameter choices in diffusion model design.

6. The five steps for one training sample:
   1. **Sample a clean image**: $\mathbf{x}_0 \sim p_{\text{data}}$ (draw from the training dataset)
   2. **Sample a random timestep**: $t \sim \text{Uniform}(1, T)$ (pick a random noise level)
   3. **Sample noise**: $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ (draw standard Gaussian noise)
   4. **Create the noisy image**: $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\,\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\epsilon}$ (add noise to the clean image using the closed-form formula)
   5. **Compute the loss**: $L = \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2$ (pass $\mathbf{x}_t` and $t` through the network, compare predicted noise to true noise)
   
   Then backpropagate this loss to update $\theta`. Repeat for many samples.

---
*Previous: [14-generative-model.md](14-generative-model.md) · Next: [16-reparameterization-trick.md](16-reparameterization-trick.md)*
