# 0-beginner: Atomic Base Concepts for Diffusion Models

This folder contains the most fundamental, atomic concepts needed to understand diffusion models. Each file is self-contained and explains **one concept** with minimal dependencies.

**Reading order:** Follow the numbering. Each concept builds only lightly on previous ones.

| # | File | Concept | One-liner |
|---|------|---------|-----------|
| 0 | [0-probability-distribution.md](0-probability-distribution.md) | Probability Distribution | What it means for data to "come from" a distribution |
| 1 | [1-gaussian-distribution.md](1-gaussian-distribution.md) | Gaussian (Normal) Distribution | The bell curve — the most important distribution in diffusion |
| 2 | [2-sampling.md](2-sampling.md) | Sampling | Drawing random values from a distribution |
| 3 | [3-mean-variance-std.md](3-mean-variance-std.md) | Mean, Variance, Standard Deviation | The numbers that describe a distribution's shape |
| 4 | [4-random-variable.md](4-random-variable.md) | Random Variable | A variable whose value is determined by a random process |
| 5 | [5-conditional-probability.md](5-conditional-probability.md) | Conditional Probability | Probability of A given that B happened |
| 6 | [6-bayes-theorem.md](6-bayes-theorem.md) | Bayes' Theorem | Reversing conditional probabilities |
| 7 | [7-kl-divergence.md](7-kl-divergence.md) | KL Divergence | Measuring how different two distributions are |
| 8 | [8-log-likelihood.md](8-log-likelihood.md) | Log-Likelihood | How we measure if a model explains data well |
| 9 | [9-neural-network-as-function-approximator.md](9-neural-network-as-function-approximator.md) | Neural Network as Function Approximator | Why we use neural nets to learn unknown functions |
| 10 | [10-noise.md](10-noise.md) | Noise (in the context of diffusion) | What noise means and why it is central to diffusion |
| 11 | [11-signal-to-noise-ratio.md](11-signal-to-noise-ratio.md) | Signal-to-Noise Ratio (SNR) | Quantifying how much signal vs. noise is present |
| 12 | [12-markov-chain.md](12-markov-chain.md) | Markov Chain | A sequence where next state depends only on current state |
| 13 | [13-latent-variable.md](13-latent-variable.md) | Latent Variable | Hidden variables we never directly observe |
| 14 | [14-generative-model.md](14-generative-model.md) | Generative Model | A model that can create new data samples |
| 15 | [15-loss-function.md](15-loss-function.md) | Loss Function & MSE | How we train models by measuring errors |
| 16 | [16-reparameterization-trick.md](16-reparameterization-trick.md) | Reparameterization Trick | Making random sampling differentiable |
| 17 | [17-score-function.md](17-score-function.md) | Score Function (∇ log p) | The gradient of log-probability — key to score-based models |
| 18 | [18-stochastic-differential-equation.md](18-stochastic-differential-equation.md) | Stochastic Differential Equation (SDE) | Differential equations with randomness |
| 19 | [19-ordinary-differential-equation.md](19-ordinary-differential-equation.md) | Ordinary Differential Equation (ODE) | Deterministic differential equations used in diffusion |
| 20 | [20-denoising.md](20-denoising.md) | Denoising | Removing noise from a corrupted signal |
| 21 | [21-timestep-and-noise-schedule.md](21-timestep-and-noise-schedule.md) | Timestep & Noise Schedule | How noise is added gradually over time |
| 22 | [22-forward-process.md](22-forward-process.md) | Forward Process (Diffusion Process) | Gradually destroying data with noise |
| 23 | [23-reverse-process.md](23-reverse-process.md) | Reverse Process (Denoising Process) | Gradually recovering data from noise |
| 24 | [24-unet-architecture.md](24-unet-architecture.md) | U-Net Architecture | The workhorse neural network of diffusion models |
| 25 | [25-elbo.md](25-elbo.md) | Evidence Lower Bound (ELBO) | The tractable objective we optimize instead of likelihood |
