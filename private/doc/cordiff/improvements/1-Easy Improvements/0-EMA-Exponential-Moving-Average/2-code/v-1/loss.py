"""Loss function, optimizer, and patch gradient accumulation configuration."""

import torch
from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss, RegressionLossCE


def configure_patch_gradient_accumulation(cfg, patch_nums_iter, patching, logger0):
    """Configure patch gradient accumulation settings."""
    if cfg.model.name in {"patched_diffusion", "lt_aware_patched_diffusion"}:
        if len(patch_nums_iter) > 1:
            if not patching:
                logger0.info("Patching is not enabled: patch gradient accumulation automatically disabled.")
                use_patch_grad_acc = False
            else:
                use_patch_grad_acc = True
        else:
            use_patch_grad_acc = False
    else:
        logger0.info("Training a non-patched model: patch gradient accumulation automatically disabled.")
        use_patch_grad_acc = None

    return use_patch_grad_acc


def create_loss_function(cfg, regression_net, prob_channels):
    """Create appropriate loss function based on model type."""
    if cfg.model.name in ("diffusion", "patched_diffusion", "lt_aware_patched_diffusion"):
        return ResidualLoss(
            regression_net=regression_net,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name in ("regression", "lt_aware_regression"):
        return RegressionLoss()
    elif cfg.model.name == "lt_aware_ce_regression":
        return RegressionLossCE(prob_channels=prob_channels)


def create_optimizer(model, cfg):
    """Create and return the optimizer."""
    return torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.training.hp.lr,
        betas=[0.9, 0.999],
        eps=1e-8,
        fused=True,
    )


def calculate_patch_iterations(cfg, batch_size_per_gpu, logger0):
    """Calculate patch iterations for gradient accumulation."""
    patch_num = getattr(cfg.training.hp, "patch_num", 1)

    if hasattr(cfg.training.hp, "max_patch_per_gpu"):
        max_patch_per_gpu = cfg.training.hp.max_patch_per_gpu
        if max_patch_per_gpu // batch_size_per_gpu < 1:
            raise ValueError(
                f"max_patch_per_gpu ({max_patch_per_gpu}) must be >= batch_size_per_gpu ({batch_size_per_gpu})."
            )

        max_patch_num_per_iter = min(patch_num, (max_patch_per_gpu // batch_size_per_gpu))
        patch_iterations = (patch_num + max_patch_num_per_iter - 1) // max_patch_num_per_iter
        patch_nums_iter = [
            min(max_patch_num_per_iter, patch_num - i * max_patch_num_per_iter)
            for i in range(patch_iterations)
        ]

        logger0.info(
            f"max_patch_num_per_iter is {max_patch_num_per_iter}, "
            f"patch_iterations is {patch_iterations}, patch_nums_iter is {patch_nums_iter}"
        )
    else:
        patch_nums_iter = [patch_num]

    return patch_nums_iter
