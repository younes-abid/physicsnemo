"""Exponential Moving Average (EMA) for model weights.

EMA maintains a smoothed copy of the training weights:
    θ_ema = decay * θ_ema + (1 - decay) * θ_train

The EMA model is never trained directly — it is only used for
inference/checkpoint saving. Standard decay for diffusion models: 0.9999.
"""

import os
import copy
import torch

from checkpointing import checkpoint_list


def create_ema_model(model, dist, logger0):
    """Create an EMA copy of the training model.

    Deep-copies the unwrapped model (strips DDP / torch.compile wrappers),
    sets it to eval mode with no gradients.

    Parameters
    ----------
    model : torch.nn.Module
        Training model (may be wrapped with DDP or torch.compile).
    dist : DistributedManager
        Distributed manager for device placement.
    logger0 : RankZeroLoggingWrapper
        Logger (rank-0 only).

    Returns
    -------
    torch.nn.Module
        The EMA model copy on the same device.
    """
    raw_model = model
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if isinstance(raw_model, torch._dynamo.eval_frame.OptimizedModule):
        raw_model = raw_model._orig_mod

    ema_model = copy.deepcopy(raw_model)
    ema_model.eval().requires_grad_(False)
    ema_model.to(dist.device)

    logger0.info("EMA model created as a deep copy of the training model.")
    return ema_model


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA parameters: θ_ema = decay * θ_ema + (1-decay) * θ_train.

    Parameters
    ----------
    ema_model : torch.nn.Module
        The EMA model (unwrapped, on device).
    model : torch.nn.Module
        The training model (may be DDP-wrapped).
    decay : float
        EMA decay rate. Higher = smoother, slower to adapt.
    """
    raw_model = model
    if hasattr(raw_model, "module"):
        raw_model = raw_model.module
    if isinstance(raw_model, torch._dynamo.eval_frame.OptimizedModule):
        raw_model = raw_model._orig_mod

    for p_ema, p_train in zip(ema_model.parameters(), raw_model.parameters()):
        p_ema.data.mul_(decay).add_(p_train.data, alpha=1.0 - decay)


def save_ema_checkpoint(ema_model, checkpoint_dir, cur_nimg, dist, logger0):
    """Save EMA model state dict alongside the main checkpoint.

    Saved as: {checkpoint_dir}/ema_model.0.{cur_nimg}.pt

    Parameters
    ----------
    ema_model : torch.nn.Module
        The EMA model to save.
    checkpoint_dir : str
        Path to the checkpoint directory.
    cur_nimg : int
        Current training step (number of images processed).
    dist : DistributedManager
        Distributed manager.
    logger0 : RankZeroLoggingWrapper
        Logger (rank-0 only).
    """
    if dist.rank == 0:
        ema_path = os.path.join(checkpoint_dir, f"ema_model.0.{cur_nimg}.pt")
        torch.save(ema_model.state_dict(), ema_path)
        logger0.info(f"Saved EMA checkpoint: {ema_path}")


def load_ema_checkpoint(ema_model, checkpoint_dir, dist, logger0):
    """Load the latest EMA checkpoint from the checkpoint directory.

    Looks for files matching: {checkpoint_dir}/ema_model.0.*.pt
    and loads the one with the highest index.

    Parameters
    ----------
    ema_model : torch.nn.Module
        The EMA model to load weights into.
    checkpoint_dir : str
        Path to the checkpoint directory.
    dist : DistributedManager
        Distributed manager.
    logger0 : RankZeroLoggingWrapper
        Logger (rank-0 only).
    """
    ema_files = checkpoint_list(checkpoint_dir, suffix=".pt")
    ema_files = [f for f in ema_files if f.startswith("ema_model")]
    if ema_files:
        latest_ema = os.path.join(checkpoint_dir, ema_files[-1])
        ema_model.load_state_dict(torch.load(latest_ema, map_location=dist.device))
        logger0.info(f"Loaded EMA checkpoint: {latest_ema}")
    else:
        logger0.info("No EMA checkpoint found, starting EMA from current model weights.")
