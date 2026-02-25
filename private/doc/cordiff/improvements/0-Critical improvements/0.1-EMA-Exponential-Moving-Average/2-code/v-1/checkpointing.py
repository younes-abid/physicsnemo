"""Checkpoint utilities: list, save, load, and cleanup of training checkpoints."""

import os
import torch

from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)
from helpers.train_helpers import is_time_for_periodic_task


def checkpoint_list(path, suffix=".mdlus"):
    """Helper function to return sorted list, in ascending order, of checkpoints in a path."""
    checkpoints = []
    if not os.path.exists(path):
        os.makedirs(path)
    for file in os.listdir(path):
        if file.endswith(suffix):
            try:
                index = int(file.split(".")[-2])
                checkpoints.append((index, file))
            except ValueError:
                continue
    checkpoints.sort(key=lambda x: x[0])
    return [file for _, file in checkpoints]


def setup_checkpoint_and_seeds(dist, cfg, cur_nimg):
    """Setup checkpoint directory and configure seeds/CUDA settings."""
    from helpers.train_helpers import set_seed, configure_cuda_for_consistent_precision

    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
    )
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()
    return checkpoint_dir


def checkpoint_block(cfg, dist, logger0, checkpoint_dir, model, optimizer, cur_nimg, done):
    """Save checkpoints periodically during training."""
    if dist.world_size > 1:
        torch.distributed.barrier()

    if is_time_for_periodic_task(
        cur_nimg,
        cfg.training.io.save_checkpoint_freq,
        done,
        cfg.training.hp.total_batch_size,
        dist.rank,
        rank_0_only=True,
    ):
        save_checkpoint(
            path=checkpoint_dir,
            models=model,
            optimizer=optimizer,
            epoch=cur_nimg,
        )


def cleanup_checkpoints_block(cfg, checkpoint_dir, dist):
    """Retain only the recent n checkpoints, if desired."""
    if dist.rank == 0 and cfg.training.io.save_n_recent_checkpoints > 0:
        for suffix in [".mdlus", ".pt"]:
            ckpts = checkpoint_list(checkpoint_dir, suffix=suffix)
            while len(ckpts) > cfg.training.io.save_n_recent_checkpoints:
                os.remove(os.path.join(checkpoint_dir, ckpts[0]))
                ckpts = ckpts[1:]
