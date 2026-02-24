# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-FileCopyrightText: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CorrDiff training entry point — slim orchestrator that imports from modules.

Modules:
    setup.py           — distributed env, loggers, config, datasets, model, DDP
    loss.py            — loss function, optimizer, patch iterations
    training_loop.py   — gradient accumulation, training iteration block
    validation.py      — validation block
    metrics_logging.py — metrics, images, TensorBoard/terminal logging
    checkpointing.py   — checkpoint save/load/cleanup
    profiling.py       — CUDA profiler wrappers
    ema.py             — Exponential Moving Average (create, update, save, load)
"""

import os
import time

import hydra
from omegaconf import DictConfig
import torch
import wandb

from physicsnemo.launch.utils import load_checkpoint, get_checkpoint_dir
from helpers.train_helpers import (
    compute_num_accumulation_rounds,
    is_time_for_periodic_task,
)
from datasets.dataset import register_dataset

# ── Import from split modules ──
from setup import (
    initialize_distributed_environment,
    initialize_loggers,
    resolve_configurations,
    setup_performance_settings,
    initialize_datasets,
    configure_patching,
    create_model,
    setup_distributed_data_parallel,
    load_regression_checkpoint,
)
from loss import (
    configure_patch_gradient_accumulation,
    create_loss_function,
    create_optimizer,
    calculate_patch_iterations,
)
from training_loop import training_iteration_block
from validation import validation_block
from metrics_logging import log_progress_block
from checkpointing import (
    setup_checkpoint_and_seeds,
    checkpoint_block,
    cleanup_checkpoints_block,
)
from profiling import (
    cuda_profiler,
    cuda_profiler_start,
    cuda_profiler_stop,
    profiler_emit_nvtx,
)
from ema import create_ema_model, load_ema_checkpoint, save_ema_checkpoint

# ── torch dynamo settings ──
torch._dynamo.reset()
torch._dynamo.config.cache_size_limit = 264
torch._dynamo.config.verbose = True
torch._dynamo.config.suppress_errors = False
torch._logging.set_logs(recompiles=True, graph_breaks=True)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    """Main training function for CorrDiff model."""

    # ── 1. Initialization ──
    dist = initialize_distributed_environment()
    writer, logger, logger0 = initialize_loggers(dist, cfg)
    dataset_cfg, validation, validation_dataset_cfg = resolve_configurations(cfg)

    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")

    fp16, enable_amp, amp_dtype, songunet_checkpoint_level = setup_performance_settings(cfg)
    logger.info(f"Saving the outputs in {os.getcwd()}")

    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = cfg.training.hp.total_batch_size // dist.world_size

    # Load current number of images for resuming
    try:
        cur_nimg = load_checkpoint(path=get_checkpoint_dir(
            str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
        ))
    except Exception:
        cur_nimg = 0

    checkpoint_dir = setup_checkpoint_and_seeds(dist, cfg, cur_nimg)

    # ── 2. Datasets ──
    dataset, dataset_iterator, validation_dataset, validation_dataset_iterator = initialize_datasets(
        cfg, dataset_cfg, validation, validation_dataset_cfg, cur_nimg, logger0
    )

    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())

    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    patching, use_patching, prob_channels, img_shape = configure_patching(
        cfg, dataset, img_shape, logger0
    )
    if use_patching:
        img_in_channels += dataset_channels

    # ── 3. Model ──
    model, use_torch_compile, use_apex_gn = create_model(
        cfg, img_in_channels, img_out_channels, img_shape, fp16,
        songunet_checkpoint_level, prob_channels, enable_amp, dist, logger0
    )
    model = setup_distributed_data_parallel(model, dist)

    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(model)

    try:
        load_checkpoint(path=checkpoint_dir, models=model)
    except Exception:
        pass

    profile_mode = getattr(cfg.training.perf, "profile_mode", False)
    regression_net = load_regression_checkpoint(
        cfg, use_apex_gn, enable_amp, profile_mode, dist, logger0
    )

    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)

    # ── 4. Training components ──
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    patch_nums_iter = calculate_patch_iterations(cfg, batch_size_per_gpu, logger0)
    use_patch_grad_acc = configure_patch_gradient_accumulation(
        cfg, patch_nums_iter, patching, logger0
    )
    loss_fn = create_loss_function(cfg, regression_net, prob_channels)
    optimizer = create_optimizer(model, cfg)

    # ── 5. EMA setup ──
    ema_decay = getattr(cfg.training.hp, "ema_decay", 0.9999)
    if ema_decay and ema_decay > 0:
        ema_model = create_ema_model(model, dist, logger0)
        load_ema_checkpoint(ema_model, checkpoint_dir, dist, logger0)
        logger0.info(f"EMA enabled with decay={ema_decay}")
    else:
        ema_model = None
        logger0.info("EMA disabled (ema_decay is 0 or null).")

    # ── 6. Load optimizer checkpoint ──
    start_time = time.time()

    if dist.world_size > 1:
        torch.distributed.barrier()
    if cfg.training.io.get("load_optimizer", True):
        try:
            load_checkpoint(
                path=checkpoint_dir,
                optimizer=optimizer,
                device=dist.device,
            )
        except Exception:
            pass

    ########################################################################
    #                          MAIN TRAINING LOOP                          #
    ########################################################################
    logger0.info("-" * 80)
    logger0.info(f"Model initialized: {cfg.model.name}, on device: {dist.device}")
    logger0.info("-" * 80)
    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    with cuda_profiler():
        with profiler_emit_nvtx():
            while not done:
                tick_start_nimg = cur_nimg
                tick_start_time = time.time()

                if cur_nimg - start_nimg == 24 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Starting Profiler at {cur_nimg}")
                    cuda_profiler_start()

                if cur_nimg - start_nimg == 25 * cfg.training.hp.total_batch_size:
                    logger0.info(f"Stopping Profiler at {cur_nimg}")
                    cuda_profiler_stop()

                # ── Training step ──
                (average_loss, average_loss_running_mean,
                 n_average_loss_running_mean, current_lr,
                 cur_nimg, done, metrics) = training_iteration_block(
                    cfg, dist, writer, dataset_iterator, use_apex_gn, input_dtype,
                    model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                    enable_amp, amp_dtype, batch_size_per_gpu, num_accumulation_rounds,
                    average_loss_running_mean, n_average_loss_running_mean,
                    optimizer, cur_nimg, done, compute_metrics_flag=True,
                    ema_model=ema_model, ema_decay=ema_decay,
                )

                # ── Validation ──
                validation_block(
                    cfg, dist, writer, logger0, validation_dataset_iterator, use_apex_gn,
                    model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                    enable_amp, amp_dtype, batch_size_per_gpu, cur_nimg, done,
                    compute_metrics_flag=True,
                )

                # ── Log progress ──
                if is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.print_progress_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    log_progress_block(
                        cur_nimg, average_loss, average_loss_running_mean, current_lr,
                        start_time, tick_start_time, tick_start_nimg, dist, logger0, metrics,
                    )

                # ── Checkpoint (training model + EMA) ──
                checkpoint_block(
                    cfg, dist, logger0, checkpoint_dir, model, optimizer, cur_nimg, done,
                )
                # Save EMA checkpoint alongside the main checkpoint
                if ema_model is not None and is_time_for_periodic_task(
                    cur_nimg,
                    cfg.training.io.save_checkpoint_freq,
                    done,
                    cfg.training.hp.total_batch_size,
                    dist.rank,
                    rank_0_only=True,
                ):
                    save_ema_checkpoint(ema_model, checkpoint_dir, cur_nimg, dist, logger0)

                # ── Cleanup old checkpoints ──
                cleanup_checkpoints_block(cfg, checkpoint_dir, dist)

    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()