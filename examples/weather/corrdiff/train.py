# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import os
import time
import psutil
from contextlib import nullcontext

import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import nvtx
import wandb

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import RegressionLoss, ResidualLoss, RegressionLossCE
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.launch.utils import (
    load_checkpoint,
    save_checkpoint,
    get_checkpoint_dir,
)

from datasets.dataset import init_train_valid_datasets_from_config, register_dataset
from helpers.train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    is_time_for_periodic_task,
)

from metrics.regression_metrics import compute_mae, compute_mse, compute_r2

torch._dynamo.reset()
# Increase the cache size limit
torch._dynamo.config.cache_size_limit = 264  # Set to a higher value
torch._dynamo.config.verbose = True  # Enable verbose logging
torch._dynamo.config.suppress_errors = False  # Forces the error to show all details
torch._logging.set_logs(recompiles=True, graph_breaks=True)

import torchvision.utils as vutils

def checkpoint_list(path, suffix=".mdlus"):
    """Helper function to return sorted list, in ascending order, of checkpoints in a path"""
    checkpoints = []
    for file in os.listdir(path):
        if file.endswith(suffix):
            # Split the filename and extract the index
            try:
                index = int(file.split(".")[-2])
                checkpoints.append((index, file))
            except ValueError:
                continue

    # Sort by index and return filenames
    checkpoints.sort(key=lambda x: x[0])
    return [file for _, file in checkpoints]


# Define safe CUDA profiler tools that fallback to no-ops when CUDA is not available
def cuda_profiler():
    if torch.cuda.is_available():
        return torch.cuda.profiler.profile()
    else:
        return nullcontext()


def cuda_profiler_start():
    if torch.cuda.is_available():
        torch.cuda.profiler.start()


def cuda_profiler_stop():
    if torch.cuda.is_available():
        torch.cuda.profiler.stop()


def profiler_emit_nvtx():
    if torch.cuda.is_available():
        return torch.autograd.profiler.emit_nvtx()
    else:
        return nullcontext()

def initialize_distributed_environment() -> tuple:
    """Initialize distributed environment for training."""
    DistributedManager.initialize()
    dist = DistributedManager()
    return dist

def initialize_loggers(dist, cfg) -> tuple:
    """Initialize all loggers including TensorBoard, Python logger, and WandB."""
    writer = None
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    
    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)
    
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name=f"CorrDiff-Training-{HydraConfig.get().job.name}",
        group="CorrDiff-DDP-Group",
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg),
        results_dir=cfg.wandb.results_dir,
    )
    
    logger0.info("-" * 80)
    logger0.info(f"WandB initialized with mode: {cfg.wandb.mode}, results_dir: {cfg.wandb.results_dir}")
    logger0.info("Full Hydra Configuration:")
    logger0.info(OmegaConf.to_yaml(cfg))
    logger0.info("-" * 80)
    
    return writer, logger, logger0

def resolve_configurations(cfg) -> tuple:
    """Resolve and parse configuration settings."""
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    
    validation = hasattr(cfg, "validation")
    validation_dataset_cfg = OmegaConf.to_container(cfg.validation) if validation else None
    
    return dataset_cfg, validation, validation_dataset_cfg

def setup_performance_settings(cfg) -> tuple:
    """Configure performance-related settings like FP optimizations."""
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    
    return fp16, enable_amp, amp_dtype, songunet_checkpoint_level

def setup_checkpoint_and_seeds(dist, cfg, cur_nimg) -> str:
    """Setup checkpoint directory and configure seeds/CUDA settings."""
    checkpoint_dir = get_checkpoint_dir(
        str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
    )
    
    set_seed(dist.rank + cur_nimg)
    configure_cuda_for_consistent_precision()
    
    return checkpoint_dir

def initialize_datasets(cfg, dataset_cfg, validation, validation_dataset_cfg, cur_nimg, logger0) -> tuple:
    """Initialize training and validation datasets."""
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2 if cfg.training.perf.dataloader_workers > 0 else None,
    }
    
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
        validation=validation,
        sampler_start_idx=cur_nimg,
    )
    
    logger0.info("-" * 80)
    logger0.info(f"Training dataset initialized with {len(dataset)} samples.")
    if validation:
        logger0.info(f"Validation dataset initialized with {len(validation_dataset)} samples.")
    logger0.info("-" * 80)
    
    return dataset, dataset_iterator, validation_dataset, validation_dataset_iterator

def configure_patching(cfg, dataset, img_shape, logger0) -> tuple:
    """Configure patch-based training settings."""
    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None
    
    # Parse patch shape
    if cfg.model.name in ["patched_diffusion", "lt_aware_patched_diffusion"]:
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None
    
    # Check if patch shape is larger than image
    if (patch_shape_x and patch_shape_y and 
        patch_shape_y >= img_shape[0] and patch_shape_x >= img_shape[1]):
        logger0.warning(
            f"Patch shape {patch_shape_y}x{patch_shape_x} is larger than "
            f"the image shape {img_shape[0]}x{img_shape[1]}. Patching will not be used."
        )
    
    patch_shape = (patch_shape_y, patch_shape_x)
    use_patching, img_shape, patch_shape = set_patch_shape(img_shape, patch_shape)
    
    if use_patching:
        patching = RandomPatching2D(
            img_shape=img_shape,
            patch_shape=patch_shape,
            patch_num=getattr(cfg.training.hp, "patch_num", 1),
        )
        logger0.info("Patch-based training enabled")
    else:
        patching = None
        logger0.info("Patch-based training disabled")
    
    return patching, use_patching, prob_channels, img_shape

def create_model(cfg, img_in_channels, img_out_channels, img_shape, fp16, 
                songunet_checkpoint_level, prob_channels, enable_amp, dist, logger0) -> torch.nn.Module:
    """Create and configure the model based on configuration."""
    model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }
    
    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels
    
    # Update model args from config
    if hasattr(cfg.model, "model_args"):
        model_args.update(OmegaConf.to_container(cfg.model.model_args))
    
    # Handle performance settings
    use_torch_compile = getattr(cfg.training.perf, "torch_compile", False)
    use_apex_gn = getattr(cfg.training.perf, "use_apex_gn", False)
    profile_mode = getattr(cfg.training.perf, "profile_mode", False)
    
    if use_apex_gn:
        model_args["use_apex_gn"] = use_apex_gn
    if profile_mode:
        model_args["profile_mode"] = profile_mode
    if enable_amp:
        model_args["amp_mode"] = enable_amp
    
    # Create appropriate model based on type
    if cfg.model.name == "regression":
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"],
            **model_args,
        )
    elif cfg.model.name in ["lt_aware_ce_regression", "lt_aware_regression"]:
        model = UNet(
            img_in_channels=img_in_channels + model_args["N_grid_channels"] + model_args["lead_time_channels"],
            **model_args,
        )
    elif cfg.model.name in ["lt_aware_patched_diffusion", "diffusion", "patched_diffusion"]:
        model = EDMPrecondSuperResolution(
            img_in_channels=img_in_channels + model_args["N_grid_channels"] + 
            (model_args["lead_time_channels"] if "lt_aware" in cfg.model.name else 0),
            **model_args,
        )
    else:
        raise ValueError(f"Invalid model: {cfg.model.name}")
    
    # Configure model
    model.train().requires_grad_(True).to(dist.device)
    
    if use_apex_gn:
        model.to(memory_format=torch.channels_last)
    
    # Check for invalid combinations
    if (cfg.model.name in ["regression", "lt_aware_regression", "lt_aware_ce_regression"] and 
        getattr(cfg.training.hp, "patch_shape_x", None) is not None):
        raise ValueError(
            f"Regression model ({cfg.model.name}) cannot be used with patch-based training."
        )
    
    return model, use_torch_compile, use_apex_gn

def setup_distributed_data_parallel(model, dist) -> torch.nn.Module:
    """Setup distributed data parallel if applicable."""
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=True,
            bucket_cap_mb=35,
            gradient_as_bucket_view=True,
        )
    return model

def load_regression_checkpoint(cfg, use_apex_gn, enable_amp, profile_mode, dist, logger0):
    """Load regression checkpoint if specified in config."""
    regression_net = None
    if (hasattr(cfg.training.io, "regression_checkpoint_path") and 
        cfg.training.io.regression_checkpoint_path is not None):
        
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                f"Expected this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        
        regression_net = Module.from_checkpoint(
            regression_checkpoint_path, override_args={"use_apex_gn": use_apex_gn}
        )
        regression_net.amp_mode = enable_amp
        regression_net.profile_mode = profile_mode
        regression_net.eval().requires_grad_(False).to(dist.device)
        
        if use_apex_gn:
            regression_net.to(memory_format=torch.channels_last)
        
        logger0.success("Loaded the pre-trained regression model")
    
    return regression_net

def configure_patch_gradient_accumulation(cfg, patch_nums_iter, patching, logger0) -> bool:
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

def calculate_patch_iterations(cfg, batch_size_per_gpu, logger0) -> list:
    """Calculate patch iterations for gradient accumulation."""
    patch_num = getattr(cfg.training.hp, "patch_num", 1)
    
    if hasattr(cfg.training.hp, "max_patch_per_gpu"):
        max_patch_per_gpu = cfg.training.hp.max_patch_per_gpu
        if max_patch_per_gpu // batch_size_per_gpu < 1:
            raise ValueError(
                f"max_patch_per_gpu ({max_patch_per_gpu}) must be greater or equal to batch_size_per_gpu ({batch_size_per_gpu})."
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

########### main loop functions ###########
def load_and_prepare_batch(dataset_iterator, use_apex_gn, dist, input_dtype):
    """Load and prepare a batch of data."""
    with nvtx.annotate("loading data", color="green"):
        img_clean, img_lr, *lead_time_label = next(dataset_iterator)
        
        if use_apex_gn:
            img_clean = img_clean.to(
                dist.device,
                dtype=input_dtype,
                non_blocking=True,
            ).to(memory_format=torch.channels_last)
            img_lr = img_lr.to(
                dist.device,
                dtype=input_dtype,
                non_blocking=True,
            ).to(memory_format=torch.channels_last)
        else:
            img_clean = (
                img_clean.to(dist.device)
                .to(input_dtype)
                .contiguous()
            )
            img_lr = (
                img_lr.to(dist.device)
                .to(input_dtype)
                .contiguous()
            )
    
    return img_clean, img_lr, lead_time_label

def prepare_loss_kwargs(model, img_clean, img_lr, lead_time_label, use_patch_grad_acc, dist, loss_fn):
    """Prepare keyword arguments for loss function."""
    loss_fn_kwargs = {
        "net": model,
        "img_clean": img_clean,
        "img_lr": img_lr,
        "augment_pipe": None,
    }
    
    if use_patch_grad_acc is not None:
        loss_fn_kwargs["use_patch_grad_acc"] = use_patch_grad_acc

    if lead_time_label:
        lead_time_label_tensor = lead_time_label[0].to(dist.device).contiguous()
        loss_fn_kwargs.update({"lead_time_label": lead_time_label_tensor})
    else:
        lead_time_label_tensor = None
        
    if use_patch_grad_acc:
        loss_fn.y_mean = None

    return loss_fn_kwargs, lead_time_label_tensor

def compute_loss_and_predictions(loss_fn, loss_fn_kwargs, patching, patch_num_per_iter, 
                               enable_amp, amp_dtype, batch_size_per_gpu, return_predictions=False):
    """Compute loss and optionally return predictions for a single patch."""
    if patching is not None:
        patching.set_patch_num(patch_num_per_iter)
        loss_fn_kwargs.update({"patching": patching})
    
    with nvtx.annotate(f"loss forward", color="green"):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
            if return_predictions:
                loss, predictions = loss_fn(**loss_fn_kwargs, return_predictions=True)
            else:
                loss = loss_fn(**loss_fn_kwargs)
                predictions = None

    loss = loss.sum() / batch_size_per_gpu
    return loss, predictions

def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, prefix: str = "", 
                    variables: list = []) -> dict:
    """
    Compute various metrics between predictions and targets.
    
    Args:
        predictions (torch.Tensor): Predicted values of shape (B, C, H, W).
        targets (torch.Tensor): Ground truth values of shape (B, C, H, W).
        prefix (str): Prefix for metric names (e.g., "train_", "val_").
        variables: List of variable names corresponding to the channels.
        
    Returns:
        dict: Dictionary containing computed metrics.
    """
    metrics = {}
    thresholds = torch.tensor([0.1, 0.2, 0.5, 1.0], device=predictions.device)
    
    for i, var in enumerate(variables):
        p = predictions[:, i, :, :]
        t = targets[:, i, :, :]
        
        # Mask NaNs in targets (if applicable)
        mask = ~torch.isnan(t)
        p = p[mask]
        t = t[mask]
        
        # Basic regression metrics
        metrics[f'{prefix}mae_{var}'] = torch.mean(torch.abs(p - t))
        metrics[f'{prefix}mse_{var}'] = torch.mean((p - t) ** 2)
        metrics[f'{prefix}rmse_{var}'] = torch.sqrt(metrics[f'{prefix}mse_{var}'])
        
        # R-squared
        ss_total = torch.sum((t - torch.mean(t)) ** 2)
        ss_residual = torch.sum((t - p) ** 2)
        metrics[f'{prefix}r2_{var}'] = 1 - (ss_residual / (ss_total + 1e-8))
        
        # Relative errors
        abs_error = torch.abs(p - t)
        relative_error = abs_error / (torch.abs(t) + 1e-8)
        metrics[f'{prefix}relative_error_{var}'] = torch.mean(relative_error)
        metrics[f'{prefix}max_relative_error_{var}'] = torch.max(relative_error)
        
        # Relative error within thresholds
        within_threshold = (relative_error.unsqueeze(-1) < thresholds).float()
        for j, threshold in enumerate(thresholds):
            # Format the threshold to a fixed number of decimal places
            threshold_str = f"{threshold.item():.1f}"
            metrics[f'{prefix}relative_error_within_{threshold_str}_{var}'] = torch.mean(within_threshold[..., j])
    
    return metrics

def prepare_images(predictions_cat: torch.Tensor, targets_cat: torch.Tensor, prefix: str, 
                   variables: list, n: int = 1) -> dict:
    """
    Prepare images for logging to TensorBoard.

    Args:
        predictions_cat (torch.Tensor): Concatenated predicted values of shape (B, C, H, W).
        targets_cat (torch.Tensor): Concatenated ground truth values of shape (B, C, H, W).
        prefix (str): Prefix for logging (e.g., "training_", "validation_").
        variables (list): List of variable names corresponding to the channels.
        n (int): Number of images to prepare (default: 1).

    Returns:
        dict: Dictionary containing titles as keys and lists of images as values.
    """
    images = {}
    predictions = predictions_cat[:n]
    targets = targets_cat[:n]

    for i, var in enumerate(variables):
        pred = predictions[:, i, :, :].unsqueeze(1)  # Shape: (n, 1, H, W)
        target = targets[:, i, :, :].unsqueeze(1)    # Shape: (n, 1, H, W)

        # Return as a list of two images (predictions and targets)
        images[f"{prefix}{var}_pred_vs_target"] = [pred, target]

    return images

def aggregate_metrics(all_metrics, dist):
    """Aggregate metrics across all distributed processes."""
    aggregated_metrics = {}
    
    for metric_batch in all_metrics:
        for metric_name, metric_value in metric_batch.items():
            if metric_name not in aggregated_metrics:
                aggregated_metrics[metric_name] = []
            aggregated_metrics[metric_name].append(metric_value)
    
    # Average metrics across batches
    final_metrics = {}
    for metric_name, metric_values in aggregated_metrics.items():
        if metric_values:  # Check if list is not empty
            metric_tensor = torch.tensor(metric_values, device=dist.device)
            if dist.world_size > 1:
                torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
                metric_tensor /= dist.world_size
            final_metrics[metric_name] = metric_tensor.mean().item()
    
    return final_metrics

def accumulate_gradients_and_metrics(model, loss_fn, dataset_iterator, use_apex_gn, dist, input_dtype,
                                   use_patch_grad_acc, patching, patch_nums_iter, enable_amp, amp_dtype,
                                   batch_size_per_gpu, num_accumulation_rounds, cfg, compute_metrics_flag=True):
    """Accumulate gradients and compute metrics over multiple rounds."""
    loss_accum = 0
    all_metrics = []
    all_predictions = []
    all_targets = []
    
    for n_i in range(num_accumulation_rounds):
        with nvtx.annotate(f"accumulation round {n_i}", color="Magenta"):
            # Load and prepare batch
            img_clean, img_lr, lead_time_label = load_and_prepare_batch(
                dataset_iterator, use_apex_gn, dist, input_dtype
            )
            
            # Prepare loss function arguments
            loss_fn_kwargs, _ = prepare_loss_kwargs(
                model, img_clean, img_lr, lead_time_label, use_patch_grad_acc, dist, loss_fn
            )
            
            # Compute loss for each patch
            for patch_num_per_iter in patch_nums_iter:
                loss, predictions = compute_loss_and_predictions(
                    loss_fn, loss_fn_kwargs, patching, patch_num_per_iter,
                    enable_amp, amp_dtype, batch_size_per_gpu, return_predictions=compute_metrics_flag
                )
                
                loss_accum += loss / num_accumulation_rounds / len(patch_nums_iter)
                
                # Store predictions and targets for metric computation
                if compute_metrics_flag and predictions is not None:
                    all_predictions.append(predictions.detach())
                    all_targets.append(img_clean.detach())
                
                with nvtx.annotate(f"loss backward", color="yellow"):
                    loss.backward()
    
    # Compute metrics if requested
    metrics = {}
    images = {}
    if compute_metrics_flag and all_predictions:
        with torch.no_grad():
            predictions_cat = torch.cat(all_predictions)
            targets_cat = torch.cat(all_targets)
            metrics = compute_metrics(predictions_cat, targets_cat, prefix="training_", 
                                      variables=cfg.dataset.output_variables)
            images = prepare_images(predictions_cat=predictions_cat,
            targets_cat=targets_cat,
            prefix="training_",
            variables=cfg.dataset.output_variables,
            n=1
        )
    
    return loss_accum, metrics, images

def aggregate_loss(loss_accum, dist):
    """Aggregate loss across all distributed processes."""
    with nvtx.annotate(f"loss aggregate", color="green"):
        loss_sum = torch.tensor([loss_accum], device=dist.device)
        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(
                loss_sum, op=torch.distributed.ReduceOp.SUM
            )
        average_loss = (loss_sum / dist.world_size).cpu().item()
    
    return average_loss

def update_learning_rates(optimizer, cfg, cur_nimg, dist, writer):
    """Update learning rates based on schedule."""
    lr_rampup = cfg.training.hp.lr_rampup
    current_lr = None
    
    for g in optimizer.param_groups:
        if lr_rampup > 0:
            g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
        if cur_nimg >= lr_rampup:
            g["lr"] *= cfg.training.hp.lr_decay ** (
                (cur_nimg - lr_rampup) // cfg.training.hp.lr_decay_rate
            )
        current_lr = g["lr"]
        if dist.rank == 0:
            writer.add_scalar("learning_rate", current_lr, cur_nimg)
    
    return current_lr

def update_running_loss(average_loss, average_loss_running_mean, n_average_loss_running_mean):
    """Update running mean of average loss."""
    average_loss_running_mean += (
        average_loss - average_loss_running_mean
    ) / n_average_loss_running_mean
    n_average_loss_running_mean += 1
    
    return average_loss_running_mean, n_average_loss_running_mean

def log_training_metrics(writer, average_loss, average_loss_running_mean, cur_nimg, dist, metrics=None):
    """Log training metrics to tensorboard."""
    if dist.rank == 0:
        writer.add_scalar("training_loss", average_loss, cur_nimg)
        writer.add_scalar("training_loss_running_mean", average_loss_running_mean, cur_nimg)
        
        # Log additional metrics if available
        if metrics:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, cur_nimg)
                
def log_images(writer, cur_nimg, dist, images):
    """Log images to TensorBoard with clear separation between predictions and targets."""
    if dist.rank == 0:
        for title, image_list in images.items():
            # Add a black separator between images
            pred, target = image_list
            separator = torch.zeros_like(pred)  # Create a black separator with the same shape as the images
            separator_width = max(1, pred.shape[3] // 100)
            separator = separator[:, :, :, :separator_width]  # Make the separator 3 pixel wide

            # Concatenate predictions, separator, and targets
            pred_vs_target = torch.cat([pred, separator, target, separator, pred - target], dim=3)  

            # Log the concatenated image to TensorBoard
            writer.add_image(title, pred_vs_target[0], global_step=cur_nimg)  # Log the first image in the batch        

def training_iteration_block(cfg, dist, writer, dataset_iterator, use_apex_gn, input_dtype,
                           model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                           enable_amp, amp_dtype, batch_size_per_gpu, num_accumulation_rounds,
                           average_loss_running_mean, n_average_loss_running_mean,
                           optimizer, cur_nimg, done, compute_metrics_flag=True):
    """Refactored training iteration block with metrics computation."""
    with nvtx.annotate("Training iteration", color="green"):
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Accumulate gradients and compute metrics
        loss_accum, metrics, images = accumulate_gradients_and_metrics(
            model, loss_fn, dataset_iterator, use_apex_gn, dist, input_dtype,
            use_patch_grad_acc, patching, patch_nums_iter, enable_amp, amp_dtype,
            batch_size_per_gpu, num_accumulation_rounds, cfg, compute_metrics_flag
        )
        
        # Aggregate loss across processes
        average_loss = aggregate_loss(loss_accum, dist)
        
        # Update running loss statistics
        average_loss_running_mean, n_average_loss_running_mean = update_running_loss(
            average_loss, average_loss_running_mean, n_average_loss_running_mean
        )
        
        # Log metrics
        log_training_metrics(writer, average_loss, average_loss_running_mean, cur_nimg, dist, metrics)
        
        #Log images
        log_images(writer, cur_nimg, dist, images)
        
        # Check for periodic tasks
        ptt = is_time_for_periodic_task(
            cur_nimg,
            cfg.training.io.print_progress_freq,
            done,
            cfg.training.hp.total_batch_size,
            dist.rank,
            rank_0_only=True,
        )
        if ptt:
            average_loss_running_mean = 0
            n_average_loss_running_mean = 1
        
        # Update weights
        with nvtx.annotate("update weights", color="blue"):
            current_lr = update_learning_rates(optimizer, cfg, cur_nimg, dist, writer)
            handle_and_clip_gradients(
                model,
                grad_clip_threshold=cfg.training.hp.grad_clip_threshold,
            )
        
        with nvtx.annotate("optimizer step", color="blue"):
            optimizer.step()
        
        # Update iteration counters
        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration

    return average_loss, average_loss_running_mean, n_average_loss_running_mean, current_lr, cur_nimg, done, metrics

def validation_block(cfg, dist, writer, logger0, validation_dataset_iterator, use_apex_gn,
                     model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                     enable_amp, amp_dtype, batch_size_per_gpu, cur_nimg, done, compute_metrics_flag=True):
    """Refactored validation block with metrics computation and logging."""
    with nvtx.annotate("validation", color="red"):
        # Validation
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            all_predictions = []
            all_targets = []

            if is_time_for_periodic_task(
                cur_nimg,
                cfg.training.io.validation_freq,
                done,
                cfg.training.hp.total_batch_size,
                dist.rank,
            ):
                with torch.no_grad():
                    for _ in range(cfg.training.io.validation_steps):
                        (
                            img_clean_valid,
                            img_lr_valid,
                            *lead_time_label_valid,
                        ) = next(validation_dataset_iterator)

                        if use_apex_gn:
                            img_clean_valid = img_clean_valid.to(
                                dist.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            ).to(memory_format=torch.channels_last)
                            img_lr_valid = img_lr_valid.to(
                                dist.device,
                                dtype=torch.float32,
                                non_blocking=True,
                            ).to(memory_format=torch.channels_last)
                        else:
                            img_clean_valid = (
                                img_clean_valid.to(dist.device)
                                .to(torch.float32)
                                .contiguous()
                            )
                            img_lr_valid = (
                                img_lr_valid.to(dist.device)
                                .to(torch.float32)
                                .contiguous()
                            )

                        loss_valid_kwargs = {
                            "net": model,
                            "img_clean": img_clean_valid,
                            "img_lr": img_lr_valid,
                            "augment_pipe": None,
                        }
                        if use_patch_grad_acc is not None and hasattr(loss_fn, "use_patch_grad_acc"):
                            loss_valid_kwargs["use_patch_grad_acc"] = use_patch_grad_acc
                        if lead_time_label_valid:
                            lead_time_label_valid = (
                                lead_time_label_valid[0]
                                .to(dist.device)
                                .contiguous()
                            )
                            loss_valid_kwargs.update({"lead_time_label": lead_time_label_valid})
                        if use_patch_grad_acc:
                            loss_fn.y_mean = None

                        for patch_num_per_iter in patch_nums_iter:
                            if patching is not None:
                                patching.set_patch_num(patch_num_per_iter)
                                loss_valid_kwargs.update({"patching": patching})
                            with torch.autocast(
                                "cuda", dtype=amp_dtype, enabled=enable_amp
                            ):
                                loss_valid, predictions = loss_fn(
                                    **loss_valid_kwargs, return_predictions=compute_metrics_flag
                                )

                            loss_valid = (
                                (loss_valid.sum() / batch_size_per_gpu)
                                .cpu()
                                .item()
                            )
                            valid_loss_accum += (
                                loss_valid
                                / cfg.training.io.validation_steps
                            )

                            # Store predictions and targets for metrics computation
                            all_predictions.append(predictions.detach())
                            all_targets.append(img_clean_valid.detach())

                    # Aggregate validation loss across processes
                    valid_loss_sum = torch.tensor(
                        [valid_loss_accum], device=dist.device
                    )
                    if dist.world_size > 1:
                        torch.distributed.barrier()
                        torch.distributed.all_reduce(
                            valid_loss_sum,
                            op=torch.distributed.ReduceOp.SUM,
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size

                    # Compute validation metrics
                    metrics = {}
                    if all_predictions:
                        predictions_cat = torch.cat(all_predictions)
                        targets_cat = torch.cat(all_targets)
                        metrics = compute_metrics(predictions_cat, targets_cat, prefix="validation_", 
                                                  variables=cfg.validation.output_variables)
                        images = prepare_images(predictions_cat=predictions_cat,
                                targets_cat=targets_cat,
                                prefix="validation_",
                                variables=cfg.validation.output_variables,
                                n=1
                            )
                    
                    if dist.rank == 0:
                        # Log images to Tensorboard
                        log_images(writer, cur_nimg, dist, images)
                        
                        # Log validation loss and metrics to TensorBoard
                        writer.add_scalar("validation_loss", average_valid_loss, cur_nimg)
                        for metric_name, metric_value in metrics.items():
                            writer.add_scalar(metric_name, metric_value, cur_nimg)

                        # Log validation loss and metrics to the terminal
                        fields = [f"validation_loss {average_valid_loss.item():<7.2f}"]
                        for metric_name, metric_value in metrics.items():
                            fields += [f"{metric_name} {metric_value:<7.2f}"]
                        joined_fields = " ".join(fields)
                        logger0.info(f"\033[91m{joined_fields}\033[0m")

def log_progress_block(cur_nimg, average_loss, average_loss_running_mean, current_lr,
                      start_time, tick_start_time, tick_start_nimg, dist, logger0, metrics):
    """Refactored log progress block."""
    tick_end_time = time.time()
    fields = []
    fields += [f"samples {cur_nimg:<9.1f}"]
    fields += [f"training_loss {average_loss:<7.2f}"]
    fields += [f"training_loss_running_mean {average_loss_running_mean:<7.2f}"]
    fields += [f"learning_rate {current_lr:<7.8f}"]
    fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
    fields += [f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"]
    fields += [f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"]
    fields += [f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
    if metrics:
        for metric_name, metric_value in metrics.items():
            fields += [f"{metric_name} {metric_value:<7.2f}"]
    
    if torch.cuda.is_available():
        fields += [f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"]
        fields += [f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
    
    logger0.info(" ".join(fields))

def checkpoint_block(cfg, dist, logger0, checkpoint_dir, model, optimizer, cur_nimg, done):
    """Refactored checkpoint block."""
    # Save checkpoints
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
    """Refactored cleanup checkpoints block."""
    # Retain only the recent n checkpoints, if desired
    if dist.rank == 0 and cfg.training.io.save_n_recent_checkpoints > 0:
        for suffix in [".mdlus", ".pt"]:
            ckpts = checkpoint_list(checkpoint_dir, suffix=suffix)
            while len(ckpts) > cfg.training.io.save_n_recent_checkpoints:
                os.remove(os.path.join(checkpoint_dir, ckpts[0]))
                ckpts = ckpts[1:]

# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    """Main training function for CorrDiff model."""
    
    # Initialize distributed environment
    dist = initialize_distributed_environment()
    
    # Initialize loggers
    writer, logger, logger0 = initialize_loggers(dist, cfg)
    
    # Resolve configurations
    dataset_cfg, validation, validation_dataset_cfg = resolve_configurations(cfg)
    
    # Register custom dataset
    register_dataset(cfg.dataset.type)
    logger0.info(f"Using dataset: {cfg.dataset.type}")
    
    # Setup performance settings
    fp16, enable_amp, amp_dtype, songunet_checkpoint_level = setup_performance_settings(cfg)
    
    logger.info(f"Saving the outputs in {os.getcwd()}")
    
    # Setup batch size
    if cfg.training.hp.batch_size_per_gpu == "auto":
        cfg.training.hp.batch_size_per_gpu = cfg.training.hp.total_batch_size // dist.world_size
    
    # Load current number of images for resuming
    try:
        cur_nimg = load_checkpoint(path=get_checkpoint_dir(
            str(cfg.training.io.get("checkpoint_dir", ".")), cfg.model.name
        ))
    except Exception:
        cur_nimg = 0
    
    # Setup checkpoint and seeds
    checkpoint_dir = setup_checkpoint_and_seeds(dist, cfg, cur_nimg)
    
    # Initialize datasets
    dataset, dataset_iterator, validation_dataset, validation_dataset_iterator = initialize_datasets(
        cfg, dataset_cfg, validation, validation_dataset_cfg, cur_nimg, logger0
    )
    
    # Parse image configuration
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels
    
    # Configure patching
    patching, use_patching, prob_channels, img_shape = configure_patching(
        cfg, dataset, img_shape, logger0
    )
    
    # Interpolate global channel if patch-based model is used
    if use_patching:
        img_in_channels += dataset_channels
    
    # Create model
    model, use_torch_compile, use_apex_gn = create_model(
        cfg, img_in_channels, img_out_channels, img_shape, fp16, 
        songunet_checkpoint_level, prob_channels, enable_amp, dist, logger0
    )
    
    # Setup distributed data parallel
    model = setup_distributed_data_parallel(model, dist)
    
    # Setup WandB model watching
    if cfg.wandb.watch_model and dist.rank == 0:
        wandb.watch(model)
    
    # Load model checkpoint
    try:
        load_checkpoint(path=checkpoint_dir, models=model)
    except Exception:
        pass
    
    # Load regression checkpoint
    profile_mode = getattr(cfg.training.perf, "profile_mode", False)
    regression_net = load_regression_checkpoint(
        cfg, use_apex_gn, enable_amp, profile_mode, dist, logger0
    )
    
    # Compile models if enabled
    if use_torch_compile:
        model = torch.compile(model)
        if regression_net:
            regression_net = torch.compile(regression_net)
    
    # Compute gradient accumulation rounds
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    batch_size_per_gpu = cfg.training.hp.batch_size_per_gpu
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")
    
    # Calculate patch iterations
    patch_nums_iter = calculate_patch_iterations(cfg, batch_size_per_gpu, logger0)
    
    # Configure patch gradient accumulation
    use_patch_grad_acc = configure_patch_gradient_accumulation(
        cfg, patch_nums_iter, patching, logger0
    )
    
    # Create loss function
    loss_fn = create_loss_function(cfg, regression_net, prob_channels)
    
    # Create optimizer
    optimizer = create_optimizer(model, cfg)
    
    # Record start time
    start_time = time.time()
    
    # Load optimizer checkpoint
    if dist.world_size > 1:
        torch.distributed.barrier()
    try:
        load_checkpoint(
            path=checkpoint_dir,
            optimizer=optimizer,
            device=dist.device,
        )
    except Exception:
        pass
    

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################
    logger0.info("-"*80)
    logger0.info(f"Model initialized: {cfg.model.name}, on device: {dist.device}")
    logger0.info("-"*80)
    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    done = False

    # init variables to monitor running mean of average loss since last periodic
    average_loss_running_mean = 0
    n_average_loss_running_mean = 1
    start_nimg = cur_nimg
    input_dtype = torch.float32
    if enable_amp:
        input_dtype = torch.float32
    elif fp16:
        input_dtype = torch.float16

    # enable profiler:
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

                # Training iteration block
                average_loss, average_loss_running_mean, n_average_loss_running_mean, current_lr, cur_nimg, done, metrics = training_iteration_block(
                cfg, dist, writer, dataset_iterator, use_apex_gn, input_dtype,
                model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                enable_amp, amp_dtype, batch_size_per_gpu, num_accumulation_rounds,
                average_loss_running_mean, n_average_loss_running_mean,
                optimizer, cur_nimg, done, compute_metrics_flag=True
                )

                # Validation block
                validation_block(
                    cfg, dist, writer, logger0, validation_dataset_iterator, use_apex_gn,
                    model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                    enable_amp, amp_dtype, batch_size_per_gpu, cur_nimg, done, compute_metrics_flag=True
                )

                # Log progress
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
                        start_time, tick_start_time, tick_start_nimg, dist, logger0, metrics
                    )

                # Checkpoint block
                checkpoint_block(
                    cfg, dist, logger0, checkpoint_dir, model, optimizer, cur_nimg, done
                )

    # Cleanup checkpoints
    cleanup_checkpoints_block(cfg, checkpoint_dir, dist)

    # Done.
    logger0.info("Training Completed.")    


if __name__ == "__main__":
    main()