"""Metrics computation, image preparation, and TensorBoard/terminal logging."""

import os
import time
import psutil
import torch


def compute_metrics(predictions: torch.Tensor, targets: torch.Tensor, prefix: str = "",
                    variables: list = []) -> dict:
    """
    Compute various metrics between predictions and targets.

    Args:
        predictions: Predicted values of shape (B, C, H, W).
        targets: Ground truth values of shape (B, C, H, W).
        prefix: Prefix for metric names (e.g., "train_", "val_").
        variables: List of variable names corresponding to the channels.

    Returns:
        Dictionary containing computed metrics.
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

        # Compute mean and std errors
        pred_mean = torch.mean(p)
        target_mean = torch.mean(t)
        pred_std = torch.std(p)
        target_std = torch.std(t)

        metrics[f'{prefix}mean_error_{var}'] = torch.abs(pred_mean - target_mean)
        metrics[f'{prefix}std_error_{var}'] = torch.abs(pred_std - target_std)
        metrics[f'{prefix}relative_mean_error_{var}'] = torch.abs(pred_mean - target_mean) / (torch.abs(target_mean) + 1e-8)
        metrics[f'{prefix}relative_std_error_{var}'] = torch.abs(pred_std - target_std) / (torch.abs(target_std) + 1e-8)

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
            threshold_str = f"{threshold.item():.1f}"
            metrics[f'{prefix}relative_error_within_{threshold_str}_{var}'] = torch.mean(within_threshold[..., j])

    return metrics


def prepare_images(predictions_cat: torch.Tensor, targets_cat: torch.Tensor, prefix: str,
                   variables: list, n: int = 1) -> dict:
    """
    Prepare images for logging to TensorBoard.

    Args:
        predictions_cat: Concatenated predicted values of shape (B, C, H, W).
        targets_cat: Concatenated ground truth values of shape (B, C, H, W).
        prefix: Prefix for logging (e.g., "training_", "validation_").
        variables: List of variable names corresponding to the channels.
        n: Number of images to prepare (default: 1).

    Returns:
        Dictionary containing titles as keys and lists of images as values.
    """
    images = {}
    predictions = predictions_cat[:n]
    targets = targets_cat[:n]

    for i, var in enumerate(variables):
        pred = predictions[:, i, :, :].unsqueeze(1)
        target = targets[:, i, :, :].unsqueeze(1)
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

    final_metrics = {}
    for metric_name, metric_values in aggregated_metrics.items():
        if metric_values:
            metric_tensor = torch.tensor(metric_values, device=dist.device)
            if dist.world_size > 1:
                torch.distributed.all_reduce(metric_tensor, op=torch.distributed.ReduceOp.SUM)
                metric_tensor /= dist.world_size
            final_metrics[metric_name] = metric_tensor.mean().item()

    return final_metrics


def aggregate_loss(loss_accum, dist):
    """Aggregate loss across all distributed processes."""
    import nvtx
    with nvtx.annotate("loss aggregate", color="green"):
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
    """Log training metrics to TensorBoard."""
    if dist.rank == 0:
        writer.add_scalar("training_loss", average_loss, cur_nimg)
        writer.add_scalar("training_loss_running_mean", average_loss_running_mean, cur_nimg)

        if metrics:
            for metric_name, metric_value in metrics.items():
                writer.add_scalar(metric_name, metric_value, cur_nimg)


def log_images(writer, cur_nimg, dist, images):
    """Log images to TensorBoard with clear separation between predictions and targets."""
    if dist.rank == 0:
        for title, image_list in images.items():
            pred, target = image_list
            separator = torch.zeros_like(pred)
            separator_width = max(1, pred.shape[3] // 100)
            separator = separator[:, :, :, :separator_width]

            pred_vs_target = torch.cat([pred, separator, target, separator, pred - target], dim=3)
            writer.add_image(title, pred_vs_target[0], global_step=cur_nimg)


def log_progress_block(cur_nimg, average_loss, average_loss_running_mean, current_lr,
                       start_time, tick_start_time, tick_start_nimg, dist, logger0, metrics):
    """Log training progress to terminal."""
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
