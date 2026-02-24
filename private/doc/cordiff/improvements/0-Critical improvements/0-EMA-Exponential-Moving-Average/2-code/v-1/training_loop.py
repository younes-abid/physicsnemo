"""Training iteration: gradient accumulation, weight update, and per-step orchestration."""

import torch
import nvtx

from helpers.train_helpers import handle_and_clip_gradients, is_time_for_periodic_task

from data import load_and_prepare_batch, prepare_loss_kwargs, compute_loss_and_predictions
from metrics_logging import (
    compute_metrics,
    prepare_images,
    aggregate_loss,
    update_learning_rates,
    update_running_loss,
    log_training_metrics,
    log_images,
)


def accumulate_gradients_and_metrics(model, loss_fn, dataset_iterator, use_apex_gn, dist, input_dtype,
                                     use_patch_grad_acc, patching, patch_nums_iter, enable_amp, amp_dtype,
                                     batch_size_per_gpu, num_accumulation_rounds, cfg, cur_nimg,
                                     compute_metrics_flag=True):
    """Accumulate gradients and compute metrics over multiple rounds."""
    loss_accum = 0
    all_predictions = []
    all_targets = []

    for n_i in range(num_accumulation_rounds):
        with nvtx.annotate(f"accumulation round {n_i}", color="Magenta"):
            img_clean, img_lr, lead_time_label = load_and_prepare_batch(
                dataset_iterator, use_apex_gn, dist, input_dtype
            )

            loss_fn_kwargs, _ = prepare_loss_kwargs(
                model, img_clean, img_lr, lead_time_label, use_patch_grad_acc, dist, loss_fn
            )

            for patch_num_per_iter in patch_nums_iter:
                loss, predictions = compute_loss_and_predictions(
                    loss_fn, loss_fn_kwargs, patching, patch_num_per_iter,
                    enable_amp, amp_dtype, batch_size_per_gpu, return_predictions=compute_metrics_flag
                )

                loss_accum += loss / num_accumulation_rounds / len(patch_nums_iter)

                if compute_metrics_flag and predictions is not None:
                    all_predictions.append(predictions.detach())
                    all_targets.append(img_clean.detach())

                with nvtx.annotate("loss backward", color="yellow"):
                    loss.backward()

    # Compute metrics if requested
    metrics = {}
    images = {}
    if compute_metrics_flag and all_predictions:
        with torch.no_grad():
            predictions_cat = torch.cat(all_predictions)
            targets_cat = torch.cat(all_targets)
            if cur_nimg % cfg.training.io.metric_log_freq == 0:
                metrics = compute_metrics(predictions_cat, targets_cat, prefix="training_",
                                          variables=cfg.dataset.output_variables)
            if cur_nimg % cfg.training.io.image_log_freq == 0:
                images = prepare_images(predictions_cat=predictions_cat,
                                        targets_cat=targets_cat,
                                        prefix="training_",
                                        variables=cfg.dataset.output_variables,
                                        n=1)

    return loss_accum, metrics, images


def training_iteration_block(cfg, dist, writer, dataset_iterator, use_apex_gn, input_dtype,
                             model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                             enable_amp, amp_dtype, batch_size_per_gpu, num_accumulation_rounds,
                             average_loss_running_mean, n_average_loss_running_mean,
                             optimizer, cur_nimg, done, compute_metrics_flag=True,
                             ema_model=None, ema_decay=0.9999):
    """Training iteration block with gradient accumulation, weight update, and EMA."""
    with nvtx.annotate("Training iteration", color="green"):
        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Accumulate gradients and compute metrics
        loss_accum, metrics, images = accumulate_gradients_and_metrics(
            model, loss_fn, dataset_iterator, use_apex_gn, dist, input_dtype,
            use_patch_grad_acc, patching, patch_nums_iter, enable_amp, amp_dtype,
            batch_size_per_gpu, num_accumulation_rounds, cfg, cur_nimg, compute_metrics_flag,
        )

        # Aggregate loss across processes
        average_loss = aggregate_loss(loss_accum, dist)

        # Update running loss statistics
        average_loss_running_mean, n_average_loss_running_mean = update_running_loss(
            average_loss, average_loss_running_mean, n_average_loss_running_mean
        )

        # Log metrics
        log_training_metrics(writer, average_loss, average_loss_running_mean, cur_nimg, dist, metrics)

        # Log images
        if cur_nimg % cfg.training.io.image_log_freq == 0 and images:
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

            # Log gradient statistics
            if dist.rank == 0 and cur_nimg % cfg.training.io.grad_log_freq == 0:
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm(2).item()
                        total_grad_norm += grad_norm ** 2
                        writer.add_scalar(f"grad_norm/{name}", grad_norm, cur_nimg)
                total_grad_norm = total_grad_norm ** 0.5
                writer.add_scalar("grad_norm/total", total_grad_norm, cur_nimg)

        with nvtx.annotate("optimizer step", color="blue"):
            optimizer.step()

        # ── EMA update (after optimizer step) ──
        if ema_model is not None:
            from ema import update_ema
            with nvtx.annotate("ema update", color="orange"):
                update_ema(ema_model, model, decay=ema_decay)

        # Update iteration counters
        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration

    return average_loss, average_loss_running_mean, n_average_loss_running_mean, current_lr, cur_nimg, done, metrics
