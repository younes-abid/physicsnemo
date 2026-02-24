"""Validation block: periodic validation loss and metrics computation."""

import torch
import nvtx

from helpers.train_helpers import is_time_for_periodic_task
from metrics_logging import compute_metrics, prepare_images, log_images


def validation_block(cfg, dist, writer, logger0, validation_dataset_iterator, use_apex_gn,
                     model, loss_fn, use_patch_grad_acc, patching, patch_nums_iter,
                     enable_amp, amp_dtype, batch_size_per_gpu, cur_nimg, done, compute_metrics_flag=True):
    """Validation block with metrics computation and logging."""
    with nvtx.annotate("validation", color="red"):
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
