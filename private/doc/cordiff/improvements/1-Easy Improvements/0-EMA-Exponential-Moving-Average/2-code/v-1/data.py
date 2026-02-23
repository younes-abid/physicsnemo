"""Data loading and loss kwargs preparation for training iterations."""

import torch
import nvtx


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

    with nvtx.annotate("loss forward", color="green"):
        with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
            if return_predictions:
                loss, predictions = loss_fn(**loss_fn_kwargs, return_predictions=True)
            else:
                loss = loss_fn(**loss_fn_kwargs)
                predictions = None

    loss = loss.sum() / batch_size_per_gpu
    return loss, predictions
