"""Setup utilities: distributed env, loggers, config, datasets, model, DDP."""

import os
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import wandb

from physicsnemo import Module
from physicsnemo.models.diffusion import UNet, EDMPrecondSuperResolution
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.patching import RandomPatching2D
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from datasets.dataset import init_train_valid_datasets_from_config, register_dataset
from helpers.train_helpers import set_patch_shape


def initialize_distributed_environment():
    """Initialize distributed environment for training."""
    DistributedManager.initialize()
    dist = DistributedManager()
    return dist


def initialize_loggers(dist, cfg):
    """Initialize all loggers including TensorBoard, Python logger, and WandB."""
    writer = None
    if dist.rank == 0:
        writer = SummaryWriter(log_dir=cfg.tensorboard.log_dir)

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


def resolve_configurations(cfg):
    """Resolve and parse configuration settings."""
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)

    validation = hasattr(cfg, "validation")
    validation_dataset_cfg = OmegaConf.to_container(cfg.validation) if validation else None

    return dataset_cfg, validation, validation_dataset_cfg


def setup_performance_settings(cfg):
    """Configure performance-related settings like FP optimizations."""
    fp_optimizations = cfg.training.perf.fp_optimizations
    songunet_checkpoint_level = cfg.training.perf.songunet_checkpoint_level
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    return fp16, enable_amp, amp_dtype, songunet_checkpoint_level


def initialize_datasets(cfg, dataset_cfg, validation, validation_dataset_cfg, cur_nimg, logger0):
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


def configure_patching(cfg, dataset, img_shape, logger0):
    """Configure patch-based training settings."""
    if cfg.model.name == "lt_aware_ce_regression":
        prob_channels = dataset.get_prob_channel_index()
    else:
        prob_channels = None

    if cfg.model.name in ["patched_diffusion", "lt_aware_patched_diffusion"]:
        patch_shape_x = cfg.training.hp.patch_shape_x
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_x = None
        patch_shape_y = None

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
                 songunet_checkpoint_level, prob_channels, enable_amp, dist, logger0):
    """Create and configure the model based on configuration."""
    model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
        "checkpoint_level": songunet_checkpoint_level,
    }

    if cfg.model.name == "lt_aware_ce_regression":
        model_args["prob_channels"] = prob_channels

    if hasattr(cfg.model, "model_args"):
        model_args.update(OmegaConf.to_container(cfg.model.model_args))

    use_torch_compile = getattr(cfg.training.perf, "torch_compile", False)
    use_apex_gn = getattr(cfg.training.perf, "use_apex_gn", False)
    profile_mode = getattr(cfg.training.perf, "profile_mode", False)

    if use_apex_gn:
        model_args["use_apex_gn"] = use_apex_gn
    if profile_mode:
        model_args["profile_mode"] = profile_mode
    if enable_amp:
        model_args["amp_mode"] = enable_amp

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

    model.train().requires_grad_(True).to(dist.device)

    if use_apex_gn:
        model.to(memory_format=torch.channels_last)

    if (cfg.model.name in ["regression", "lt_aware_regression", "lt_aware_ce_regression"] and
        getattr(cfg.training.hp, "patch_shape_x", None) is not None):
        raise ValueError(
            f"Regression model ({cfg.model.name}) cannot be used with patch-based training."
        )

    return model, use_torch_compile, use_apex_gn


def setup_distributed_data_parallel(model, dist):
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
