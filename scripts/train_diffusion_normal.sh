#!/bin/bash
# 0. paths
DATA_PATH="/mnt/storage/younes.abid/physicsnemo/data/custom_data_2/"
ERA5_WRF_COMBINED_CONCATENATED_DIR=$DATA_PATH"ERA5_WRF_combined_concatenated_432/"
STATS_DIR=$DATA_PATH"stats_432/"

F1=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2019-01-01_2019-06-09_150.nc"
F2=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2019-06-10_2020-01-05_150.nc"
F3=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2020-01-06_2020-07-09_150.nc"
F4=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2020-07-10_2020-12-30_150.nc"
F5=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2020-12-31_2021-06-27_150.nc"
F6=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2021-06-28_2021-12-22_150.nc"
F7=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2021-12-23_2022-06-04_150.nc"
F8=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2022-06-05_2022-11-22_150.nc"
F9=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2022-11-23_2023-05-13_150.nc"
F10=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2023-05-14_2023-10-25_150.nc"
F11=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2023-10-26_2024-04-29_150.nc"
VAL=$ERA5_WRF_COMBINED_CONCATENATED_DIR"2024-04-30_2024-05-30_21.nc"

STAT=$STATS_DIR"stat.json"

CHECKPOINT_PATH="/app/checkpoints_diffusion/"
###################################################
############# warmup round of training ############
###################################################
# 1. warm up on one file
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* 2>/dev/null | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
highest_checkpoint=${highest_checkpoint:-0}  # Default to 0 if empty
training_duration=5000
echo "Warm up on one file from $highest_checkpoint until duration: $training_duration"
if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom.py::CustomDataset" \
        dataset.data_path=$F1 \
        validation.data_path=$VAL \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="amp-fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal

else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

###################################################
############# first round of training #############
###################################################
# every sample is seen only one time
# onefile containd 150 days* 24 hours = 3600 samples
# 3 files = 10800 samples
# as we checkpoint every 5000, we take 15000
DURATION_INCREMENT=15000

# 2. Train on files 1,2,3
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F1, \
            $F2, \
            $F3 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi
# 3. Train on files 4,5,6
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F4, \
            $F5, \
            $F6 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

# 4. Train on files 7,8,9
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F7, \
            $F8, \
            $F9 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi
# 5. Train on files 10,11
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F10, \
            $F11 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

###################################################
############# Second round of training ############
###################################################
# Longer training durations each sample is seen twice
# onefile containd 150 days* 24 hours = 3600 samples
# 3 files = 10800 samples
# 10800 * 2 = 21600 (as we checkpoint every 5000, we take 25000)
DURATION_INCREMENT=25000

# 6. Train on files 1,2,3 (2nd round)
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F1, \
            $F2, \
            $F3 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

# 7. Train on files 3,4,5 (2nd round)
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F3, \
            $F4, \
            $F5 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

# 8. Train on files 5,6,7 (2nd round)
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F5, \
            $F6, \
            $F7 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

# 9. Train on files 7,8,9 (2nd round)
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F7, \
            $F8, \
            $F9 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi

# 10. Train on files 9,10,11 (2nd round)
echo "--------------------------------------"
highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
training_duration=$((training_duration + DURATION_INCREMENT))
echo "Resuming from $highest_checkpoint until duration: $training_duration"

if (( training_duration > highest_checkpoint )); then
    cd /app && torchrun --nproc_per_node=8 \
        examples/weather/corrdiff/train.py \
        hydra.run.dir=/app/outputs \
        dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
        dataset.data_path="[ \
            $F9, \
            $F10, \
            $F11 \
        ]" \
        validation.data_path="[ \
            $VAL \
        ]" \
        dataset.stats_path=$STAT \
        validation.stats_path=$STAT \
        training.hp.training_duration=$training_duration \
        training.hp.grad_clip_threshold=null \
        training.hp.lr=0.00005 \
        training.perf.fp_optimizations="fp32" \
        training.io.load_optimizer=False \
        --config-name=config_training_custom_diffusion_normal
else
    echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
fi