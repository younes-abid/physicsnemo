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
CHECKPOINT_PATH="/app/checkpoints_regression/"

# Function to train on a set of files
train_on_files() {
    local files=("$@")  # Array of input files
    local highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* 2>/dev/null | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
    highest_checkpoint=${highest_checkpoint:-0}  # Default to 0 if empty
    training_duration=$((training_duration + DURATION_INCREMENT))
    echo "Resuming from $highest_checkpoint until duration: $training_duration"

    # Format the file paths as a comma-separated list
    local formatted_files=$(printf ', "%s"' "${files[@]}")
    formatted_files="[${formatted_files:2}]"  # Remove the leading comma and space

    if (( training_duration > highest_checkpoint )); then
        cd /app && torchrun --nproc_per_node=8 \
            examples/weather/corrdiff/train.py \
            hydra.run.dir=/app/outputs \
            dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
            dataset.data_path="$formatted_files" \
            validation.data_path="[\"$VAL\"]" \
            dataset.stats_path=$STAT \
            validation.stats_path=$STAT \
            training.hp.training_duration=$training_duration \
            training.hp.grad_clip_threshold=null \
            training.hp.lr=0.00005 \
            training.perf.fp_optimizations="fp32" \
            training.io.load_optimizer=False \
            --config-name=config_training_custom_regression_normal
    else
        echo "Training duration is lower than or equal to the highest checkpoint. Skipping training."
    fi
}

# 1. Warm-up round of training
DURATION_INCREMENT=5000
training_duration=0
echo "Warm up on one file"
train_on_files "$F1"

echo "--------------------------------------"
echo "First round of training"
echo "1 file = 150 days *24 hours =3600 steps"
echo "3 files = 450 days *24 hours =10800 steps"
echo "We set the increment to 15000 to ensure we go beyond 10800 steps"
DURATION_INCREMENT=15000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F4" "$F5" "$F6"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F10" "$F11"

echo "--------------------------------------"
echo "Second round of training"
echo "1 file = 150 days *24 hours =3600 steps"
echo "3 files = 450 days *24 hours =10800 steps"
echo "We set the increment to 25000 to ensure we go beyond 10800 steps twice"
DURATION_INCREMENT=25000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F3" "$F4" "$F5"
train_on_files "$F5" "$F6" "$F7"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F9" "$F10" "$F11"

echo "--------------------------------------"
echo "Third round of training"
echo "1 file = 150 days *24 hours =3600 steps"
echo "3 files = 450 days *24 hours =10800 steps"
echo "We set the increment to 35000 to ensure we go beyond 10800 steps thrice"
DURATION_INCREMENT=35000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F3" "$F4" "$F5"
train_on_files "$F5" "$F6" "$F7"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F9" "$F10" "$F11"

echo "--------------------------------------"
echo "Forth round of training"
echo "1 file = 150 days *24 hours =3600 steps"
echo "3 files = 450 days *24 hours =10800 steps"
echo "We set the increment to 45000 to ensure we go beyond 10800 steps four times"
DURATION_INCREMENT=45000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F3" "$F4" "$F5"
train_on_files "$F5" "$F6" "$F7"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F9" "$F10" "$F11"