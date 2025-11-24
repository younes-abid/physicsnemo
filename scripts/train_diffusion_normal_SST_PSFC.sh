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

MODEL="diffusion" # can be regression or diffusion
VARIABLES="SST_PSFC" # can be on of these options ["U10_V10", "T2_TSK", "Q2_rain_rate", "SST_PSFC"]

STAT=$STATS_DIR"stat.json"
CONFIG_NAME="config_training_custom_${MODEL}_normal_${VARIABLES}"
CHECKPOINT_PATH="/app/checkpoints/${VARIABLES}/checkpoints_${MODEL}/"
LOGS_PATH="/app/logs/${VARIABLES}/${MODEL}/"

# Create logs directory if it doesn't exist
mkdir -p $LOGS_PATH

# Create a timestamped main log file for this training session
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOGS_PATH}training_${TIMESTAMP}.log"

# Log function to write to both console and log file
log_message() {
    local message="$1"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" | tee -a "$MAIN_LOG"
}

# Function to train on a set of files
train_on_files() {
    local files=("$@")  # Array of input files
    local highest_checkpoint=$(ls $CHECKPOINT_PATH*checkpoint.0* 2>/dev/null | sort -t '.' -k 3 -n | tail -n 1 | awk -F '.' '{print $3}')
    highest_checkpoint=${highest_checkpoint:-0}  # Default to 0 if empty
    training_duration=$((training_duration + DURATION_INCREMENT))
    
    # Create a log file for this specific training run
    local train_log="${LOGS_PATH}train_duration_${training_duration}_${TIMESTAMP}.log"
    
    log_message "Resuming from $highest_checkpoint until duration: $training_duration"
    echo "Files: ${files[@]}" >> "$MAIN_LOG"

    # Format the file paths as a comma-separated list
    local formatted_files=$(printf ', "%s"' "${files[@]}")
    formatted_files="[${formatted_files:2}]"  # Remove the leading comma and space

    if (( training_duration > highest_checkpoint )); then
        log_message "Starting training... (logs: $train_log)"
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
            --config-name=$CONFIG_NAME \
            >> "$train_log" 2>&1
        
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            log_message "Training completed successfully (duration: $training_duration)"
        else
            log_message "Training failed with exit code $exit_code (duration: $training_duration)"
        fi
    else
        log_message "Training duration is lower than or equal to the highest checkpoint. Skipping training."
    fi
}

# 1. Warm-up round of training
DURATION_INCREMENT=5000
training_duration=0
log_message "========================================"
log_message "Warm up on one file"
log_message "========================================"
train_on_files "$F1"

log_message "========================================"
log_message "First round of training"
log_message "1 file = 150 days *24 hours =3600 steps"
log_message "3 files = 450 days *24 hours =10800 steps"
log_message "We set the increment to 15000 to ensure we go beyond 10800 steps"
log_message "========================================"
DURATION_INCREMENT=15000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F4" "$F5" "$F6"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F10" "$F11"

log_message "========================================"
log_message "Second round of training"
log_message "1 file = 150 days *24 hours =3600 steps"
log_message "3 files = 450 days *24 hours =10800 steps"
log_message "We set the increment to 25000 to ensure we go beyond 10800 steps twice"
log_message "========================================"
DURATION_INCREMENT=25000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F3" "$F4" "$F5"
train_on_files "$F5" "$F6" "$F7"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F9" "$F10" "$F11"

log_message "========================================"
log_message "Third round of training"
log_message "1 file = 150 days *24 hours =3600 steps"
log_message "3 files = 450 days *24 hours =10800 steps"
log_message "We set the increment to 35000 to ensure we go beyond 10800 steps thrice"
log_message "========================================"
DURATION_INCREMENT=35000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F3" "$F4" "$F5"
train_on_files "$F5" "$F6" "$F7"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F9" "$F10" "$F11"

log_message "========================================"
log_message "Fourth round of training"
log_message "1 file = 150 days *24 hours =3600 steps"
log_message "3 files = 450 days *24 hours =10800 steps"
log_message "We set the increment to 45000 to ensure we go beyond 10800 steps four times"
log_message "========================================"
DURATION_INCREMENT=45000
train_on_files "$F1" "$F2" "$F3"
train_on_files "$F3" "$F4" "$F5"
train_on_files "$F5" "$F6" "$F7"
train_on_files "$F7" "$F8" "$F9"
train_on_files "$F9" "$F10" "$F11"

log_message "========================================"
log_message "All training rounds completed!"
log_message "Main log file: $MAIN_LOG"
log_message "========================================"