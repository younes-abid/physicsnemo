#!/bin/bash
# 0. paths
DATA_PATH="/mnt/storage/younes.abid/physicsnemo/data/sar2height/"
FILTERED_PATCHES_DIR=$DATA_PATH"processed/filtered_patches/"
STATS_DIR=$DATA_PATH"processed/filtered_patches/stats_432/"

# File 1: X2 - exists ✓
F1="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X2_SLC_SLH_56495_20210517T180931_1054patches.nc"

# File 2: X8 - exists ✓
F2="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X8_SLC_SLH_54750_20210503T175109_876patches.nc"

# File 3: X4_46812 - exists ✓
F3="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X4_SLC_SLH_46812_20210311T110553_876patches.nc"

# File 4: X4_45265 - exists ✓
F4="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X4_SLC_SLH_45265_20210302T235532_1170patches.nc"

# File 5: X7_41464 - exists ✓
F5="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X7_SLC_SLH_41464_20210217T220314_1120patches.nc"

# File 6: X7_46269 - DOES NOT EXIST as filtered file (skipping)
# File 7: X7_58884 - exists ✓
F6="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X7_SLC_SLH_58884_20210531T214045_1178patches.nc"

# File 8: X7_41463 - DOES NOT EXIST as filtered file (skipping)
# File 9: X7_123088 - exists ✓
F7="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X7_SLC_SLH_123088_20210825T214159_1273patches.nc"

# File 10: X7_123096 - DOES NOT EXIST as filtered file (skipping)
# File 11: X7_118443 - DOES NOT EXIST as filtered file (skipping)
# File 12: X7_119466 - DOES NOT EXIST as filtered file (skipping)
# File 13: X7_122288 - DOES NOT EXIST as filtered file (skipping)
# File 14: X7_148224 - exists ✓
VAL="${FILTERED_PATCHES_DIR}filtered_patches_ICEYE_X7_SLC_SLH_148224_20211013T214143_1330patches.nc"

MODEL="regression" # can be regression or diffusion
VARIABLES="intensity_db-intensity_percentile_rescaled" 

STAT=$STATS_DIR"stat.json"
CONFIG_NAME="config_training_custom_${MODEL}_normal_${VARIABLES}"
CHECKPOINT_PATH="/app/checkpoints/sar2height/${VARIABLES}/checkpoints_${MODEL}/"
LOGS_PATH="/app/logs/sar2height/${VARIABLES}/${MODEL}/"

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