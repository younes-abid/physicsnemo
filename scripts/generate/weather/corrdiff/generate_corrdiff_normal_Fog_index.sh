#!/bin/bash
# Generation script for CorrDiff Fog_index model
# This script generates predictions using trained regression and diffusion models

# 0. Configuration paths
DATA_PATH="/mnt/storage/younes.abid/physicsnemo/data/custom_data_2/"
ERA5_WRF_COMBINED_CONCATENATED_DIR=$DATA_PATH"ERA5_WRF_combined_concatenated_432/"
STATS_DIR=$DATA_PATH"stats_432/"

# Model configuration
VARIABLES="Fog_index"
MODEL_TYPE="corrdiff"  # Combined regression + diffusion

# Data file to use for generation (easily changeable)
# Current default: last available data file
DATA_FILE="${ERA5_WRF_COMBINED_CONCATENATED_DIR}2024-04-30_2024-05-30_21.nc"

# Alternative data files (uncomment to use different data):
# DATA_FILE="${ERA5_WRF_COMBINED_CONCATENATED_DIR}2023-10-26_2024-04-29_150.nc"
# DATA_FILE="${ERA5_WRF_COMBINED_CONCATENATED_DIR}2023-05-14_2023-10-25_150.nc"

# Checkpoint paths (automatically finds the latest checkpoints)
REGRESSION_CKPT_DIR="/app/checkpoints/${VARIABLES}/checkpoints_regression/"
DIFFUSION_CKPT_DIR="/app/checkpoints/${VARIABLES}/checkpoints_diffusion/"

# Output configuration
OUTPUT_DIR="/app/outputs/generation/${VARIABLES}/"
LOGS_DIR="/app/logs/${VARIABLES}/generation/"

# Statistics file
STAT_FILE=$STATS_DIR"stat.json"

# Configuration file
CONFIG_NAME="config_generate_custom_${VARIABLES}"

# Create output and logs directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_DIR

# Create timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOGS_DIR}generation_${TIMESTAMP}.log"

# Log function
log_message() {
    local message="$1"
    echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" | tee -a "$LOG_FILE"
}

# Function to find latest checkpoint
find_latest_checkpoint() {
    local checkpoint_dir="$1"
    local pattern="$2"
    local latest_checkpoint=$(ls ${checkpoint_dir}${pattern}*.mdlus 2>/dev/null | sort -t '.' -k 3 -n | tail -n 1)
    echo "$latest_checkpoint"
}

# Find latest checkpoints
REGRESSION_CHECKPOINT=$(find_latest_checkpoint "$REGRESSION_CKPT_DIR" "UNet")
DIFFUSION_CHECKPOINT=$(find_latest_checkpoint "$DIFFUSION_CKPT_DIR" "EDMPrecondSuperResolution")

# Validate checkpoints exist
if [ -z "$REGRESSION_CHECKPOINT" ]; then
    log_message "ERROR: No regression checkpoint found in $REGRESSION_CKPT_DIR"
    exit 1
fi

if [ -z "$DIFFUSION_CHECKPOINT" ]; then
    log_message "ERROR: No diffusion checkpoint found in $DIFFUSION_CKPT_DIR"
    exit 1
fi

# Validate data file exists
if [ ! -f "$DATA_FILE" ]; then
    log_message "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Validate stats file exists
if [ ! -f "$STAT_FILE" ]; then
    log_message "ERROR: Statistics file not found: $STAT_FILE"
    exit 1
fi

# Output filename with timestamp
OUTPUT_FILE="${OUTPUT_DIR}corrdiff_${VARIABLES}_predictions_${TIMESTAMP}.nc"

log_message "========================================"
log_message "CorrDiff Generation Script"
log_message "========================================"
log_message "Variables: $VARIABLES"
log_message "Data file: $DATA_FILE"
log_message "Regression checkpoint: $REGRESSION_CHECKPOINT"
log_message "Diffusion checkpoint: $DIFFUSION_CHECKPOINT"
log_message "Statistics file: $STAT_FILE"
log_message "Output file: $OUTPUT_FILE"
log_message "Config: $CONFIG_NAME"
log_message "========================================"

# Run generation
log_message "Starting generation process..."

cd /app && torchrun --nproc_per_node=8 \
    examples/weather/corrdiff/generate.py \
    hydra.run.dir=/app/outputs \
    dataset.type="/app/examples/weather/corrdiff/datasets/custom_list_2.py::CustomDataset" \
    dataset.data_path="[\"$DATA_FILE\"]" \
    dataset.stats_path="$STAT_FILE" \
    generation.io.res_ckpt_filename="$DIFFUSION_CHECKPOINT" \
    generation.io.reg_ckpt_filename="$REGRESSION_CHECKPOINT" \
    generation.io.output_filename="$OUTPUT_FILE" \
    --config-name="$CONFIG_NAME" \
    >> "$LOG_FILE" 2>&1

# Check exit code
exit_code=$?
if [ $exit_code -eq 0 ]; then
    log_message "Generation completed successfully!"
    log_message "Output saved to: $OUTPUT_FILE"
    log_message "Log file: $LOG_FILE"
else
    log_message "Generation failed with exit code: $exit_code"
    log_message "Check log file for details: $LOG_FILE"
fi

log_message "========================================"
log_message "Generation script finished"
log_message "========================================"

exit $exit_code