#!/bin/bash



#These variables are automatically set by torchrun, so you don't need to set them manually if using torchrun.
# export MASTER_ADDR=localhost
# export MASTER_PORT=29500
# export WORLD_SIZE=8  # Total number of GPUs
# export RANK=0        # Rank of the current process

export OMP_NUM_THREADS=12  # Allocate 12 threads per GPU (we have 8 GPUs, and 96 CPUs total)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# python3 examples/weather/corrdiff/train.py \
#     dataset.data_path="/app/data/weather_data/hrrr_mini_train.nc" \
#     dataset.stats_path="/app/data/weather_data/stats.json" \
#     --config-name=config_training_hrrr_mini_regression
#     #--cfg job

torchrun --nproc_per_node=8 \
    examples/weather/corrdiff/train.py \
    hydra.run.dir=/app/outputs \
    dataset.data_path="/app/data/weather_data/hrrr_mini_train.nc" \
    dataset.stats_path="/app/data/weather_data/stats.json" \
    --config-name=config_training_hrrr_mini_regression