#!/bin/bash

# Run the inference script for correlation and diffusion
python3 examples/weather/corrdiff/train.py \
    dataset.data_path="/app/data/weather data/hrrr_mini_train.nc" \
    --config-name=config_training_gefs_hrrr_regression