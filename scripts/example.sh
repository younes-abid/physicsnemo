#!/bin/bash

python3 examples/weather/corrdiff/train.py \
    dataset.data_path="/app/data/weather_data/hrrr_mini_train.nc" \
    dataset.stats_path="/app/data/weather_data/stats.json" \
    --config-name=config_training_hrrr_mini_regression
    #--cfg job