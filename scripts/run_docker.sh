#!/bin/bash

# Run the Docker container in interactive mode with GPU support
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=8g \
    -p 8888:8888 \
    -p 6006:6006 \
    -v "$(pwd)/examples:/app/examples" \
    -v "$(pwd)/physicsnemo:/app/physicsnemo" \
    -v "$(pwd)/data:/app/data" \
    -v "/mnt/storage:/mnt/storage" \
    -v "$(pwd)/scripts:/app/scripts" \
    -v "$(pwd)/notebooks:/app/notebooks" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/outputs/checkpoints_regression:/app/checkpoints_regression" \
    -v "$(pwd)/outputs/checkpoints_diffusion:/app/checkpoints_diffusion" \
    -v "$(pwd)/outputs/wandb:/app/wandb" \
    -v "$(pwd)/outputs/tensorboard:/app/tensorboard" \
    physiscsnemo:latest

#     -v "/:/app/host" \