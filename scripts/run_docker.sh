#!/bin/bash

# Run the Docker container in interactive mode with GPU support
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=8g \
    -p 8888:8888 \
    -v "$(pwd)/examples:/app/examples" \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/scripts:/app/scripts" \
    -v "$(pwd)/notebooks:/app/notebooks" \
    -v "$(pwd)/outputs:/app/outputs" \
    -v "$(pwd)/outputs/checkpoints_regression:/app/checkpoints_regression" \
    physiscsnemo:latest

#     -v "/:/app/host" \