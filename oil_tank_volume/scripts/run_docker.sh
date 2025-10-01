#!/bin/bash

# Run the Docker container in interactive mode with GPU support
docker run -it --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=8g \
    -p 8889:8888 \
    -v "$(pwd)/../data:/app/data" \
    -v "$(pwd)/../outputs:/app/outputs" \
    -v "$(pwd)/scripts:/app/scripts" \
    -v "$(pwd)/notebooks:/app/notebooks" \
    oil_tank_volume:latest

#     -v "/:/app/host" \