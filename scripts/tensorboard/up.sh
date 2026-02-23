#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/tensorboard/logdir"
    echo "Example: $0 /app/tensorboard/Fog_index/diffusion"
    exit 1
fi
tensorboard --logdir=$1 --host 0.0.0.0 --port 6006