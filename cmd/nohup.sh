#!/bin/bash
# ----------------------------------------------------------------
# nohup run/nohup.sh &
# ----------------------------------------------------------------
DATASET_DIRS=/mnt/sda/dataset
DATA_DIRS=/mnt/sda/data

EXP_NAME="pilot"
DOCKERFILE_NAME="Dockerfile.cu117"
TORCH_VERSION="torch-2.0.1"
# cu111 - torch-1.9.0
# cu117 - torch-1.13.0, torch-2.0.0, torch-2.0.1


if [[ $1 == "train" ]]; then
    bash docker.sh exec "cmd/train.sh $2"
else
    echo "usage: bash nohup.sh [train|inference] [config]"
fi
