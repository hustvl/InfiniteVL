#!/bin/bash

# Configure your cluster variables
NPROC_PER_NODE=8   # Set to your number of GPUs
NNODES=1
RANK=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500

# Accept config path as the first argument
CONFIG_PATH=$1

torchrun \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $RANK \
    src/train.py $CONFIG_PATH