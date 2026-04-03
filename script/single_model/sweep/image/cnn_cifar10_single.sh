#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=single

# CIFAR10
DATASET_TYPE=CIFAR10
python main.py \
    --task_type $TASK_TYPE \
    --dataset_type $DATASET_TYPE \
    --model_type $MODEL_TYPE \
    --mix_status $MIX_STATUS \
    --no_structure \
    --sweep
