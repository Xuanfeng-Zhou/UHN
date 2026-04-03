#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
DATASET_TYPE=MNIST
MODEL_TYPE=MLP
MIX_STATUS=single

python main.py \
    --task_type $TASK_TYPE \
    --dataset_type $DATASET_TYPE \
    --model_type $MODEL_TYPE \
    --mix_status $MIX_STATUS \
    --no_structure \
    --sweep
