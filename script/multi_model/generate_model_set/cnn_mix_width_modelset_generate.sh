#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=mixed
DATASET_TYPE=CIFAR10

SAMPLE_FULL_NUM=500
MAX_SAMPLE_TRIALS=50000
SAMPLE_PRIME_NUM=100
SAMPLE_PRIME_EACH_NUM=50
MODEL_SET_SAVE_DIR=model_datasets


# Mixing width
MIX_TYPE=width
python generate_model_dataset.py \
    --task_type $TASK_TYPE \
    --dataset_type $DATASET_TYPE \
    --model_type $MODEL_TYPE \
    --mix_status $MIX_STATUS \
    --mix_type $MIX_TYPE \
    --sample_full_num $SAMPLE_FULL_NUM \
    --max_sample_trials $MAX_SAMPLE_TRIALS \
    --sample_prime_num $SAMPLE_PRIME_NUM \
    --sample_prime_each_num $SAMPLE_PRIME_EACH_NUM \
    --model_set_save_dir $MODEL_SET_SAVE_DIR \
    --seed 362
