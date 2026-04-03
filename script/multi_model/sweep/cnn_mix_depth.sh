#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=mixed

# CIFAR10
DATASET_TYPE=CIFAR10

# Fixed multi-model set testing
MIX_TYPE=depth
MULTI_MODEL_MODE=fixed
TEST_MODE=hold_out
SAMPLE_FULL_NUM=100
SAMPLE_PRIME_NUM=40
SAMPLE_PRIME_EACH_NUM=20
MODEL_SET_SAVE_DIR=model_datasets
MODEL_SET_SEED=75381

python main.py \
    --task_type $TASK_TYPE \
    --dataset_type $DATASET_TYPE \
    --model_type $MODEL_TYPE \
    --mix_status $MIX_STATUS \
    --mix_type $MIX_TYPE \
    --multi_model_mode $MULTI_MODEL_MODE \
    --test_mode $TEST_MODE \
    --sample_full_num $SAMPLE_FULL_NUM \
    --sample_prime_num $SAMPLE_PRIME_NUM \
    --sample_prime_each_num $SAMPLE_PRIME_EACH_NUM \
    --model_set_save_dir $MODEL_SET_SAVE_DIR \
    --model_set_seed $MODEL_SET_SEED \
    --sweep
