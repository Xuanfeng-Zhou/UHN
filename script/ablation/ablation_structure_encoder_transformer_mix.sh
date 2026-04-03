#!/bin/bash
TASK_TYPE=TEXT_CLASSIFICATION
MODEL_TYPE=TRANSFORMER
MIX_STATUS=mixed

# AG News
DATASET_TYPE=AG_NEWS

# Fixed multi-model set testing
MULTI_MODEL_MODE=fixed
TEST_MODE=hold_out
SAMPLE_FULL_NUM=1000
SAMPLE_PRIME_NUM=100
SAMPLE_PRIME_EACH_NUM=50
MODEL_SET_SAVE_DIR=model_datasets
MODEL_SET_SEED=57055

INIT_LR=2e-4
TRAIN_LR=5e-5

INIT_STEPS=8000
TRAIN_EPOCHS=100

SEEDS=(21962 59358 78117)
# No structure encoder

for SEED in "${SEEDS[@]}"
do
    echo "Using random seed: $SEED"
    python main.py \
        --task_type $TASK_TYPE \
        --dataset_type $DATASET_TYPE \
        --model_type $MODEL_TYPE \
        --mix_status $MIX_STATUS \
        --multi_model_mode $MULTI_MODEL_MODE \
        --test_mode $TEST_MODE \
        --sample_full_num $SAMPLE_FULL_NUM \
        --sample_prime_num $SAMPLE_PRIME_NUM \
        --sample_prime_each_num $SAMPLE_PRIME_EACH_NUM \
        --model_set_save_dir $MODEL_SET_SAVE_DIR \
        --model_set_seed $MODEL_SET_SEED \
        --init_steps $INIT_STEPS \
        --train_epochs $TRAIN_EPOCHS \
        --init_lr $INIT_LR \
        --train_lr $TRAIN_LR \
        --no_structure \
        --seed $SEED
done
