#!/bin/bash
TASK_TYPE=TEXT_CLASSIFICATION
MODEL_TYPE=TRANSFORMER
MIX_STATUS=single
TRAIN_EPOCHS=100
INIT_LR=5e-5
TRAIN_LR=1e-4

# IMDB
DATASET_TYPE=IMDB

INIT_STEPS_LIST=(
    0
    100
)

SEEDS=(21962 59358 78117)
for INIT_STEPS in "${INIT_STEPS_LIST[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        python main.py \
            --task_type $TASK_TYPE \
            --dataset_type $DATASET_TYPE \
            --model_type $MODEL_TYPE \
            --mix_status $MIX_STATUS \
            --init_steps $INIT_STEPS \
            --train_epochs $TRAIN_EPOCHS \
            --init_lr $INIT_LR \
            --train_lr $TRAIN_LR \
            --no_structure \
            --ablation \
            --check_weight_stats \
            --seed $SEED
    done
done
