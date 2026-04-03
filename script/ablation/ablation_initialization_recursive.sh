#!/bin/bash
INIT_LR=1e-4
TRAIN_LR=2e-5
TRAIN_STEPS=30000

INIT_STEPS_LIST=(
    0
    4000
)

SEEDS=(21962 59358 78117)
for INIT_STEPS in "${INIT_STEPS_LIST[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        python main_recursive.py \
            --init_steps $INIT_STEPS \
            --train_steps $TRAIN_STEPS \
            --init_lr $INIT_LR \
            --train_lr $TRAIN_LR \
            --check_weight_stats \
            --ablation \
            --seed $SEED
    done
done
