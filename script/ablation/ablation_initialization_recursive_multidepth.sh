#!/bin/bash
INIT_LR=1e-4
TRAIN_LR=2e-5
TRAIN_STEPS=30000
# Init steps when depth = 1
BASE_INIT_STEPS_LIST=(
    0
    4000
)

DEPTHS=(2 3)
SEEDS=(21962 59358 78117)
for DEPTH in "${DEPTHS[@]}"; do
    for BASE_INIT_STEPS in "${BASE_INIT_STEPS_LIST[@]}"
    do
        INIT_STEPS=$((BASE_INIT_STEPS / 2 * (DEPTH + 1)))
        echo "Running recursive task with depth: $DEPTH and init steps: $INIT_STEPS"
        for SEED in "${SEEDS[@]}"
        do
            python main_recursive.py \
                --init_steps $INIT_STEPS \
                --train_steps $TRAIN_STEPS \
                --init_lr $INIT_LR \
                --train_lr $TRAIN_LR \
                --check_weight_stats \
                --ablation \
                --recursive_depth $DEPTH \
                --seed $SEED
        done
    done
done
