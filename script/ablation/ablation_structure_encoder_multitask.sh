#!/bin/bash
INIT_LR=1e-4
INIT_STEPS=500
TRAIN_LR=2e-5
TRAIN_STEPS=200000

SEEDS=(21962 59358 78117)
for SEED in ${SEEDS[@]}; do
    echo "Running with seed: $SEED"
    python main_multi.py \
        --init_steps $INIT_STEPS \
        --train_steps $TRAIN_STEPS \
        --init_lr $INIT_LR \
        --train_lr $TRAIN_LR \
        --no_structure \
        --seed $SEED
done
