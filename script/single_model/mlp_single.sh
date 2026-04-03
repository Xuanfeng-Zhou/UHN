#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
DATASET_TYPE=MNIST
MODEL_TYPE=MLP
MIX_STATUS=single
INIT_LR=2e-4
TRAIN_LR=2e-4

INIT_STEPS=100
TRAIN_EPOCHS=100
SEEDS=(21962 59358 78117)
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
        --seed $SEED \
        # --validate
done
