#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=single

SEEDS=(21962 59358 78117)
for SEED in "${SEEDS[@]}"
do
    # MNIST
    DATASET_TYPE=MNIST
    INIT_LR=2e-4
    TRAIN_LR=1e-4
    INIT_STEPS=200
    TRAIN_EPOCHS=100
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


    # CIFAR10
    DATASET_TYPE=CIFAR10
    INIT_LR=0
    TRAIN_LR=2e-4
    INIT_STEPS=0
    TRAIN_EPOCHS=800
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
