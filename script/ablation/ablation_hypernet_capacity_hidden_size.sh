#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=single
INIT_LR=0.0
TRAIN_LR=2e-4

# CIFAR10
DATASET_TYPE=CIFAR10
INIT_STEPS=0
TRAIN_EPOCHS=800

HIDDEN_SIZES_LIST=(
    32
    64
    # 128
    256
)


SEEDS=(21962 59358 78117)
for HIDDEN_SIZE in "${HIDDEN_SIZES_LIST[@]}"
do
    echo "Running ablation study with HIDDEN_SIZE=$HIDDEN_SIZE"
    for SEED in "${SEEDS[@]}"
    do
        echo "Using random seed: $SEED"
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
            --ablation_hidden_size $HIDDEN_SIZE \
            --seed $SEED
    done
done
