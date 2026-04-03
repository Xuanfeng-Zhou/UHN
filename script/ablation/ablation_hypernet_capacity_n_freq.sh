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

N_FREQS_LIST=(
    4096
    256
    512
    1024
    # 2048
)


SEEDS=(21962 59358 78117)
for N_FREQS in "${N_FREQS_LIST[@]}"
do
    echo "Running ablation study with N_FREQS=$N_FREQS"
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
            --ablation_index_fourier_n_freqs $N_FREQS \
            --seed $SEED
    done
done
