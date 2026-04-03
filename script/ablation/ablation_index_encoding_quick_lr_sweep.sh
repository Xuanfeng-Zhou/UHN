#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=single
INIT_LR=0.0

# CIFAR10
DATASET_TYPE=CIFAR10
INIT_STEPS=0
TRAIN_EPOCHS=800


# Sweep lr
SEEDS=(21962 59358 78117)
LRS=(2e-4 1e-4 5e-5)

for TRAIN_LR in "${LRS[@]}"
do
    for SEED in "${SEEDS[@]}"
    do
        echo "Using learning rate: $TRAIN_LR"
        # Try raw encoding first
        INDEX_ENCODING_TYPE=raw
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
            --ablation_index_encoding_type $INDEX_ENCODING_TYPE \
            --seed $SEED \
            --grad_clip \
            --validate

        # Test positional encoding later
        INDEX_ENCODING_TYPE=positional
        INDEX_POSITIONAL_SIGMA=100.0
        INDEX_POSITIONAL_N_FREQ=32
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
            --ablation_index_encoding_type $INDEX_ENCODING_TYPE \
            --ablation_index_positional_n_freqs $INDEX_POSITIONAL_N_FREQ \
            --ablation_index_positional_sigma $INDEX_POSITIONAL_SIGMA \
            --seed $SEED \
            --grad_clip \
            --validate
    done
done
