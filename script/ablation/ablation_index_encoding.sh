#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=single
INIT_LR=0.0


# CIFAR10
DATASET_TYPE=CIFAR10
INIT_STEPS=0
TRAIN_EPOCHS=800


SEEDS=(21962 59358 78117)



# Try raw encoding first
INDEX_ENCODING_TYPE=raw
# Use the swept best lr
TRAIN_LR=2e-4
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
        --ablation_index_encoding_type $INDEX_ENCODING_TYPE \
        --grad_clip \
        --seed $SEED
done


# Test positional encoding later
INDEX_ENCODING_TYPE=positional
INDEX_POSITIONAL_SIGMA=100.0
INDEX_POSITIONAL_N_FREQ=32
# Use the swept best lr
TRAIN_LR=2e-4
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
        --ablation_index_encoding_type $INDEX_ENCODING_TYPE \
        --ablation_index_positional_n_freqs $INDEX_POSITIONAL_N_FREQ \
        --ablation_index_positional_sigma $INDEX_POSITIONAL_SIGMA \
        --grad_clip \
        --seed $SEED
done
