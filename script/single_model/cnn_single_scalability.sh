#!/bin/bash
TASK_TYPE=IMAGE_CLASSIFICATION
MODEL_TYPE=CNN
MIX_STATUS=single
DATASET_TYPE=CIFAR10

INIT_LR=0.0
TRAIN_LR=2e-4
INIT_STEPS=0
TRAIN_EPOCHS=800

SEEDS=(21962 59358 78117)

# --------------------- ResNet-32 on CIFAR-10 ---------------------
CNN_LAYER_NUM_PER_STAGE=10
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
        --cnn_layer_num_per_stage $CNN_LAYER_NUM_PER_STAGE \
        --seed $SEED \
        --ablation
done


# --------------------- ResNet-44 on CIFAR-10 ---------------------
CNN_LAYER_NUM_PER_STAGE=14
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
        --cnn_layer_num_per_stage $CNN_LAYER_NUM_PER_STAGE \
        --seed $SEED \
        --ablation
done


# --------------------- ResNet-56 on CIFAR-10 ---------------------
CNN_LAYER_NUM_PER_STAGE=18
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
        --cnn_layer_num_per_stage $CNN_LAYER_NUM_PER_STAGE \
        --seed $SEED \
        --ablation
done


# --------------------- ResNet-20 on CIFAR-10 ---------------------
CNN_LAYER_NUM_PER_STAGE=6
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
        --cnn_layer_num_per_stage $CNN_LAYER_NUM_PER_STAGE \
        --seed $SEED \
        --ablation
done
