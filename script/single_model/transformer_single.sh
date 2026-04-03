#!/bin/bash
TASK_TYPE=TEXT_CLASSIFICATION
MODEL_TYPE=TRANSFORMER
MIX_STATUS=single

SEEDS=(21962 59358 78117)
for SEED in "${SEEDS[@]}"
do
    # AG_NEWS
    DATASET_TYPE=AG_NEWS
    INIT_LR=0
    TRAIN_LR=5e-5
    INIT_STEPS=0
    TRAIN_EPOCHS=50
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

    # IMDB
    DATASET_TYPE=IMDB
    INIT_LR=5e-5
    TRAIN_LR=1e-4
    INIT_STEPS=100
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
done
