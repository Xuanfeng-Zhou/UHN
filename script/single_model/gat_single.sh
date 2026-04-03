#!/bin/bash
TASK_TYPE=GRAPH_NODE_CLASSIFICATION
MIX_STATUS=single

MODEL_TYPE=GAT

SEEDS=(21962 59358 78117 23547 81797 81105 95819 42270 24014 20964)
for SEED in "${SEEDS[@]}"
do
    # Citeseer
    DATASET_TYPE=CITESEER
    TRAIN_LR=5e-5
    TRAIN_EPOCHS=400
    INIT_LR=2e-4
    INIT_STEPS=200
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

    # Cora
    DATASET_TYPE=CORA
    TRAIN_LR=5e-5
    TRAIN_EPOCHS=400
    INIT_LR=2e-4
    INIT_STEPS=100
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

    # PubMed
    DATASET_TYPE=PUBMED
    TRAIN_LR=2e-5
    TRAIN_EPOCHS=400
    INIT_LR=2e-4
    INIT_STEPS=100
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
