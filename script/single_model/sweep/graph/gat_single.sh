#!/bin/bash
TASK_TYPE=GRAPH_NODE_CLASSIFICATION
MIX_STATUS=single
MODEL_TYPE=GAT

DATASET_LIST=(
    CITESEER
    CORA
    PUBMED
)

for DATASET_TYPE in "${DATASET_LIST[@]}"
do
    python main.py \
        --task_type $TASK_TYPE \
        --dataset_type $DATASET_TYPE \
        --model_type $MODEL_TYPE \
        --mix_status $MIX_STATUS \
        --no_structure \
        --sweep
done
