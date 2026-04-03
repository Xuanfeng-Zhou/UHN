#!/bin/bash
TASK_TYPE=FORMULA_REGRESSION
MODEL_TYPE=KAN
MIX_STATUS=single

DATASET_LIST=(
    "SPECIAL_ELLIPJ"
    "SPECIAL_ELLIPKINC" 
    "SPECIAL_ELLIPEINC"

    "SPECIAL_LPMV0"
    "SPECIAL_LPMV1"
    "SPECIAL_LPMV2"

    "SPECIAL_JV"
    "SPECIAL_YV"
    "SPECIAL_KV"
    "SPECIAL_IV"

    "SPECIAL_SPH_HARM01"
    "SPECIAL_SPH_HARM11"
    "SPECIAL_SPH_HARM02"
    "SPECIAL_SPH_HARM12"
    "SPECIAL_SPH_HARM22"
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
