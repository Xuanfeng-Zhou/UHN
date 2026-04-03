#!/bin/bash
TASK_TYPE=TEXT_CLASSIFICATION
MODEL_TYPE=TRANSFORMER
MIX_STATUS=single

# IMDB
DATASET_TYPE=IMDB
python main.py \
    --task_type $TASK_TYPE \
    --dataset_type $DATASET_TYPE \
    --model_type $MODEL_TYPE \
    --mix_status $MIX_STATUS \
    --no_structure \
    --sweep
