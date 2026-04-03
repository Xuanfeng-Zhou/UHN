#!/bin/bash
TASK_TYPE=FORMULA_REGRESSION
MODEL_TYPE=KAN
MIX_STATUS=single
TRAIN_EPOCHS=4000

SEEDS=(21962 59358 78117)
for SEED in "${SEEDS[@]}"
do
    DATASET_TYPE=SPECIAL_ELLIPKINC
    TRAIN_LR=1e-4
    INIT_LR=2e-4
    INIT_STEPS=50
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

    DATASET_TYPE=SPECIAL_ELLIPEINC
    TRAIN_LR=1e-4
    INIT_LR=1e-4
    INIT_STEPS=50
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

    DATASET_TYPE=SPECIAL_ELLIPJ
    TRAIN_LR=1e-4
    INIT_LR=1e-4
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

    DATASET_TYPE=SPECIAL_IV
    TRAIN_LR=1e-4
    INIT_LR=5e-5
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

    DATASET_TYPE=SPECIAL_JV
    TRAIN_LR=5e-5
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

    DATASET_TYPE=SPECIAL_KV
    TRAIN_LR=1e-4
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

    DATASET_TYPE=SPECIAL_LPMV0
    TRAIN_LR=5e-5
    INIT_LR=5e-5
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

    DATASET_TYPE=SPECIAL_LPMV1
    TRAIN_LR=5e-5
    INIT_LR=5e-5
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

    DATASET_TYPE=SPECIAL_LPMV2
    TRAIN_LR=1e-4
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

    DATASET_TYPE=SPECIAL_SPH_HARM01
    TRAIN_LR=1e-4
    INIT_LR=0
    INIT_STEPS=0
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

    DATASET_TYPE=SPECIAL_SPH_HARM02
    TRAIN_LR=2e-5
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

    DATASET_TYPE=SPECIAL_SPH_HARM11
    TRAIN_LR=5e-5
    INIT_LR=5e-5
    INIT_STEPS=50
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

    DATASET_TYPE=SPECIAL_SPH_HARM12
    TRAIN_LR=1e-4
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

    DATASET_TYPE=SPECIAL_SPH_HARM22
    TRAIN_LR=2e-5
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

    DATASET_TYPE=SPECIAL_YV
    TRAIN_LR=1e-4
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
