#!/bin/bash
# Zero-shot retrieval evaluation on MIMIC-CXR test set
# Uses a trained SCALPEL checkpoint

MODEL_NAME="SCALPEL-CXRBERT-DINOv2-L-14"
CHECKPOINT="logs/scalpel_cxrbert_l/checkpoints/epoch_30.pt"
TEST_JSON="./data/mimic_cxr/reports_test.json"
IMG_ROOT="./data/mimic_cxr/images"

python llm2clip/training/main.py \
    --model ${MODEL_NAME} \
    --force-custom-clip \
    --resume ${CHECKPOINT} \
    --val-data ${TEST_JSON} \
    --dataset-type medical_json \
    --img-root ${IMG_ROOT} \
    --eval-batch-size 64 \
    --precision amp_bf16 \
    --workers 4 \
    --logs ./logs \
    --name eval_test
