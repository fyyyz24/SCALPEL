#!/bin/bash
# SCALPEL Training Pipeline
# Stage 1 (Optional): CRC Fine-tuning → Stage 2: Feature Extraction
# → Stage 3: Metadata Extraction → Stage 4: SCALPEL Training
#
# Usage:
#   # Full pipeline (skip CRC if checkpoint exists)
#   bash scripts/train_scalpel.sh
#
#   # Skip CRC — use raw HuggingFace model for feature extraction
#   SKIP_CRC=1 LLM_MODEL=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext bash scripts/train_scalpel.sh
#
#   # Quick test: skip CRC, skip metadata extraction (no ANAO)
#   SKIP_CRC=1 SKIP_METADATA=1 bash scripts/train_scalpel.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Server paths (modify these to match your setup) ----
MIMIC_RAW_PATH="${MIMIC_RAW_PATH:-./data/mimic-cxr-dataset/data}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data/mimic_cxr}"
TEXT_FEATURES_DIR="${DATA_DIR}/text_features"
METADATA_DIR="${DATA_DIR}/metadata"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
LOGS_DIR="${PROJECT_ROOT}/logs"

# ---- Configurable ----
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"   # set to 1 if data already prepared
SKIP_CRC="${SKIP_CRC:-0}"
SKIP_METADATA="${SKIP_METADATA:-0}"
LLM_MODEL="${LLM_MODEL:-${CHECKPOINT_DIR}/medical_llm_crc}"
MODEL_NAME="SCALPEL-PMCLLaMA-DINOv2-L-14"
TRAIN_JSON="${DATA_DIR}/reports_train.json"
VAL_JSON="${DATA_DIR}/reports_val.json"
EPOCHS=30
BATCH_SIZE=256
LR=1e-4
ANAO_ANAT_WEIGHT=0.1
ANAO_NEG_WEIGHT=0.1
SEED=42

echo "============================================"
echo "  SCALPEL Training Pipeline"
echo "  Raw data: ${MIMIC_RAW_PATH}"
echo "  Output:   ${DATA_DIR}"
echo "  Skip Prep: ${SKIP_DATA_PREP}  |  Skip CRC: ${SKIP_CRC}  |  Skip Metadata: ${SKIP_METADATA}"
echo "  LLM: ${LLM_MODEL}"
echo "============================================"
echo ""

# ---- Stage 0: Data Preparation (Parquet -> SCALPEL JSON) ----
if [ "${SKIP_DATA_PREP}" = "1" ]; then
    echo "--- Stage 0: Data Preparation SKIPPED (SKIP_DATA_PREP=1) ---"
elif [ -f "${TRAIN_JSON}" ]; then
    echo "--- Stage 0: Data already prepared at ${DATA_DIR}, skipping ---"
else
    echo "--- Stage 0: Data Preparation (Parquet -> JSON) ---"
    mkdir -p "${DATA_DIR}"
    bash scripts/data/download_mimic_cxr.sh --data-path "${MIMIC_RAW_PATH}" --output "${DATA_DIR}"
    echo "  Data prepared -> ${DATA_DIR}"
fi

# ---- Stage 1: CRC Fine-tuning (Optional) ----
if [ "${SKIP_CRC}" = "1" ]; then
    echo "--- Stage 1: CRC Fine-tuning SKIPPED (SKIP_CRC=1) ---"
    echo "  Will use LLM model directly: ${LLM_MODEL}"
else
    if [ ! -d "${CHECKPOINT_DIR}/medical_llm_crc" ]; then
        echo "--- Stage 1: CRC Fine-tuning ---"
        bash scripts/run_crc_finetune.sh small
    else
        echo "--- Stage 1: CRC checkpoint exists, skipping ---"
    fi
fi

# ---- Stage 2: Text Feature Extraction ----
echo "--- Stage 2: Text Feature Extraction ---"
mkdir -p "${TEXT_FEATURES_DIR}"
python llm2clip/data/extract_embedding.py \
    --data_path "${DATA_DIR}" \
    --checkpoint "${LLM_MODEL}"
echo "  Features → ${TEXT_FEATURES_DIR}"

# ---- Stage 3: ANAO Metadata Extraction ----
if [ "${SKIP_METADATA}" = "1" ]; then
    echo "--- Stage 3: ANAO Metadata SKIPPED (SKIP_METADATA=1) ---"
    ANAO_FLAG=""
    METADATA_ARG=""
else
    echo "--- Stage 3: ANAO Metadata Extraction ---"
    mkdir -p "${METADATA_DIR}"
    python llm2clip/data/extract_medical_metadata.py \
        --reports_file "${TRAIN_JSON}" \
        --output "${METADATA_DIR}/anao_labels_train.json"
    python llm2clip/data/extract_medical_metadata.py \
        --reports_file "${VAL_JSON}" \
        --output "${METADATA_DIR}/anao_labels_val.json"
    echo "  Metadata → ${METADATA_DIR}"
    ANAO_FLAG="--use-anao-loss"
    METADATA_ARG="--medical-metadata-path ${METADATA_DIR}/anao_labels_train.json"
fi

# ---- Stage 4: SCALPEL Training ----
echo "--- Stage 4: SCALPEL Training ---"
mkdir -p "${LOGS_DIR}"

python llm2clip/training/main.py \
    --model "${MODEL_NAME}" \
    --force-custom-clip \
    --pretrained-visual-model "vit_large_patch14_dinov2" \
    --pretrained-image "dinov2_vitl14" \
    --train-data "${TRAIN_JSON}" \
    --val-data "${VAL_JSON}" \
    --dataset-type "medical_json" \
    --medical-dataset-name "mimic_cxr" \
    --img-root "${DATA_DIR}/images" \
    --text-feature-path "${TEXT_FEATURES_DIR}/text_embeddings.pt" \
    ${METADATA_ARG} \
    ${ANAO_FLAG} \
    --anao-anat-weight ${ANAO_ANAT_WEIGHT} \
    --anao-neg-weight ${ANAO_NEG_WEIGHT} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --text-lr 2e-5 \
    --visual-lr 1e-4 \
    --wd 0.01 \
    --warmup 2000 \
    --lock-text \
    --precision "amp_bf16" \
    --workers 8 \
    --logs "${LOGS_DIR}" \
    --name "scalpel_mimic_cxr" \
    --report-to "tensorboard" \
    --seed ${SEED} \
    --save-frequency 5 \
    --val-frequency 1 \
    --zeroshot-frequency 5 \
    --log-every-n-steps 50

echo ""
echo "=== Done ==="
echo "Checkpoints: ${LOGS_DIR}/scalpel_mimic_cxr/checkpoints/"
