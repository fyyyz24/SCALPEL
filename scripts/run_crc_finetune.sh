#!/bin/bash
# SCALPEL Stage 1 (Optional): CRC Fine-tuning of Medical LLM
# Uses run_mntp.py to train bidirectional medical LLM with LoRA.
#
# Three profiles for different hardware budgets:
#   bash scripts/run_crc_finetune.sh small    # ~4GB VRAM, PubMedBERT
#   bash scripts/run_crc_finetune.sh medium   # ~12GB VRAM, QLoRA 4bit + 13B
#   bash scripts/run_crc_finetune.sh large    # ~40GB VRAM, LoRA bf16 + 13B
#   bash scripts/run_crc_finetune.sh          # default = small
#
# SKIP this stage entirely if you:
#   - Use a pre-trained medical embedding model directly (e.g. MedCPT)
#   - Find that raw LLM features already work well enough for your task

set -e

PROFILE="${1:-small}"

# === Common settings ===
OUTPUT_DIR="./checkpoints/medical_llm_crc"
TRAIN_FILE="./data/mimic_cxr/text/crc_train.json"
VAL_FILE="./data/mimic_cxr/text/crc_train.json"
MAX_SEQ_LENGTH=512
STEPS=10000
WARMUP_STEPS=500
SEED=42

case "$PROFILE" in
  small)
    echo "=== Profile: small (~4GB VRAM) — PubMedBERT + LoRA ==="
    BASE_MODEL="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    BATCH_SIZE=64
    GRAD_ACCUM=2
    LR=5e-4
    LORA_R=8
    LORA_ALPHA=16
    LORA_DROPOUT=0.05
    QLORA=""
    DTYPE="float16"
    USE_BF16=""
    ;;

  medium)
    echo "=== Profile: medium (~12GB VRAM) — PMC-LLaMA 13B + QLoRA 4bit ==="
    BASE_MODEL="axiong/PMC_LLaMA_13B"
    BATCH_SIZE=16
    GRAD_ACCUM=4
    LR=2e-4
    LORA_R=8
    LORA_ALPHA=16
    LORA_DROPOUT=0.05
    # QLoRA: 4-bit quantized base model, only LoRA params get gradients
    QLORA="--load_in_4bit --bnb_4bit_compute_dtype bfloat16 --bnb_4bit_use_double_quant"
    DTYPE="bfloat16"
    USE_BF16="--bf16"
    ;;

  large)
    echo "=== Profile: large (~40GB VRAM) — PMC-LLaMA 13B + LoRA bf16 ==="
    BASE_MODEL="axiong/PMC_LLaMA_13B"
    BATCH_SIZE=32
    GRAD_ACCUM=4
    LR=2e-4
    LORA_R=8
    LORA_ALPHA=16
    LORA_DROPOUT=0.05
    QLORA=""
    DTYPE="bfloat16"
    USE_BF16="--bf16"
    ;;

  *)
    echo "Unknown profile: $PROFILE (use: small | medium | large)"
    exit 1
    ;;
esac

echo "Base model: ${BASE_MODEL}"
echo "Output: ${OUTPUT_DIR}"

python llm_caption_contrastive/run_mntp.py \
    --model_name_or_path "${BASE_MODEL}" \
    --train_file "${TRAIN_FILE}" \
    --validation_file "${VAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --learning_rate ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --weight_decay 0.01 \
    --max_steps ${STEPS} \
    --lr_scheduler_type "cosine" \
    --save_steps 2000 \
    --logging_steps 100 \
    --seed ${SEED} \
    --do_train \
    --do_eval \
    --evaluation_strategy "steps" \
    --eval_steps 2000 \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT} \
    --mask_token_type "blank" \
    --stop_after_n_steps ${STEPS} \
    --attn_implementation "flash_attention_2" \
    --torch_dtype "${DTYPE}" \
    ${QLORA} \
    ${USE_BF16} \
    --overwrite_output_dir

echo "=== CRC Fine-tuning complete ==="
echo "Checkpoint: ${OUTPUT_DIR}"
