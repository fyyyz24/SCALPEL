![](https://capsule-render.vercel.app/api?type=waving&height=240&color=0:D22229,100:2B4FA3&section=header&text=SCALPEL:%20Semantic%20Cross-modal%20Alignment%20via%20LLM-Powered&desc=Encoder%20Learning%20for%20Medical%20Vision-Language%20Representation&fontSize=16&descSize=16&fontColor=FFFFFF&fontAlignY=36&descAlignY=58&animation=fadeIn&textBg=false)
# SCALPEL: Semantic Cross-modal Alignment via LLM-Powered Encoder Learning for Medical Vision-Language Representation

Official code repository for **SCALPEL**, a plug-and-play framework for Semantic Cross-modal Alignment via LLM-Powered Encoder Learning for Medical Vision-Language Representation. 
SCALPEL injects clinical knowledge from medical LLMs into
cross-modal alignment through three innovations: (1) Clinical Report Contrastive (CRC)
fine-tuning to convert generative LLMs into bidirectional clinical text encoders,
(2) asymmetric architecture with offline feature caching for memory-efficient training,
and (3) Anatomy-Negation Aware Objective (ANAO) that explicitly penalizes anatomical
laterality confusion and clinical negation blindness.

## Installation

```bash
conda create -n scalpel python=3.10 -y && conda activate scalpel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Data Preparation

### MIMIC-CXR

Apply for access at [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/), then:

```bash
# Option A: Download from PhysioNet with credentials
bash scripts/data/download_mimic_cxr.sh --download --user USERNAME --password PASSWORD

# Option B: Convert existing HuggingFace Parquet files
bash scripts/data/download_mimic_cxr.sh --data-path /path/to/parquet/files
```

### IU X-Ray

```bash
bash scripts/data/download_iu_xray.sh
```

### Quick Test (Dummy Data)

```bash
python scripts/data/generate_dummy_data.py --num_samples 200 --output data/dummy_medical
```

## Training

### Full Pipeline

```bash
bash scripts/train_scalpel.sh
```

### CRC Fine-tuning (optional, LLM path only)

```bash
bash scripts/run_crc_finetune.sh medium
```

### Standalone Training

```bash
python llm2clip/training/main.py \
    --model SCALPEL-CXRBERT-DINOv2-L-14 \
    --force-custom-clip \
    --train-data ./data/mimic_cxr/reports_train.json \
    --val-data ./data/mimic_cxr/reports_val.json \
    --dataset-type medical_json \
    --img-root ./data/mimic_cxr/images \
    --medical-metadata-path ./data/mimic_cxr/metadata/anao_labels_train.json \
    --use-anao-loss \
    --epochs 30 --batch-size 128 --lr 1e-4 \
    --precision amp_bf16 --logs ./logs --name scalpel
```

## Evaluation

### Cross-modal Retrieval

```bash
bash scripts/eval/evaluate_retrieval.sh
```

### Zero-shot Classification (CheXpert)

```bash
python scripts/eval/evaluate_zeroshot_cls.py \
    --checkpoint logs/scalpel/checkpoints/epoch_30.pt \
    --model SCALPEL-CXRBERT-DINOv2-L-14 \
    --dataset chexpert --data_path /path/to/chexpert
```

### Zero-shot Medical VQA

```bash
python scripts/eval/evaluate_vqa.py \
    --checkpoint logs/scalpel/checkpoints/epoch_30.pt \
    --model SCALPEL-CXRBERT-DINOv2-L-14 \
    --dataset vqa_rad --data_path /path/to/vqa_rad
```

## Repository Structure

```
llm2clip/
├── eva_clip/          # Core model: CustomCLIP, ANAO loss, HF/TiMM adapters
│   ├── anao_loss.py   # Anatomy-Negation Aware Objective
│   ├── hf_model.py    # HuggingFace text encoder wrapper
│   ├── model.py       # CLIP / CustomCLIP / TextProj
│   └── model_configs/ # Pre-defined architecture configurations
├── training/          # Training pipeline, data loading, distributed utils
├── data/              # Medical NER, metadata extraction, data preparation
└── llm2vec/           # LLM bidirectional conversion (LLM2Vec integration)
llm_caption_contrastive/
├── run_mntp.py        # CRC fine-tuning via MNTP
└── run_supervised.py  # Supervised contrastive fine-tuning (SimCSE)
scripts/
├── crc_simple.py      # Lightweight CRC training loop
├── extract_llm_features.py  # Offline LLM feature extraction
├── train_scalpel.sh   # Full training pipeline
├── data/              # Dataset download and preparation
├── eval/              # Retrieval, classification, VQA evaluation
└── measure_efficiency.py    # Params / FLOPs measurement
```


