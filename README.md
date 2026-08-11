![](https://capsule-render.vercel.app/api?type=waving&height=260&color=0:D22229,100:2B4FA3&section=header&text=SCALPEL:%20Semantic%20Cross-modal%20Alignment%20via%20LLM-Powered&fontSize=22&fontColor=FFFFFF&fontAlignY=38&animation=fadeIn&textBg=false&fontWeight=700&desc=Encoder%20Learning%20for%20Medical%20Vision-Language%20Representation-nl-Yunzhan%20Fu,%20Enyu%20Bao,%20Xiangyu%20Shen,%20Yihao%20Wu,%20Chunbo%20Jiang,%20Fangli%20Guan,%20and%20Liqi%20Yan&descSize=16&descAlignY=62)
# SCALPEL: Semantic Cross-modal Alignment via LLM-Powered Encoder Learning for Medical Vision-Language Representation 

This is pytorch official code repository for our paper **SCALPEL**, a plug-and-play framework for Semantic Cross-modal Alignment via LLM-Powered Encoder Learning for Medical Vision-Language Representation. accepted by [PRCV 2026](https://www.prcv.cn/web/#/home) [[🎩 arXiv]](https://arxiv.org/abs/2607.26885)
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
├── eva_clip/          
│   ├── anao_loss.py   
│   ├── hf_model.py    
│   ├── model.py       
│   └── model_configs/ 
├── training/          
├── data/              
└── llm2vec/           
llm_caption_contrastive/
├── run_mntp.py        
└── run_supervised.py  
scripts/
├── crc_simple.py   
├── extract_llm_features.py  
├── train_scalpel.sh   
├── data/              
├── eval/              
└── measure_efficiency.py    
```


