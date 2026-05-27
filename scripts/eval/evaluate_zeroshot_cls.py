#!/usr/bin/env python
"""
Zero-shot medical image classification using trained SCALPEL checkpoint.
Evaluates on CheXpert, RSNA Pneumonia, NIH ChestX-ray14, and COVID datasets.

Uses prompt templates: "A chest X-ray showing {disease}" style matching.

Usage:
    python scripts/eval/evaluate_zeroshot_cls.py \
        --checkpoint logs/scalpel_cxrbert_l/checkpoints/epoch_30.pt \
        --model SCALPEL-CXRBERT-DINOv2-L-14 \
        --dataset chexpert \
        --data_path /path/to/chexpert
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Add llm2clip/ to path for internal imports (eva_clip, llm2vec, etc.)
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "../.."))          # repo root
sys.path.insert(0, os.path.join(_script_dir, "../../llm2clip")) # llm2clip/ dir
from llm2clip.eva_clip.factory import create_model, get_tokenizer, get_model_config
from llm2clip.eva_clip.transform import image_transform


# ---- Prompt Templates ----
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

RSNA_LABELS = [
    "No Pneumonia",
    "Pneumonia",
]

NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural Thickening", "Hernia", "No Finding",
]

COVID_LABELS = [
    "COVID-19 Negative",
    "COVID-19 Positive",
]

PROMPT_TEMPLATES = [
    "A chest X-ray showing {}.",
    "Radiograph indicating {}.",
    "Findings consistent with {}.",
    "Medical image demonstrating {}.",
    "X-ray with evidence of {}.",
]


def load_model(model_name, checkpoint_path, device):
    model = create_model(model_name, pretrained=None, precision="amp_bf16",
                         device=device, force_custom_clip=True)
    sd = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if next(iter(sd.keys())).startswith("module"):
        sd = {k[7:]: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def encode_texts(model, texts, tokenizer, device):
    input_ids = tokenizer(texts).to(device)
    features = model.encode_text(input_ids, normalize=True)
    return features


@torch.no_grad()
def encode_images(model, images, device, dtype=torch.bfloat16):
    return model.encode_image(images.to(device=device, dtype=dtype), normalize=True)


def compute_zeroshot_classifier(model, labels, tokenizer, device, templates=None):
    """Build zero-shot classifier weights from label prompts."""
    if templates is None:
        templates = ["{}"]
    all_prompts = []
    for label in labels:
        all_prompts.extend([t.format(label) for t in templates])
    text_feats = encode_texts(model, all_prompts, tokenizer, device)
    # Average across templates: [num_labels, dim]
    text_feats = text_feats.view(len(labels), len(templates), -1).mean(dim=1)
    return F.normalize(text_feats, dim=-1)


def evaluate_chexpert(model, data_path, tokenizer, transforms, device):
    """Zero-shot multi-label classification on CheXpert."""
    from torch.utils.data import DataLoader, Dataset
    import pandas as pd
    from PIL import Image

    labels = CHEXPERT_LABELS
    classifier = compute_zeroshot_classifier(model, labels, tokenizer, device, PROMPT_TEMPLATES)

    # Load CheXpert validation set
    csv_path = os.path.join(data_path, "valid.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(data_path, "CheXpert-v1.0-small", "valid.csv")
    df = pd.read_csv(csv_path)
    df = df.fillna(0)

    # Filter frontal views only (standard protocol)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']

    # Strip Kaggle packaging prefix if present (Path = "CheXpert-v1.0-small/valid/..." but files at "valid/...")
    path_stripped = False
    if not os.path.exists(os.path.join(data_path, df.iloc[0]['Path'])):
        path_stripped = True
        df['Path'] = df['Path'].str.replace(r'^CheXpert-v[0-9.]+-[^/]+/', '', regex=True)
        if not os.path.exists(os.path.join(data_path, df.iloc[0]['Path'])):
            # Try without the "valid/" or "train/" prefix too
            df['Path'] = df['Path'].str.replace(r'^(valid|train)/', '', regex=True)

    all_probs, all_gts = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="CheXpert"):
        img_path = os.path.join(data_path, row["Path"])
        if not os.path.exists(img_path):
            continue
        try:
            img = transforms(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        except:
            continue
        img_feat = encode_images(model, img, device)
        logits = (img_feat @ classifier.T).squeeze(0).cpu().float()
        probs = torch.sigmoid(logits).numpy()
        all_probs.append(probs)

        gt = row[labels].values.astype(float)
        # -1 → 0 (uncertain → negative) for CheXpert
        gt[gt == -1] = 0
        all_gts.append(gt)

    if not all_probs:
        print("No valid images found.")
        return

    all_probs = np.stack(all_probs)
    all_gts = np.stack(all_gts)

    # Per-class AUC
    print(f"\n{'Finding':<35} {'AUC':>8}")
    print("-" * 45)
    for i, label in enumerate(labels):
        if all_gts[:, i].sum() > 0:
            auc = roc_auc_score(all_gts[:, i], all_probs[:, i])
            print(f"{label:<35} {auc:>8.4f}")
    mean_auc = np.mean([roc_auc_score(all_gts[:, i], all_probs[:, i])
                        for i in range(len(labels)) if all_gts[:, i].sum() > 0])
    print(f"\n{'Mean AUC':<35} {mean_auc:>8.4f}")


def evaluate_rsna(model, data_path, tokenizer, transforms, device):
    """Zero-shot classification on RSNA Pneumonia Detection."""
    import pandas as pd
    from PIL import Image
    import glob

    labels = RSNA_LABELS
    classifier = compute_zeroshot_classifier(model, labels, tokenizer, device, PROMPT_TEMPLATES)

    csv_path = os.path.join(data_path, "stage_2_test_labels.csv") if \
        os.path.exists(os.path.join(data_path, "stage_2_test_labels.csv")) else \
        os.path.join(data_path, "stage_2_train_labels.csv")
    img_dir = os.path.join(data_path, "test") if \
        os.path.exists(os.path.join(data_path, "test")) else \
        os.path.join(data_path, "train")

    df = pd.read_csv(csv_path)
    preds, targets = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="RSNA"):
        patient_id = row["patientId"]
        img_files = glob.glob(os.path.join(img_dir, f"{patient_id}*.dcm")) + \
                    glob.glob(os.path.join(img_dir, f"{patient_id}*.png")) + \
                    glob.glob(os.path.join(img_dir, f"{patient_id}*.jpg"))
        if not img_files:
            continue
        img_path = img_files[0]
        try:
            if img_path.endswith(".dcm"):
                import pydicom
                img = pydicom.dcmread(img_path).pixel_array
                img = Image.fromarray(img).convert("RGB")
            else:
                img = Image.open(img_path).convert("RGB")
            img_tensor = transforms(img).unsqueeze(0).to(device)
        except:
            continue

        img_feat = encode_images(model, img_tensor, device)
        logits = (img_feat @ classifier.T).squeeze(0)
        pred = logits.argmax().item()
        preds.append(pred)
        targets.append(int(row["Target"]))

    if preds:
        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds)
        print(f"\nRSNA Accuracy: {acc:.4f}, F1: {f1:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="SCALPEL-CXRBERT-DINOv2-L-14")
    parser.add_argument("--dataset", type=str, choices=["chexpert", "rsna", "covid"],
                        required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.model} from {args.checkpoint}")
    model = load_model(args.model, args.checkpoint, device)
    tokenizer = get_tokenizer(args.model)

    config = get_model_config(args.model)
    image_size = config["vision_cfg"].get("image_size", 224)
    transforms = image_transform(image_size, is_train=False)

    if args.dataset == "chexpert":
        evaluate_chexpert(model, args.data_path, tokenizer, transforms, device)
    elif args.dataset == "rsna":
        evaluate_rsna(model, args.data_path, tokenizer, transforms, device)


if __name__ == "__main__":
    main()
