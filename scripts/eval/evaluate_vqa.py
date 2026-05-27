#!/usr/bin/env python
"""
Zero-shot Medical VQA evaluation using trained SCALPEL checkpoint.
No fine-tuning needed — uses the CLIP-style similarity between
image and "question + answer" pairs.

Usage:
    python scripts/eval/evaluate_vqa.py \
        --checkpoint logs/scalpel_cxrbert_l/checkpoints/epoch_30.pt \
        --model SCALPEL-CXRBERT-DINOv2-L-14 \
        --dataset vqa_rad \
        --data_path /path/to/vqa_rad.json
"""

import argparse
import json
import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../llm2clip"))

from llm2clip.eva_clip.factory import create_model, get_tokenizer, get_model_config


def load_model(model_name, checkpoint_path, device):
    """Load trained SCALPEL model from checkpoint."""
    model = create_model(
        model_name,
        pretrained=None,
        precision="amp_bf16",
        device=device,
        force_custom_clip=True,
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if next(iter(state_dict.keys())).startswith("module"):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def encode_texts(model, texts, tokenizer, device):
    """Batch-encode text strings."""
    if isinstance(texts, str):
        texts = [texts]
    input_ids = tokenizer(texts).to(device)
    with torch.no_grad():
        features = model.encode_text(input_ids, normalize=True)
    return features


def encode_images(model, images, device, dtype=torch.bfloat16):
    """Encode a batch of pre-processed images."""
    with torch.no_grad():
        features = model.encode_image(images.to(device=device, dtype=dtype), normalize=True)
    return features


def evaluate_vqa_rad(model, data_path, tokenizer, transforms, device):
    """
    Zero-shot VQA on VQA-RAD dataset.
    VQA-RAD format: list of {"question": "...", "image": "path", "answer": "...", "answer_type": "..."}
    """
    data = json.load(open(data_path))
    correct, total = 0, 0

    for item in tqdm(data, desc="VQA-RAD"):
        # Load and preprocess image
        img_path = item.get("image_path", item.get("image"))
        if not os.path.exists(img_path):
            # Try relative to data_path
            img_path = os.path.join(os.path.dirname(data_path), img_path)
        if not os.path.exists(img_path):
            continue

        from PIL import Image
        img = transforms(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        img_feat = encode_images(model, img, device)

        question = item["question"]
        candidates = item.get("answer_choices", [item["answer"]])
        gt_answer = item["answer"].lower().strip()

        if not candidates or len(candidates) < 2:
            # Open-ended: can't do zero-shot without candidates
            continue

        # Encode "question + answer" for each candidate
        prompts = [f"{question} {ans}" for ans in candidates]
        text_feats = encode_texts(model, prompts, tokenizer, device)

        # Cosine similarity
        sim = (img_feat @ text_feats.T).squeeze(0)
        best_idx = sim.argmax().item()
        pred_answer = candidates[best_idx].lower().strip()

        if pred_answer == gt_answer:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\nVQA-RAD Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def evaluate_slake(model, data_path, tokenizer, transforms, device):
    """
    Zero-shot VQA on SLAKE dataset.
    SLAKE format: list of {"question": "...", "img_name": "...", "answer": "...", "q_lang": "en"}
    """
    data = json.load(open(data_path))
    correct, total = 0, 0
    img_dir = os.path.join(os.path.dirname(data_path), "imgs")

    for item in tqdm(data, desc="SLAKE"):
        img_path = os.path.join(img_dir, item["img_name"])
        if not os.path.exists(img_path):
            continue

        from PIL import Image
        img = transforms(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        img_feat = encode_images(model, img, device)

        question = item["question"]
        gt_answer = item["answer"].lower().strip()

        # SLAKE has closed-set answers (yes/no for closed questions)
        if item.get("q_type") == "OPEN":
            candidates = list(set(item.get("answer", []) for item in data if item.get("q_type") == "OPEN"))
        else:
            candidates = ["yes", "no"]

        prompts = [f"{question} {ans}" for ans in candidates]
        if len(prompts) < 2:
            continue

        text_feats = encode_texts(model, prompts, tokenizer, device)
        sim = (img_feat @ text_feats.T).squeeze(0)
        pred_answer = candidates[sim.argmax().item()].lower().strip()

        if pred_answer == gt_answer:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"\nSLAKE Accuracy: {acc:.4f} ({correct}/{total})")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, default="SCALPEL-CXRBERT-DINOv2-L-14")
    parser.add_argument("--dataset", type=str, choices=["vqa_rad", "slake"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model: {args.model}")
    model = load_model(args.model, args.checkpoint, device)
    tokenizer = get_tokenizer(args.model)

    # Get image transforms
    from llm2clip.eva_clip.transform import image_transform
    config = get_model_config(args.model)
    image_size = config["vision_cfg"].get("image_size", 224)
    transforms = image_transform(image_size, is_train=False)

    # Run evaluation
    if args.dataset == "vqa_rad":
        evaluate_vqa_rad(model, args.data_path, tokenizer, transforms, device)
    elif args.dataset == "slake":
        evaluate_slake(model, args.data_path, tokenizer, transforms, device)


if __name__ == "__main__":
    main()
