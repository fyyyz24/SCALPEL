#!/usr/bin/env python
"""
Extract text features from a medical LLM for offline training.
Reads reports from JSON, encodes them with LLM2Vec, saves to .pt file.

Usage:
    # Without CRC (raw PMC-LLaMA bidirectional)
    python scripts/extract_llm_features.py \
        --model_path axiong/PMC_LLaMA_13B \
        --reports data/mimic_cxr/reports_train.json \
        --output data/mimic_cxr/text_features/pmcllama_raw.pt

    # With CRC-tuned LoRA weights
    python scripts/extract_llm_features.py \
        --model_path axiong/PMC_LLaMA_13B \
        --peft_path checkpoints/medical_llm_crc \
        --reports data/mimic_cxr/reports_train.json \
        --output data/mimic_cxr/text_features/pmcllama_crc.pt
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../llm2clip"))

from llm2vec import LLM2Vec
from transformers import AutoTokenizer


def load_reports(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    texts = []
    ids = []
    for item in data:
        report = item.get("report", item.get("findings", ""))
        if not report:
            report = item.get("impression", "")
        rid = item.get("id", str(len(ids)))
        texts.append(report)
        ids.append(rid)
    return ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--peft_path", type=str, default=None)
    parser.add_argument("--reports", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--cpu_offload", action="store_true", default=False)
    args = parser.parse_args()

    print(f"Loading reports from {args.reports}")
    ids, texts = load_reports(args.reports)
    print(f"Loaded {len(texts)} reports")

    peft_path = args.peft_path if args.peft_path else None
    print(f"Loading LLM: {args.model_path}")
    if peft_path:
        print(f"  with PEFT weights: {peft_path}")

    # Build kwargs for from_pretrained
    extra_kwargs = {}
    if args.load_in_4bit:
        print("  using 4-bit quantization (~7GB VRAM)")
        extra_kwargs["load_in_4bit"] = True
        extra_kwargs["bnb_4bit_compute_dtype"] = torch.bfloat16
        extra_kwargs["bnb_4bit_use_double_quant"] = True
    if args.cpu_offload:
        print("  using CPU offloading for inference")
        extra_kwargs["device_map"] = "auto"
        extra_kwargs["offload_folder"] = "/tmp/offload"
        extra_kwargs["offload_state_dict"] = True

    # Load base model first, then resize to match CRC vocab before loading PEFT
    if peft_path:
        from llm2vec.models.bidirectional_llama import LlamaBiModel
        base = LlamaBiModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if not args.load_in_4bit else "auto",
            **extra_kwargs,
        )
        # Add mask token to match CRC checkpoint (CRC training added <mask>)
        tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tok.padding_side = "left"  # LLM2Vec requires left-padding
        if hasattr(tok, 'mask_token') and tok.mask_token is None:
            tok.add_tokens(["<mask>"])
            tok.mask_token = "<mask>"
        if len(tok) > base.config.vocab_size:
            base.resize_token_embeddings(len(tok))
        # Now load PEFT weights (vocab size matches)
        from peft import PeftModel
        base = PeftModel.from_pretrained(base, peft_path)
        base = base.merge_and_unload()
        # Wrap in LLM2Vec with pooled mode
        model = LLM2Vec(
            model=base, tokenizer=tok, pooling_mode="mean",
            max_length=args.max_length, doc_max_length=args.max_length,
        )
    else:
        model = LLM2Vec.from_pretrained(
            args.model_path,
            peft_path,
            merge_peft=True if peft_path else False,
            enable_bidirectional=True,
            torch_dtype=torch.bfloat16 if not args.load_in_4bit else "auto",
            max_length=args.max_length,
            doc_max_length=args.max_length,
            **extra_kwargs,
        )

    device = torch.device(args.device)
    if not args.cpu_offload and not args.load_in_4bit:
        model.to(device)
    # 4-bit models are auto-placed by bitsandbytes; don't re-move them

    all_features = {}
    # Key by both sequential index (0, 1, 2...) AND report ID
    # for compatibility with MedicalJsonDataset which uses str(idx)
    for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
        batch_texts = texts[i:i + args.batch_size]
        batch_ids = ids[i:i + args.batch_size]
        features = model.encode(batch_texts, batch_size=args.batch_size,
                                convert_to_tensor=True, show_progress_bar=False)
        for j, (rid, feat) in enumerate(zip(batch_ids, features)):
            idx = i + j
            # Store under both the integer index (as str) and the report ID
            # MedicalJsonDataset reads by str(idx)
            all_features[str(idx)] = feat.cpu().half()
            all_features[rid] = all_features[str(idx)]  # also by report ID

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(all_features, args.output)
    print(f"Saved {len(all_features)} entries (2x reports) to {args.output}")
    print(f"Feature dim: {all_features['0'].shape}")


if __name__ == "__main__":
    main()
