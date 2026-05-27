#!/usr/bin/env python
"""
Simplified CRC fine-tuning: converts PMC-LLaMA to bidirectional encoder
using MNTP + LoRA. Bypasses the problematic LlamaBiForMNTP class.

Uses LLM2Vec's LlamaBiModel + a simple MNTP training loop.
4-bit quantization for memory efficiency (~12GB VRAM).

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/crc_simple.py \
        --model_path axiong/PMC_LLaMA_13B \
        --train_file data/mimic_cxr/text/crc_train.json \
        --output_dir checkpoints/medical_llm_crc \
        --max_steps 5000 --batch_size 4 --lr 2e-4
"""

import argparse
import json
import math
import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../llm2clip"))

from llm2vec.models.bidirectional_llama import LlamaBiModel
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model


class BiEncoderWithHead(nn.Module):
    """Bidirectional encoder + simple masked prediction head."""
    def __init__(self, model_name_or_path, load_in_4bit=True):
        super().__init__()
        print(f"Loading {model_name_or_path} (4bit={load_in_4bit})...")
        load_kwargs = {}
        if load_in_4bit:
            load_kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "bnb_4bit_use_double_quant": True,
                "device_map": "auto",
            }
        self.encoder = LlamaBiModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
            **load_kwargs,
        )
        hidden_dim = self.encoder.config.hidden_size
        self.lm_head = nn.Linear(hidden_dim, self.encoder.config.vocab_size, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        # Match device and cast hidden to lm_head's float32 dtype
        if self.lm_head.weight.device != hidden.device:
            self.lm_head = self.lm_head.to(hidden.device)
        logits = self.lm_head(hidden.float())
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        # Save only LoRA weights
        self.encoder.save_pretrained(path)


def load_texts(path, max_samples=None):
    texts = []
    with open(path) as f:
        for line in f:
            try:
                item = json.loads(line)
                texts.append(item.get("text", item.get("report", "")))
            except:
                texts.append(line.strip())
    if max_samples:
        texts = texts[:max_samples]
    return texts


def mask_tokens(input_ids, tokenizer, mlm_prob=0.15):
    """Randomly mask tokens for MNTP training."""
    labels = input_ids.clone()
    rand = torch.rand(input_ids.shape, device=input_ids.device)
    mask = rand < mlm_prob
    # Don't mask special tokens
    special_mask = (input_ids == tokenizer.pad_token_id) | \
                   (input_ids == tokenizer.bos_token_id) | \
                   (input_ids == tokenizer.eos_token_id)
    mask = mask & ~special_mask
    labels[~mask] = -100
    input_ids = input_ids.clone()
    input_ids[mask] = tokenizer.mask_token_id if tokenizer.mask_token_id else tokenizer.unk_token_id
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--mlm_prob", type=float, default=0.30)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--no_4bit", action="store_true")
    args = parser.parse_args()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.mask_token is None:
        tokenizer.add_tokens(["<mask>"])
        tokenizer.mask_token = "<mask>"

    print("Loading model...")
    model = BiEncoderWithHead(args.model_path, load_in_4bit=not args.no_4bit)
    # Resize embeddings if we added new tokens
    if len(tokenizer) > model.encoder.config.vocab_size:
        model.encoder.resize_token_embeddings(len(tokenizer))
        # Place new head on same device/dtype as encoder
        old_device = next(model.lm_head.parameters()).device
        old_dtype = next(model.lm_head.parameters()).dtype
        model.lm_head = nn.Linear(model.encoder.config.hidden_size, len(tokenizer), bias=False)
        model.lm_head = model.lm_head.to(device=old_device, dtype=old_dtype)
        print(f"Resized embeddings: {model.encoder.config.vocab_size} -> {len(tokenizer)}")

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    model.encoder = get_peft_model(model.encoder, lora_config)
    print("LoRA applied. Trainable params:")
    model.encoder.print_trainable_parameters()

    print("Loading training data...")
    texts = load_texts(args.train_file)
    print(f"Loaded {len(texts)} reports")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup, num_training_steps=args.max_steps
    )

    global_step, accum_loss, batch_count = 0, 0.0, 0
    pbar = tqdm(total=args.max_steps, desc="CRC training")
    model.train()

    while global_step < args.max_steps:
        for i in range(0, len(texts), args.batch_size * args.grad_accum):
            if global_step >= args.max_steps:
                break

            batch_texts = texts[i:i + args.batch_size]
            if not batch_texts:
                continue

            tokens = tokenizer(batch_texts, padding=True, truncation=True,
                               max_length=args.max_length, return_tensors="pt")
            input_ids = tokens["input_ids"].to(model.encoder.device)
            attention_mask = tokens["attention_mask"].to(model.encoder.device)

            input_ids, labels = mask_tokens(input_ids, tokenizer, args.mlm_prob)

            loss = model(input_ids, attention_mask, labels=labels)
            loss = loss / args.grad_accum
            loss.backward()

            accum_loss += loss.item()
            batch_count += 1

            # Step after grad_accum micro-batches
            if batch_count % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.update(1)

                if global_step % args.log_steps == 0:
                    pbar.set_postfix({"loss": f"{accum_loss/args.log_steps:.4f}"})
                    accum_loss = 0.0

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"step-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    model.encoder.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    print(f"\nSaved checkpoint to {save_path}")

    # Final save
    os.makedirs(args.output_dir, exist_ok=True)
    model.encoder.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nCRC fine-tuning complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
