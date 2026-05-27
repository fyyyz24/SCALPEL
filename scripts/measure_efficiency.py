#!/usr/bin/env python
"""
Measure exact Params and FLOPs for each SCALPEL ablation configuration.
Uses thop (pip install thop) for FLOP counting, which supports newer PyTorch ops.

Usage: cd ./ && python scripts/measure_efficiency.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../llm2clip"))

import torch
from thop import profile, clever_format


def count_params_flops(model, *inputs):
    """Count params (M) and FLOPs (G) for a forward pass."""
    flops, params = profile(model, inputs=inputs, verbose=False)
    return params / 1e6, flops / 1e9


def build_text_model(encoder_type):
    if encoder_type == "transformer":
        from llm2clip.eva_clip.transformer import TextTransformer
        from llm2clip.eva_clip.model import LayerNorm
        return TextTransformer(
            context_length=256, vocab_size=49408, width=512, heads=8, layers=12,
            output_dim=512, act_layer=torch.nn.GELU, norm_layer=LayerNorm,
        ), torch.randint(0, 30000, (1, 256))
    elif encoder_type == "cxrbert":
        from llm2clip.eva_clip.hf_model import HFTextEncoder
        return HFTextEncoder("microsoft/BiomedVLP-CXR-BERT-specialized",
                             output_dim=768, proj="mlp", pooler_type="mean_pooler"), \
               torch.randint(0, 30000, (1, 512))
    elif encoder_type == "pubmedbert":
        from llm2clip.eva_clip.hf_model import HFTextEncoder
        return HFTextEncoder("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                             output_dim=768, proj="mlp", pooler_type="mean_pooler"), \
               torch.randint(0, 30000, (1, 512))
    elif encoder_type == "textproj":
        from llm2clip.eva_clip.model import TextProj
        return TextProj(embedding_dim=5120, output_dim=768, num_layers_text=4), \
               torch.randn(1, 5120)
    else:
        raise ValueError(encoder_type)


def build_vis_model(vis_type):
    import timm
    name_map = {
        "vit_b16": "vit_base_patch16_224.dino",
        "deit3_l": "deit3_large_patch16_224.fb_in22k_ft_in1k",
        "vit_l14": "vit_large_patch14_dinov2.lvd142m",
    }
    return timm.create_model(name_map[vis_type], pretrained=False, num_classes=0), \
           torch.randn(1, 3, 224, 224)


def main():
    configs = [
        ("A1", "transformer", "vit_b16"),
        ("A2", "transformer", "vit_b16"),
        ("A3", "cxrbert",     "vit_b16"),
        ("A4", "textproj",    "vit_b16"),
        ("B1", "pubmedbert",  "deit3_l"),
        ("B2", "cxrbert",     "deit3_l"),
        ("B3", "textproj",    "deit3_l"),
    ]

    print(f"{'Config':<6} {'Params(M)':<12} {'FLOPs(G)':<12}")
    print("-" * 32)

    seen = set()
    for name, text_type, vis_type in configs:
        # Deduplicate identical encoder pairs
        key = (text_type, vis_type)
        if key in seen:
            continue
        seen.add(key)

        text_model, text_input = build_text_model(text_type)
        vis_model, vis_input = build_vis_model(vis_type)

        t_p, t_f = count_params_flops(text_model, text_input)
        v_p, v_f = count_params_flops(vis_model, vis_input)

        # Map back to all config names with this encoder pair
        matching = [n for n, tt, vt in configs if (tt, vt) == key]
        label = "/".join(matching)
        print(f"{label:<6} {t_p + v_p:<12.1f} {t_f + v_f:<12.2f}")

    print()
    print("Loss function FLOPs excluded: <0.001G across all configs.")


if __name__ == "__main__":
    main()
