#!/usr/bin/env python3
"""
Overlay saved attention weights onto a low-res thumbnail of the original WSI —
bright = the model focused heavily here, dim = mostly ignored.

Requires --save-attention to have been used during training first.

Usage
    python generate_attention_heatmap.py --backbone densenet121 --variant gated --patient TCGA-XX-XXXX
"""

import argparse
import os
from pathlib import Path

import numpy as np
import openslide
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE = Path(os.environ.get("TCGA_SCRATCH_ROOT", str(Path.home() / "siddhartha" / "data")))
RAW_DIR = BASE / "raw"
ATTENTION_ROOT = BASE / "attention_weights"
OUT_ROOT = BASE / "heatmaps"

PATCH_SIZE = 256  # must match the patch size used during extraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--variant", required=True, choices=["gated", "plain", "topk"])
    ap.add_argument("--patient", required=True)
    ap.add_argument("--thumb-size", type=int, default=1024)
    args = ap.parse_args()

    attn_path = ATTENTION_ROOT / f"{args.backbone}_{args.variant}" / f"{args.patient}.npz"
    if not attn_path.exists():
        print(f"ERROR: {attn_path} not found.")
        print("Run training with --save-attention first, e.g.:")
        print(f"  python train_pooled_classifier.py --backbone {args.backbone} "
              f"--pooling attention --attention-variant {args.variant} --save-attention")
        return

    data = np.load(attn_path)
    weights = data["weights"]
    coords = data["coords"]  # (N, 3): slide_idx, x, y — patches may span >1 slide
    slide_files = data["slide_files"] if "slide_files" in data else None

    if slide_files is None or len(slide_files) == 0:
        # fallback for older single-slide-only saves without slide_files
        svs_files = list((RAW_DIR / args.patient / "WSI").glob("*.svs"))
        if not svs_files:
            print(f"ERROR: no WSI file found for {args.patient}.")
            return
        slide_files = [str(svs_files[0])]

    # normalize weights for visual contrast (raw attention weights can be tiny)
    w_norm = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

    n_slides = len(slide_files)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for slide_idx, slide_path in enumerate(slide_files):
        slide_mask = coords[:, 0] == slide_idx
        if not slide_mask.any():
            continue
        slide_coords = coords[slide_mask][:, 1:]  # drop slide_idx column -> (x, y)
        slide_weights = w_norm[slide_mask]

        slide = openslide.OpenSlide(str(slide_path))
        w0, h0 = slide.level_dimensions[0]
        thumb = slide.get_thumbnail((args.thumb_size, args.thumb_size))
        thumb_w, thumb_h = thumb.size
        scale_x, scale_y = thumb_w / w0, thumb_h / h0
        slide.close()

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(thumb)
        cmap = cm.get_cmap("jet")
        for (x, y), w in zip(slide_coords, slide_weights):
            tx, ty = x * scale_x, y * scale_y
            tw, th = PATCH_SIZE * scale_x, PATCH_SIZE * scale_y
            ax.add_patch(plt.Rectangle((tx, ty), tw, th, color=cmap(w), alpha=0.45, linewidth=0))
        ax.axis("off")
        slide_note = f" — slide {slide_idx+1}/{n_slides}" if n_slides > 1 else ""
        ax.set_title(f"{args.patient} — {args.backbone} ({args.variant} attention){slide_note}")

        sm = cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="attention (normalized)")

        suffix = f"_slide{slide_idx}" if n_slides > 1 else ""
        out_path = OUT_ROOT / f"{args.patient}_{args.backbone}_{args.variant}{suffix}.png"
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"-> {out_path}")


if __name__ == "__main__":
    main()
