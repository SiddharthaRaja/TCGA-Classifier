#!/usr/bin/env python3
"""
TCGA-BRCA WSI Preprocessing — Patch Feature Extraction
========================================================
Pipeline:
  1. Open SVS file with OpenSlide
  2. Detect tissue regions (Otsu thresholding on thumbnail)
  3. Tile tissue into 512x512 patches at 20x magnification
  4. Extract features from each patch using ResNet50 (timm, ImageNet pretrained)
  5. Mean-pool all patch features → 1 vector per slide (1 x 2048)
  6. Save per-patient feature CSV + XLSX

Output per patient:
  - wsi_features/<patient_id>_patch_features.npy  (raw patch features, N x 2048)
  - wsi_preprocessed.csv                          (one row per patient, 2048 features)
  - wsi_preprocessed.xlsx                         (formatted)

Requirements:
  pip install openslide-python opencv-python torch torchvision timm Pillow numpy pandas openpyxl tqdm

Windows note:
  OpenSlide requires the C binaries. Download from:
  https://openslide.org/api/python/#installing
  Then add the bin folder to PATH, or set OPENSLIDE_PATH in this script.

Estimated time (CPU, 16GB RAM):
  ~10-20 min per slide, ~40-80 min total for 4 patients
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
import cv2
from PIL import Image
from tqdm import tqdm
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ── Windows: set OpenSlide binary path if needed ──────────────────────────────
# Download from https://openslide.org/api/python/#installing
# Unzip and point this to the 'bin' folder, e.g.:
# OPENSLIDE_PATH = r"C:\openslide-win64\bin"
OPENSLIDE_PATH = None

if OPENSLIDE_PATH:
    os.add_dll_directory(OPENSLIDE_PATH)

import openslide

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR         = Path("TCGA_BRCA_data")
OUTPUT_DIR       = Path("wsi_features")
OUTPUT_CSV       = Path("wsi_preprocessed.csv")
OUTPUT_XLSX      = Path("wsi_preprocessed.xlsx")

PATCH_SIZE       = 512        # pixels at target magnification
TARGET_MAG       = 20         # target magnification (20x)
TISSUE_THRESHOLD = 0.5        # min tissue fraction to keep a patch
BATCH_SIZE       = 16         # patches per forward pass (reduce if OOM)
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


# ── Step 1: Find WSI files ─────────────────────────────────────────────────────
def find_wsi_files(data_dir: Path):
    patients = []
    for patient_dir in sorted(data_dir.iterdir()):
        wsi_dir = patient_dir / "WSI"
        if not wsi_dir.exists():
            continue
        svs_files = list(wsi_dir.glob("*.svs"))
        if not svs_files:
            print(f"    ⚠ No SVS files found for {patient_dir.name}")
            continue
        # Take the first SVS (diagnostic slide)
        patients.append({
            "patient_id": patient_dir.name,
            "svs_path"  : svs_files[0]
        })
        print(f"    {patient_dir.name} → {svs_files[0].name}")
    return patients


# ── Step 2: Get best level for target magnification ───────────────────────────
def get_level_for_mag(slide, target_mag=20):
    """
    Find the slide level closest to the target magnification.
    Falls back to level 0 if magnification metadata is missing.
    """
    try:
        native_mag = float(slide.properties.get(
            openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
        downsample = native_mag / target_mag
        level = slide.get_best_level_for_downsample(downsample)
    except Exception:
        level = 0
    return level


# ── Step 3: Tissue detection ──────────────────────────────────────────────────
def get_tissue_mask(slide, level=-1, scale=32):
    """
    Returns a binary tissue mask at a downsampled resolution.
    Uses Otsu thresholding on the saturation channel (HSV).
    """
    # Use the lowest resolution level for fast thumbnail
    thumb_level = slide.level_count - 1
    thumb_size  = slide.level_dimensions[thumb_level]
    thumb = slide.read_region((0, 0), thumb_level, thumb_size).convert("RGB")
    thumb_np = np.array(thumb)

    # Convert to HSV, threshold on saturation
    hsv   = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    sat   = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return mask, thumb_level, thumb_size


# ── Step 4: Generate patch coordinates ────────────────────────────────────────
def get_patch_coords(slide, level, patch_size, tissue_threshold):
    """
    Returns list of (x, y) top-left coordinates (at level 0) for patches
    that contain sufficient tissue.
    """
    mask, thumb_level, thumb_size = get_tissue_mask(slide)

    # Scale factor between level 0 and thumbnail
    w0, h0   = slide.level_dimensions[0]
    wt, ht   = thumb_size
    scale_x  = w0 / wt
    scale_y  = h0 / ht

    # Patch size at level 0
    level_ds      = slide.level_downsamples[level]
    patch_size_l0 = int(patch_size * level_ds)

    coords = []
    for y in range(0, h0 - patch_size_l0, patch_size_l0):
        for x in range(0, w0 - patch_size_l0, patch_size_l0):
            # Map to thumbnail coordinates
            yt = int(y / scale_y)
            xt = int(x / scale_x)
            pt = int(patch_size_l0 / scale_x)
            ph = int(patch_size_l0 / scale_y)

            # Clamp
            yt2 = min(yt + ph, mask.shape[0])
            xt2 = min(xt + pt, mask.shape[1])

            if yt2 <= yt or xt2 <= xt:
                continue

            region = mask[yt:yt2, xt:xt2]
            tissue_frac = region.sum() / (region.size * 255 + 1e-6)

            if tissue_frac >= tissue_threshold:
                coords.append((x, y))

    return coords


# ── Step 5: Load pretrained ResNet50 ──────────────────────────────────────────
def load_model(device):
    """
    Loads ResNet50 from timm with ImageNet pretrained weights.
    Removes the classification head to get 2048-dim feature vectors.
    """
    print(f"    Loading ResNet50 (timm) on {device.upper()} ...")
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=0   # removes classifier head → outputs 2048-dim features
    )
    model.eval()
    model = model.to(device)
    return model


def get_transforms():
    """ImageNet normalisation transforms."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225]
        )
    ])


# ── Step 6: Extract features ──────────────────────────────────────────────────
def extract_features(slide, level, coords, model, transform, device,
                     patch_size, batch_size=16):
    """
    Extracts ResNet50 features for all tissue patches.
    Returns array of shape (N_patches, 2048).
    """
    all_features = []

    for i in tqdm(range(0, len(coords), batch_size),
                  desc="        Extracting patches", leave=False):
        batch_coords = coords[i:i + batch_size]
        batch_imgs   = []

        for (x, y) in batch_coords:
            try:
                patch = slide.read_region(
                    (x, y), level, (patch_size, patch_size)
                ).convert("RGB")
                batch_imgs.append(transform(patch))
            except Exception:
                # Skip unreadable patches
                continue

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model(batch_tensor).cpu().numpy()
        all_features.append(feats)

    if not all_features:
        return np.zeros((1, 2048))

    return np.vstack(all_features)


# ── Step 7: Process one patient ───────────────────────────────────────────────
def process_patient(patient, model, transform, device):
    pid      = patient["patient_id"]
    svs_path = patient["svs_path"]

    print(f"\n  Processing {pid}")
    print(f"    File: {svs_path.name}")

    slide = openslide.OpenSlide(str(svs_path))
    dims  = slide.level_dimensions[0]
    print(f"    Dimensions (level 0): {dims[0]} x {dims[1]}")

    # Get level for 20x
    level = get_level_for_mag(slide, TARGET_MAG)
    print(f"    Using level {level} for ~{TARGET_MAG}x magnification")

    # Get tissue patch coordinates
    print(f"    Detecting tissue and generating {PATCH_SIZE}x{PATCH_SIZE} patches ...")
    coords = get_patch_coords(slide, level, PATCH_SIZE, TISSUE_THRESHOLD)
    print(f"    Found {len(coords)} tissue patches")

    if len(coords) == 0:
        print(f"    ⚠ No tissue patches found — skipping")
        slide.close()
        return None

    # Extract features
    print(f"    Extracting features (ResNet50, batch={BATCH_SIZE}, device={DEVICE}) ...")
    features = extract_features(
        slide, level, coords, model, transform, device,
        PATCH_SIZE, BATCH_SIZE
    )
    slide.close()

    print(f"    Patch features shape: {features.shape}")

    # Save raw patch features
    OUTPUT_DIR.mkdir(exist_ok=True)
    npy_path = OUTPUT_DIR / f"{pid}_patch_features.npy"
    np.save(str(npy_path), features)
    print(f"    Raw features saved → {npy_path}")

    # Mean pool → slide-level vector
    slide_vector = features.mean(axis=0)
    print(f"    Slide vector shape after mean pooling: {slide_vector.shape}")

    # Build row dict
    rec = {"patient_id": pid}
    for i, val in enumerate(slide_vector):
        rec[f"wsi_feat_{i}"] = float(val)

    # Add metadata
    rec["n_patches"]  = len(coords)
    rec["slide_w"]    = dims[0]
    rec["slide_h"]    = dims[1]

    return rec


# ── Step 8: Write XLSX ─────────────────────────────────────────────────────────
def write_xlsx(df, path):
    n_feat_cols = len([c for c in df.columns if c.startswith("wsi_feat_")])
    summary_rows = [
        ("Total patients",       len(df)),
        ("Feature dimensions",   n_feat_cols),
        ("Model",                "ResNet50 (ImageNet pretrained, timm)"),
        ("Patch size",           f"{PATCH_SIZE}x{PATCH_SIZE} px"),
        ("Magnification",        f"{TARGET_MAG}x"),
        ("Aggregation",          "Mean pooling across all tissue patches"),
        ("Device used",          DEVICE.upper()),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    # Only write metadata + first 50 feature cols to XLSX (too wide otherwise)
    meta_cols = ["patient_id", "n_patches", "slide_w", "slide_h"]
    feat_cols = [c for c in df.columns if c.startswith("wsi_feat_")][:50]
    df_xlsx   = df[meta_cols + feat_cols].copy()

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df_xlsx.to_excel(writer, sheet_name="WSI Features (first 50)", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    wb = load_workbook(str(path))
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    header_fill = PatternFill("solid", start_color="1F4E79")
    alt_fill    = PatternFill("solid", start_color="FCE4D6")  # orange tint for WSI
    center      = Alignment(horizontal="center", vertical="center")
    thin        = Side(style="thin", color="CCCCCC")
    border      = Border(left=thin, right=thin, top=thin, bottom=thin)

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = center
            cell.border = border
        for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
            fill = alt_fill if row_idx % 2 == 0 else PatternFill()
            for cell in row:
                cell.fill = fill
                cell.alignment = center
                cell.border = border
        for col in ws.columns:
            max_len = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 30)
        ws.freeze_panes = "A2"
    wb.save(str(path))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  TCGA-BRCA WSI Preprocessing — ResNet50 Feature Extraction")
    print("=" * 65)
    print(f"\n  Device: {DEVICE.upper()}")
    print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"  Magnification: {TARGET_MAG}x")
    print(f"  Aggregation: Mean pooling")

    # 1. Find WSI files
    print("\n[1/4] Finding SVS files ...")
    patients = find_wsi_files(DATA_DIR)
    if not patients:
        print("  ERROR: No SVS files found.")
        print("  Make sure tcga_brca_downloader.py ran with DOWNLOAD_WSI=True")
        return
    print(f"    Found {len(patients)} patient(s).")

    # 2. Load model
    print("\n[2/4] Loading feature extractor ...")
    model     = load_model(DEVICE)
    transform = get_transforms()
    print(f"    ResNet50 loaded — output dim: 2048")

    # 3. Process each patient
    print("\n[3/4] Processing slides ...")
    records = []
    for patient in patients:
        try:
            rec = process_patient(patient, model, transform, DEVICE)
            if rec is not None:
                records.append(rec)
        except Exception as e:
            print(f"    ✗ {patient['patient_id']}: {e}")

    if not records:
        print("\n  No slides processed successfully.")
        return

    # 4. Export
    print("\n[4/4] Exporting outputs ...")
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"    CSV  saved → {OUTPUT_CSV.resolve()}")
    print(f"    Shape: {df.shape}  (patients x features)")
    write_xlsx(df, OUTPUT_XLSX)
    print(f"    XLSX saved → {OUTPUT_XLSX.resolve()}")
    print(f"    Raw .npy files → {OUTPUT_DIR.resolve()}/")

    print("\n" + "=" * 65)
    print(f"  Done! {len(df)} patient(s) x {len(df.columns)} columns")
    print(f"  (2048 WSI features + 3 metadata cols per patient)")
    print("=" * 65)

    print("\n  Next steps:")
    print("  1. Run merge_modalities.py to combine WSI + Clinical + CNV + Biospecimen")
    print("  2. Or re-run on GPU by setting DEVICE='cuda' at the top of this script")


if __name__ == "__main__":
    main()
