#!/usr/bin/env python3
"""
TCGA-BRCA Feature Engineering
===============================
Adds engineered features on top of the merged multimodal dataset.

New features added:
  1.  receptor_subtype_code     — 3-bit integer from ER/PR/HER2 (0-7)
  2.  is_triple_negative        — 1 if ER-/PR-/HER2- (Basal-like signature)
  3.  is_her2_positive          — 1 if HER2+ regardless of ER/PR
  4.  is_luminal                — 1 if ER+ or PR+ (Luminal A/B signature)
  5.  genomic_instability_score — weighted CNV instability index
  6.  gain_loss_ratio           — ratio of gained to lost genome fraction
  7.  chr17_her2_interaction    — chr17 CNV mean * HER2 binary (HER2-enriched marker)
  8.  chr8_chr16_ratio          — chr8 gain / chr16 loss (Luminal A marker)
  9.  wsi_patch_density         — n_patches / (slide_w * slide_h) (tissue density)
  10. age_stage_interaction     — age * tumor_stage_numeric
  11. survival_age_ratio        — overall_survival_days / age_at_diagnosis

Outputs:
  - multimodal_engineered.csv
  - multimodal_engineered.xlsx

Requirements: pip install pandas openpyxl
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

INPUT_CSV  = Path("multimodal_dataset.csv")
OUTPUT_CSV = Path("multimodal_engineered.csv")
OUTPUT_XLSX= Path("multimodal_engineered.xlsx")


def safe_col(df, name):
    """Return column if it exists, else None series."""
    return df[name] if name in df.columns else None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n  Engineering features ...")
    added = []

    # ── 1. Receptor subtype code (3-bit: ER | PR | HER2) ─────────────────────
    er  = df.get("clin_er_status_binary",  pd.Series([0]*len(df)))
    pr  = df.get("clin_pr_status_binary",  pd.Series([0]*len(df)))
    her2= df.get("clin_her2_status_binary",pd.Series([0]*len(df)))

    df["eng_receptor_subtype_code"] = (
        er.fillna(0).astype(int) * 4 +
        pr.fillna(0).astype(int) * 2 +
        her2.fillna(0).astype(int) * 1
    )
    added.append("eng_receptor_subtype_code")

    # ── 2. Triple negative flag ───────────────────────────────────────────────
    df["eng_is_triple_negative"] = (
        (er.fillna(0) == 0) & (pr.fillna(0) == 0) & (her2.fillna(0) == 0)
    ).astype(int)
    added.append("eng_is_triple_negative")

    # ── 3. HER2 positive flag ─────────────────────────────────────────────────
    df["eng_is_her2_positive"] = (her2.fillna(0) == 1).astype(int)
    added.append("eng_is_her2_positive")

    # ── 4. Luminal flag (ER+ or PR+) ─────────────────────────────────────────
    df["eng_is_luminal"] = (
        (er.fillna(0) == 1) | (pr.fillna(0) == 1)
    ).astype(int)
    added.append("eng_is_luminal")

    # ── 5. Genomic instability score ──────────────────────────────────────────
    # Weighted combination of CNV instability features
    feats = {
        "cnv_n_segments"         : 0.15,
        "cnv_n_gain_segments"    : 0.15,
        "cnv_n_loss_segments"    : 0.15,
        "cnv_std_segment_mean"   : 0.25,
        "cnv_n_chromosomes_affected": 0.30,
    }
    score = pd.Series([0.0] * len(df), index=df.index)
    total_weight = 0
    for col, w in feats.items():
        if col in df.columns:
            score += df[col].fillna(0) * w
            total_weight += w
    if total_weight > 0:
        score = score / total_weight
    df["eng_genomic_instability_score"] = score
    added.append("eng_genomic_instability_score")

    # ── 6. Gain/loss ratio ────────────────────────────────────────────────────
    gained = df.get("cnv_pct_genome_gained", pd.Series([0]*len(df))).fillna(0)
    lost   = df.get("cnv_pct_genome_lost",   pd.Series([0]*len(df))).fillna(0)
    df["eng_gain_loss_ratio"] = gained / (lost + 1e-6)
    added.append("eng_gain_loss_ratio")

    # ── 7. Chr17 x HER2 interaction (HER2-enriched signature) ────────────────
    chr17 = df.get("cnv_chr17_mean", pd.Series([0]*len(df))).fillna(0)
    df["eng_chr17_her2_interaction"] = chr17 * her2.fillna(0)
    added.append("eng_chr17_her2_interaction")

    # ── 8. Chr8 / Chr16 ratio (Luminal A signature) ───────────────────────────
    chr8  = df.get("cnv_chr8_mean",  pd.Series([0]*len(df))).fillna(0)
    chr16 = df.get("cnv_chr16_mean", pd.Series([0]*len(df))).fillna(0)
    df["eng_chr8_chr16_ratio"] = chr8 / (chr16.abs() + 1e-6)
    added.append("eng_chr8_chr16_ratio")

    # ── 9. WSI patch density (tissue coverage proxy) ──────────────────────────
    n_patches = df.get("wsi_n_patches", pd.Series([0]*len(df))).fillna(0)
    slide_w   = df.get("wsi_slide_w",   pd.Series([1]*len(df))).fillna(1)
    slide_h   = df.get("wsi_slide_h",   pd.Series([1]*len(df))).fillna(1)
    total_px  = slide_w * slide_h
    df["eng_wsi_patch_density"] = n_patches / (total_px / 1e8 + 1e-6)
    added.append("eng_wsi_patch_density")

    # ── 10. Age x Stage interaction ───────────────────────────────────────────
    age   = df.get("clin_age_at_diagnosis",   pd.Series([0]*len(df))).fillna(0)
    stage = df.get("clin_tumor_stage_numeric", pd.Series([0]*len(df))).fillna(0)
    df["eng_age_stage_interaction"] = age * stage
    added.append("eng_age_stage_interaction")

    # ── 11. Survival / age ratio ──────────────────────────────────────────────
    survival = df.get("clin_overall_survival_days", pd.Series([0]*len(df))).fillna(0)
    df["eng_survival_age_ratio"] = survival / (age.abs() + 1e-6)
    added.append("eng_survival_age_ratio")

    print(f"  Added {len(added)} engineered features:")
    for f in added:
        print(f"    + {f}")

    return df, added


def write_xlsx(df, path, engineered_cols):
    summary_rows = [
        ("Total patients",          len(df)),
        ("Total features",          len(df.columns) - 1),
        ("Clinical features",       len([c for c in df.columns if c.startswith("clin_")])),
        ("CNV features",            len([c for c in df.columns if c.startswith("cnv_")])),
        ("Biospecimen features",    len([c for c in df.columns if c.startswith("bio_")])),
        ("WSI features",            len([c for c in df.columns if c.startswith("wsi_")])),
        ("Engineered features",     len(engineered_cols)),
        ("Missing values",          int(df.isna().sum().sum())),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Engineered Dataset", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    wb   = load_workbook(str(path))
    thin = Side(style="thin", color="CCCCCC")
    bdr  = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Colour map per modality prefix
    colour_map = {
        "clin_": "D6E4F0",
        "cnv_":  "FFF2CC",
        "bio_":  "E2EFDA",
        "wsi_":  "FCE4D6",
        "eng_":  "EDE7F6",  # purple tint for engineered
    }

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        for cell in ws[1]:
            cell.font      = Font(bold=True, color="FFFFFF", name="Arial", size=10)
            cell.fill      = PatternFill("solid", start_color="1F4E79")
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = bdr

        for ri, row in enumerate(ws.iter_rows(min_row=2), start=2):
            for cell in row:
                col_name = ws.cell(1, cell.column).value or ""
                fill_hex  = next(
                    (v for k, v in colour_map.items() if col_name.startswith(k)),
                    "F5F5F5"
                )
                cell.fill      = PatternFill("solid", start_color=fill_hex) \
                                  if ri % 2 == 0 else PatternFill()
                cell.alignment = Alignment(horizontal="center", vertical="center")
                cell.border    = bdr
                cell.font      = Font(name="Arial", size=9)

        for col in ws.columns:
            ml = max((len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(ml + 4, 40)
        ws.freeze_panes = "A2"

    wb.save(str(path))


def main():
    print("=" * 60)
    print("  TCGA-BRCA Feature Engineering")
    print("=" * 60)

    if not INPUT_CSV.exists():
        print(f"  ERROR: {INPUT_CSV} not found. Run merge_modalities.py first.")
        return

    df = pd.read_csv(INPUT_CSV)
    print(f"\n  Input : {df.shape[0]} patients x {df.shape[1]} columns")

    df, engineered_cols = engineer_features(df)

    print(f"\n  Output: {df.shape[0]} patients x {df.shape[1]} columns")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  CSV  saved → {OUTPUT_CSV.resolve()}")
    write_xlsx(df, OUTPUT_XLSX, engineered_cols)
    print(f"  XLSX saved → {OUTPUT_XLSX.resolve()}")

    print("\n" + "=" * 60)
    print("  Feature summary:")
    print("=" * 60)
    prefixes = {"clin_": "Clinical", "cnv_": "CNV", "bio_": "Biospecimen",
                "wsi_": "WSI", "eng_": "Engineered"}
    for prefix, name in prefixes.items():
        count = len([c for c in df.columns if c.startswith(prefix)])
        print(f"  {name:<15} {count} features")
    print(f"  {'patient_id':<15} 1 column")
    print(f"  {'TOTAL':<15} {len(df.columns)} columns")
    print("\n  Next step: run train_classifier.py")


if __name__ == "__main__":
    main()
