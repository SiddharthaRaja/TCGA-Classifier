#!/usr/bin/env python3
"""
TCGA-BRCA Dimensionality Reduction
=====================================
Applies PCA to reduce high-dimensional features, especially WSI (2048 features).
Modality-wise PCA:
  - WSI features (2048) → reduced to N_WSI_COMPONENTS (default: 50)
  - CNV features (36)   → kept as-is (already low-dimensional)
  - Clinical features   → kept as-is
  - Biospecimen features→ kept as-is

Also applies overall PCA on the full feature matrix for comparison.

Outputs:
  - multimodal_reduced.csv       (PCA-reduced dataset, ready for classifier)
  - multimodal_reduced.xlsx      (formatted Excel)
  - pca_wsi_model.pkl            (fitted WSI PCA model, for inference)
  - pca_full_model.pkl           (fitted full PCA model, optional)
  - pca_variance_explained.png   (scree plot)

Requirements: pip install pandas numpy scikit-learn matplotlib openpyxl
"""

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_CSV        = Path("multimodal_dataset.csv")   # or multimodal_engineered.csv
OUTPUT_CSV       = Path("multimodal_reduced.csv")
OUTPUT_XLSX      = Path("multimodal_reduced.xlsx")
WSI_PCA_PKL      = Path("pca_wsi_model.pkl")
FULL_PCA_PKL     = Path("pca_full_model.pkl")
SCREE_PLOT       = Path("pca_variance_explained.png")

N_WSI_COMPONENTS  = 50    # WSI: 2048 → 50 components
N_FULL_COMPONENTS = 100   # Full matrix PCA (optional, see APPLY_FULL_PCA)
APPLY_FULL_PCA    = False  # Set True to also apply full-matrix PCA
LABEL_COL         = "clin_pam50_subtype"
# ──────────────────────────────────────────────────────────────────────────────


def detect_modality_columns(df):
    """Detect column groups by prefix."""
    wsi_cols  = [c for c in df.columns if c.startswith("wsi_")]
    cnv_cols  = [c for c in df.columns if c.startswith("cnv_")]
    clin_cols = [c for c in df.columns if c.startswith("clin_")
                 and c not in (LABEL_COL, "clin_pam50_encoded")]
    bio_cols  = [c for c in df.columns if c.startswith("bio_")]
    eng_cols  = [c for c in df.columns if c.startswith("eng_")]
    meta_cols = ["patient_id", LABEL_COL, "clin_pam50_encoded"]
    meta_cols = [c for c in meta_cols if c in df.columns]
    return wsi_cols, cnv_cols, clin_cols, bio_cols, eng_cols, meta_cols


def apply_wsi_pca(df, wsi_cols, n_components):
    """Apply PCA to WSI columns only."""
    if not wsi_cols:
        print("    No WSI columns found — skipping WSI PCA.")
        return df, None, []

    print(f"    WSI PCA: {len(wsi_cols)} features → {n_components} components")
    X_wsi = df[wsi_cols].values.astype(float)

    # Standardize before PCA
    scaler = StandardScaler()
    X_wsi_scaled = scaler.fit_transform(X_wsi)

    n_comp = min(n_components, X_wsi_scaled.shape[0], X_wsi_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca.fit_transform(X_wsi_scaled)

    var_explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"    Variance explained by {n_comp} WSI components: {var_explained:.1f}%")

    # Create new column names
    new_cols = [f"wsi_pca_{i+1}" for i in range(n_comp)]
    df_wsi_reduced = pd.DataFrame(X_reduced, columns=new_cols, index=df.index)

    # Drop original WSI cols, add PCA cols
    df = df.drop(columns=wsi_cols)
    df = pd.concat([df, df_wsi_reduced], axis=1)

    return df, (pca, scaler), new_cols


def apply_full_pca(df, feature_cols, n_components):
    """Apply PCA to entire feature matrix."""
    print(f"    Full PCA: {len(feature_cols)} features → {n_components} components")
    X = df[feature_cols].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(n_components, X_scaled.shape[0], X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    var_explained = pca.explained_variance_ratio_.cumsum()[-1] * 100
    print(f"    Variance explained by {n_comp} full components: {var_explained:.1f}%")

    new_cols = [f"pca_{i+1}" for i in range(n_comp)]
    df_reduced = pd.DataFrame(X_reduced, columns=new_cols, index=df.index)

    return df_reduced, (pca, scaler), new_cols


def plot_scree(pca_wsi, pca_full, output_path):
    """Plot variance explained scree plots."""
    n_plots = 2 if pca_full is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    if pca_wsi is not None:
        pca, _ = pca_wsi
        cumvar = pca.explained_variance_ratio_.cumsum() * 100
        axes[0].plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=4)
        axes[0].axhline(y=90, color="r", linestyle="--", label="90% threshold")
        axes[0].axhline(y=95, color="g", linestyle="--", label="95% threshold")
        axes[0].set_xlabel("Number of Components")
        axes[0].set_ylabel("Cumulative Variance Explained (%)")
        axes[0].set_title("WSI PCA — Scree Plot")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if pca_full is not None:
        pca, _ = pca_full
        cumvar = pca.explained_variance_ratio_.cumsum() * 100
        axes[1].plot(range(1, len(cumvar) + 1), cumvar, "g-o", markersize=4)
        axes[1].axhline(y=90, color="r", linestyle="--", label="90% threshold")
        axes[1].set_xlabel("Number of Components")
        axes[1].set_ylabel("Cumulative Variance Explained (%)")
        axes[1].set_title("Full Matrix PCA — Scree Plot")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Scree plot saved → {output_path}")


def write_xlsx(df, path):
    summary_rows = [
        ("Total patients",    len(df)),
        ("Total features",    len(df.columns) - 1),
        ("WSI PCA components", len([c for c in df.columns if c.startswith("wsi_pca_")])),
        ("Clinical features", len([c for c in df.columns if c.startswith("clin_")])),
        ("CNV features",      len([c for c in df.columns if c.startswith("cnv_")])),
        ("Biospecimen features", len([c for c in df.columns if c.startswith("bio_")])),
        ("Engineered features", len([c for c in df.columns if c.startswith("eng_")])),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Reduced Dataset", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    wb = load_workbook(str(path))
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    header_fill = PatternFill("solid", start_color="1F4E79")
    alt_fill    = PatternFill("solid", start_color="D6E4F0")
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
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 45)
        ws.freeze_panes = "A2"
    wb.save(str(path))


def main():
    print("=" * 65)
    print("  TCGA-BRCA Dimensionality Reduction")
    print("=" * 65)

    print("\n[1/5] Loading dataset ...")
    if not INPUT_CSV.exists():
        print(f"  ERROR: {INPUT_CSV} not found.")
        return
    df = pd.read_csv(INPUT_CSV)
    print(f"    Loaded {df.shape[0]} patients, {df.shape[1]} columns")

    print("\n[2/5] Detecting modality columns ...")
    wsi_cols, cnv_cols, clin_cols, bio_cols, eng_cols, meta_cols = detect_modality_columns(df)
    print(f"    WSI cols:        {len(wsi_cols)}")
    print(f"    CNV cols:        {len(cnv_cols)}")
    print(f"    Clinical cols:   {len(clin_cols)}")
    print(f"    Biospecimen cols:{len(bio_cols)}")
    print(f"    Engineered cols: {len(eng_cols)}")
    print(f"    Meta cols:       {meta_cols}")

    print("\n[3/5] Applying WSI PCA ...")
    pca_wsi_model = None
    if wsi_cols:
        df, pca_wsi_model, wsi_pca_cols = apply_wsi_pca(df, wsi_cols, N_WSI_COMPONENTS)
        with open(WSI_PCA_PKL, "wb") as f:
            pickle.dump(pca_wsi_model, f)
        print(f"    WSI PCA model saved → {WSI_PCA_PKL}")
    else:
        print("    No WSI columns — skipping.")

    print("\n[4/5] Applying full matrix PCA (optional) ...")
    pca_full_model = None
    if APPLY_FULL_PCA:
        feature_cols = [c for c in df.columns if c not in meta_cols]
        df_full_reduced, pca_full_model, full_pca_cols = apply_full_pca(df, feature_cols, N_FULL_COMPONENTS)
        meta_df = df[meta_cols]
        df = pd.concat([meta_df, df_full_reduced], axis=1)
        with open(FULL_PCA_PKL, "wb") as f:
            pickle.dump(pca_full_model, f)
        print(f"    Full PCA model saved → {FULL_PCA_PKL}")
    else:
        print("    Skipped (APPLY_FULL_PCA=False). Set to True in config to enable.")

    print("\n[5/5] Saving outputs ...")
    plot_scree(pca_wsi_model, pca_full_model, SCREE_PLOT)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"    CSV  saved → {OUTPUT_CSV.resolve()}")
    write_xlsx(df, OUTPUT_XLSX)
    print(f"    XLSX saved → {OUTPUT_XLSX.resolve()}")

    print("\n" + "=" * 65)
    print("  Summary:")
    print("=" * 65)
    wsi_pca_count = len([c for c in df.columns if c.startswith("wsi_pca_")])
    print(f"  Input features:  {df.shape[1] - 1}")
    print(f"  Output features: {df.shape[1] - 1}")
    print(f"  Patients:        {df.shape[0]}")
    if wsi_pca_count:
        print(f"  WSI reduced to:  {wsi_pca_count} PCA components (from 2048)")
    print(f"\n  Next step: run train_classifier.py using {OUTPUT_CSV}")
    print("  Update DATA_CSV in train_classifier.py to: multimodal_reduced.csv")
    print("=" * 65)


if __name__ == "__main__":
    main()
