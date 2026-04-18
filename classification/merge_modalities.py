#!/usr/bin/env python3
"""
TCGA-BRCA Multimodal Merge
===========================
Merges Clinical + CNV + Biospecimen preprocessed CSVs on patient_id.
Optionally merges WSI features if wsi_preprocessed.csv exists.

Outputs:
  - multimodal_dataset.csv
  - multimodal_dataset.xlsx

Requirements:
  pip install pandas openpyxl
"""

import warnings
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
CLINICAL_CSV     = Path("clinical_final.csv")
CNV_CSV          = Path("cnv_preprocessed.csv")
BIOSPECIMEN_CSV  = Path("biospecimen_preprocessed.csv")
WSI_CSV          = Path("wsi_preprocessed.csv")       # merged if exists
OUTPUT_CSV       = Path("multimodal_dataset.csv")
OUTPUT_XLSX      = Path("multimodal_dataset.xlsx")
# ──────────────────────────────────────────────────────────────────────────────


def load_and_prefix(path: Path, prefix: str, skip_cols: list) -> pd.DataFrame:
    """
    Load a CSV and add a prefix to all feature columns
    (keeps patient_id and any skip_cols unprefixed).
    """
    df = pd.read_csv(path)
    rename = {
        c: f"{prefix}_{c}"
        for c in df.columns
        if c not in skip_cols
    }
    df = df.rename(columns=rename)
    print(f"    {path.name}: {len(df)} patients, {len(df.columns)-1} features")
    return df


def write_xlsx(df: pd.DataFrame, path: Path, modality_ranges: dict):
    """Write formatted XLSX with colour-coded modality columns."""
    # Modality colours
    colours = {
        "clinical":    "D6E4F0",  # blue
        "cnv":         "FFF2CC",  # yellow
        "biospecimen": "E2EFDA",  # green
        "wsi":         "FCE4D6",  # orange
        "shared":      "F2F2F2",  # grey for patient_id
    }

    summary_rows = [
        ("Total patients",         len(df)),
        ("Total features",         len(df.columns) - 1),
        ("Clinical features",      modality_ranges.get("clinical", 0)),
        ("CNV features",           modality_ranges.get("cnv", 0)),
        ("Biospecimen features",   modality_ranges.get("biospecimen", 0)),
        ("WSI features",           modality_ranges.get("wsi", 0)),
        ("Missing values (total)", int(df.isna().sum().sum())),
        ("PAM50 label status",     "Pending — scale to 50+ patients"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Multimodal Dataset", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    wb = load_workbook(str(path))
    ws = wb["Multimodal Dataset"]

    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    header_fill = PatternFill("solid", start_color="1F4E79")
    center      = Alignment(horizontal="center", vertical="center")
    thin        = Side(style="thin", color="CCCCCC")
    border      = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Header row
    for cell in ws[1]:
        cell.font      = header_font
        cell.fill      = header_fill
        cell.alignment = center
        cell.border    = border

    # Colour-code data rows by modality
    col_colours = {}
    for col in df.columns:
        if col == "patient_id":
            col_colours[col] = colours["shared"]
        elif col.startswith("clin_"):
            col_colours[col] = colours["clinical"]
        elif col.startswith("cnv_"):
            col_colours[col] = colours["cnv"]
        elif col.startswith("bio_"):
            col_colours[col] = colours["biospecimen"]
        elif col.startswith("wsi_"):
            col_colours[col] = colours["wsi"]
        else:
            col_colours[col] = colours["shared"]

    for row_idx, row in enumerate(ws.iter_rows(min_row=2), start=2):
        for cell in row:
            col_name  = ws.cell(1, cell.column).value or ""
            hex_color = col_colours.get(col_name, "FFFFFF")
            cell.fill      = PatternFill("solid", start_color=hex_color)
            cell.alignment = center
            cell.border    = border

    for col in ws.columns:
        max_len = max((len(str(c.value)) if c.value else 0) for c in col)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 40)
    ws.freeze_panes = "B2"

    # Format summary sheet
    ws2 = wb["Summary"]
    for cell in ws2[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center
        cell.border = border
    for row in ws2.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = center
            cell.border = border
    for col in ws2.columns:
        max_len = max((len(str(c.value)) if c.value else 0) for c in col)
        ws2.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 40)

    wb.save(str(path))


def main():
    print("=" * 65)
    print("  TCGA-BRCA Multimodal Merge")
    print("=" * 65)

    # ── Load all modalities ───────────────────────────────────────────────────
    print("\n[1/4] Loading modality CSVs ...")

    missing = [f for f in [CLINICAL_CSV, CNV_CSV, BIOSPECIMEN_CSV] if not f.exists()]
    if missing:
        for f in missing:
            print(f"  ERROR: {f} not found.")
        return

    clinical    = load_and_prefix(CLINICAL_CSV,    "clin", ["patient_id", "pam50_subtype", "pam50_encoded"])
    cnv         = load_and_prefix(CNV_CSV,         "cnv",  ["patient_id"])
    biospecimen = load_and_prefix(BIOSPECIMEN_CSV, "bio",  ["patient_id"])

    modality_ranges = {
        "clinical":    len(clinical.columns) - 1,
        "cnv":         len(cnv.columns) - 1,
        "biospecimen": len(biospecimen.columns) - 1,
        "wsi":         0,
    }

    # ── Merge on patient_id ───────────────────────────────────────────────────
    print("\n[2/4] Merging on patient_id ...")
    df = clinical.merge(cnv,         on="patient_id", how="inner")
    df = df.merge(biospecimen,       on="patient_id", how="inner")

    print(f"    After Clinical + CNV merge:         {df.shape}")

    # ── Optionally merge WSI ──────────────────────────────────────────────────
    if WSI_CSV.exists():
        print(f"    WSI file found — merging ...")
        wsi = load_and_prefix(WSI_CSV, "wsi", ["patient_id"])
        df  = df.merge(wsi, on="patient_id", how="left")
        modality_ranges["wsi"] = len(wsi.columns) - 1
        print(f"    After WSI merge: {df.shape}")
    else:
        print(f"    WSI file not found — skipping (add later)")

    print(f"\n    Final shape: {df.shape[0]} patients x {df.shape[1]} columns")

    # ── Check for missing values ──────────────────────────────────────────────
    print("\n[3/4] Checking data quality ...")
    total_missing = df.isna().sum().sum()
    print(f"    Total missing values: {total_missing}")
    if total_missing > 0:
        missing_cols = df.columns[df.isna().any()].tolist()
        print(f"    Columns with missing values: {missing_cols}")
        # Fill remaining NaNs with 0
        df = df.fillna(0)
        print(f"    Filled remaining NaNs with 0")

    # ── Export ────────────────────────────────────────────────────────────────
    print("\n[4/4] Exporting outputs ...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"    CSV  saved → {OUTPUT_CSV.resolve()}")
    write_xlsx(df, OUTPUT_XLSX, modality_ranges)
    print(f"    XLSX saved → {OUTPUT_XLSX.resolve()}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Dataset Summary:")
    print("=" * 65)
    print(f"  Patients  : {df.shape[0]}")
    print(f"  Total features : {df.shape[1] - 1}")
    print(f"  ├─ Clinical    : {modality_ranges['clinical']} features  (blue in XLSX)")
    print(f"  ├─ CNV         : {modality_ranges['cnv']} features  (yellow in XLSX)")
    print(f"  ├─ Biospecimen : {modality_ranges['biospecimen']} features  (green in XLSX)")
    print(f"  └─ WSI         : {modality_ranges['wsi']} features  (orange — add later)")
    print(f"\n  PAM50 label    : clin_pam50_subtype (currently NaN)")
    print(f"  Label encoder  : label_encoder.pkl")
    print(f"\n  Next step: scale to 50+ patients to get PAM50 labels,")
    print(f"  then run train_classifier.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
