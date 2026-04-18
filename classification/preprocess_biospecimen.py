#!/usr/bin/env python3
"""
TCGA-BRCA Biospecimen Preprocessing
=====================================
Parses the patient-specific biospecimen XML files:
  nationwidechildrens.org_biospecimen.TCGA-*.xml

Ignores the global *_brca.txt files (same file copied to every patient folder).

Extracts a flat feature row per patient covering:
  Patient level  : tissue_source_site, gender, days_to_index
  Sample level   : sample counts by type (tumor/normal), initial weights,
                   oct_embedded, sample_type_ids
  Portion level  : portion counts, weight, is_ffpe
  Analyte level  : analyte type counts (DNA/RNA/protein), concentration,
                   spectrophotometer_method
  Aliquot level  : aliquot counts, concentration stats, volume stats,
                   plate info

Outputs:
  - biospecimen_preprocessed.csv
  - biospecimen_preprocessed.xlsx

Requirements:
  pip install pandas openpyxl scikit-learn lxml
"""

import warnings
from pathlib import Path

import pandas as pd
from lxml import etree
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR          = Path("TCGA_BRCA_data")
OUTPUT_CSV        = Path("biospecimen_preprocessed.csv")
OUTPUT_XLSX       = Path("biospecimen_preprocessed.xlsx")
MISSING_THRESHOLD = 0.60
# ──────────────────────────────────────────────────────────────────────────────

_NOT_AVAILABLE = {
    "[not available]", "[not applicable]", "[unknown]", "not available",
    "not applicable", "unknown", "na", "n/a", "null", "none", ""
}

def _tag(el):
    """Return local tag name without namespace."""
    return el.tag.split("}")[-1] if "}" in el.tag else el.tag

def _find(root, *local_names):
    """Find first element matching any local tag name, return stripped text."""
    for el in root.iter():
        if _tag(el) in local_names:
            if el.text and el.text.strip():
                return el.text.strip()
    return None

def _find_all(root, local_name):
    """Return list of all elements matching local tag name."""
    return [el for el in root.iter() if _tag(el) == local_name]

def _clean(val):
    if val is None:
        return None
    v = str(val).strip().lower()
    return None if v in _NOT_AVAILABLE else str(val).strip()

def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def find_biospecimen_xmls(data_dir: Path):
    """Return only the patient-specific biospecimen XMLs (not the global brca.txt files)."""
    found = []
    for xml_path in sorted(data_dir.glob("*/Biospecimen/*.xml")):
        name = xml_path.name.lower()
        if "biospecimen." in name and "ssf" not in name:
            found.append(xml_path)
    return found


def parse_biospecimen_xml(xml_path: Path) -> dict:
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    patient_id = xml_path.parent.parent.name
    rec = {"patient_id": patient_id}

    # ── Patient level ─────────────────────────────────────────────────────────
    rec["tissue_source_site"] = _clean(_find(root, "tissue_source_site"))
    rec["gender"]             = _clean(_find(root, "gender"))
    rec["days_to_index"]      = _to_float(_find(root, "days_to_index"))

    # ── Sample level ──────────────────────────────────────────────────────────
    samples = _find_all(root, "sample")
    rec["n_samples_total"] = len(samples)

    sample_types = [_clean(_find(s, "sample_type")) for s in samples]
    rec["n_samples_tumor"]  = sum(1 for t in sample_types if t and "tumor" in t.lower())
    rec["n_samples_normal"] = sum(1 for t in sample_types if t and "normal" in t.lower())

    # Initial weights across all tumor samples
    tumor_samples = [s for s, t in zip(samples, sample_types)
                     if t and "tumor" in t.lower()]
    weights = [_to_float(_find(s, "initial_weight")) for s in tumor_samples]
    weights = [w for w in weights if w is not None]
    rec["tumor_initial_weight_mean"] = round(sum(weights) / len(weights), 2) if weights else None
    rec["tumor_initial_weight_max"]  = max(weights) if weights else None

    # OCT embedded (any tumor sample)
    oct_vals = [_clean(_find(s, "oct_embedded")) for s in tumor_samples]
    rec["oct_embedded"] = 1 if any(v and v.lower() == "true" for v in oct_vals) else 0

    # Sample type IDs present (e.g. 01=Primary Tumor, 10=Blood Normal, 11=Solid Normal)
    type_ids = [_clean(_find(s, "sample_type_id")) for s in samples]
    rec["has_primary_tumor"]  = 1 if "01" in type_ids else 0
    rec["has_solid_normal"]   = 1 if "11" in type_ids else 0
    rec["has_blood_normal"]   = 1 if "10" in type_ids else 0
    rec["has_metastatic"]     = 1 if "06" in type_ids else 0

    # ── Portion level ─────────────────────────────────────────────────────────
    portions = _find_all(root, "portion")
    rec["n_portions"] = len(portions)

    portion_weights = [_to_float(_find(p, "weight")) for p in portions]
    portion_weights = [w for w in portion_weights if w is not None]
    rec["portion_weight_mean"] = round(sum(portion_weights) / len(portion_weights), 4) \
        if portion_weights else None

    ffpe_vals = [_clean(_find(p, "is_ffpe")) for p in portions]
    rec["has_ffpe"] = 1 if any(v and v.lower() == "true" for v in ffpe_vals) else 0

    # ── Analyte level ─────────────────────────────────────────────────────────
    analytes = _find_all(root, "analyte")
    rec["n_analytes"] = len(analytes)

    analyte_types = [_clean(_find(a, "analyte_type")) for a in analytes]
    rec["n_analytes_dna"]     = sum(1 for t in analyte_types if t and "dna" in t.lower())
    rec["n_analytes_rna"]     = sum(1 for t in analyte_types if t and "rna" in t.lower())
    rec["n_analytes_protein"] = sum(1 for t in analyte_types if t and "protein" in t.lower())

    concentrations = [_to_float(_find(a, "concentration")) for a in analytes]
    concentrations = [c for c in concentrations if c is not None]
    rec["analyte_concentration_mean"] = round(sum(concentrations) / len(concentrations), 4) \
        if concentrations else None
    rec["analyte_concentration_max"]  = max(concentrations) if concentrations else None

    spectro = [_clean(_find(a, "spectrophotometer_method")) for a in analytes]
    spectro = [s for s in spectro if s]
    rec["spectrophotometer_method"] = spectro[0] if spectro else None

    # ── Aliquot level ─────────────────────────────────────────────────────────
    aliquots = _find_all(root, "aliquot")
    rec["n_aliquots"] = len(aliquots)

    aliquot_concs = [_to_float(_find(a, "concentration")) for a in aliquots]
    aliquot_concs = [c for c in aliquot_concs if c is not None]
    rec["aliquot_concentration_mean"] = round(sum(aliquot_concs) / len(aliquot_concs), 4) \
        if aliquot_concs else None

    aliquot_vols = [_to_float(_find(a, "volume")) for a in aliquots]
    aliquot_vols = [v for v in aliquot_vols if v is not None]
    rec["aliquot_volume_mean"] = round(sum(aliquot_vols) / len(aliquot_vols), 4) \
        if aliquot_vols else None

    aliquot_amounts = [_to_float(_find(a, "amount")) for a in aliquots]
    aliquot_amounts = [a for a in aliquot_amounts if a is not None]
    rec["aliquot_amount_mean"] = round(sum(aliquot_amounts) / len(aliquot_amounts), 4) \
        if aliquot_amounts else None

    return rec


def build_dataframe(records):
    return pd.DataFrame(records)


def drop_high_missing(df):
    missing_rate = df.isna().mean()
    to_drop = missing_rate[missing_rate > MISSING_THRESHOLD].index.tolist()
    if to_drop:
        print(f"    Dropping {len(to_drop)} column(s) with >{MISSING_THRESHOLD*100:.0f}% missing:")
        for c in to_drop:
            print(f"      • {c}  ({missing_rate[c]*100:.0f}% missing)")
    return df.drop(columns=to_drop)


def onehot_encode(df):
    skip = {"patient_id"}
    obj_cols = [c for c in df.select_dtypes(include="object").columns if c not in skip]
    if not obj_cols:
        print("    No categorical columns to encode.")
        return df
    print(f"    One-hot encoding {len(obj_cols)} column(s):")
    for c in obj_cols:
        print(f"      {c}: {sorted(df[c].dropna().unique().tolist())}")
    df = pd.get_dummies(df, columns=obj_cols, dummy_na=False)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    return df


def impute_missing(df):
    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def scale_features(df):
    binary_cols = {"oct_embedded", "has_primary_tumor", "has_solid_normal",
                   "has_blood_normal", "has_metastatic", "has_ffpe"}

    def is_binary(series):
        vals = set(series.dropna().tolist())
        return vals.issubset({0, 1, 0.0, 1.0})

    scale_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in binary_cols and not is_binary(df[c])
    ]
    if not scale_cols:
        print("    No continuous columns to scale.")
        return df, None, []

    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    print(f"    Scaled {len(scale_cols)} column(s): {scale_cols}")
    return df, scaler, scale_cols


def write_xlsx(df, path):
    summary_rows = [
        ("Total patients",    len(df)),
        ("Total features",    len(df.columns)),
        ("Avg missing rate",  f"{df.isna().mean().mean()*100:.1f}%"),
        ("Numeric features",  len(df.select_dtypes(include="number").columns)),
        ("Source",            "Patient-specific biospecimen XML only"),
        ("Ignored files",     "*_brca.txt (global cohort files)"),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])

    with pd.ExcelWriter(str(path), engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Biospecimen Features", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    wb = load_workbook(str(path))
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    header_fill = PatternFill("solid", start_color="1F4E79")
    alt_fill    = PatternFill("solid", start_color="E2EFDA")  # green tint for biospecimen
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
            max_len = max(
                (len(str(c.value)) if c.value else 0) for c in col)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 4, 40)
        ws.freeze_panes = "A2"
    wb.save(str(path))


def main():
    print("=" * 65)
    print("  TCGA-BRCA Biospecimen Preprocessing")
    print("=" * 65)

    print("\n[1/6] Finding patient-specific biospecimen XMLs ...")
    xml_files = find_biospecimen_xmls(DATA_DIR)
    if not xml_files:
        print("  ERROR: No biospecimen XML files found.")
        return
    for xf in xml_files:
        print(f"    • {xf.name}")
    print(f"    Found {len(xml_files)} file(s).")

    print("\n[2/6] Parsing XML files ...")
    records = []
    for xf in xml_files:
        try:
            rec = parse_biospecimen_xml(xf)
            records.append(rec)
            print(f"    ✓ {rec['patient_id']} — "
                  f"{rec['n_samples_total']} samples, "
                  f"{rec['n_analytes']} analytes, "
                  f"{rec['n_aliquots']} aliquots")
        except Exception as e:
            print(f"    ✗ {xf.name}: {e}")

    if not records:
        print("  No records parsed.")
        return

    print("\n[3/6] Building DataFrame ...")
    df = build_dataframe(records)
    print(f"    Shape: {df.shape}")

    print("\n[4/6] Dropping high-missing columns ...")
    df = drop_high_missing(df)

    print("\n[5/6] One-hot encoding categoricals + imputing NaNs ...")
    df = onehot_encode(df)
    df = impute_missing(df)
    print(f"    Shape after cleaning: {df.shape}")

    print("\n[6/6] Scaling continuous features ...")
    df, scaler, scaled_cols = scale_features(df)

    print("\n  Exporting outputs ...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"    CSV  saved → {OUTPUT_CSV.resolve()}")
    write_xlsx(df, OUTPUT_XLSX)
    print(f"    XLSX saved → {OUTPUT_XLSX.resolve()}")

    print("\n" + "=" * 65)
    print("  Final column overview:")
    print("=" * 65)
    for col in df.columns:
        print(f"  {col:<45} {str(df[col].dtype):<10} missing={df[col].isna().sum()}")

    print(f"\n  Done! {len(df)} patient(s) x {len(df.columns)} features")


if __name__ == "__main__":
    main()
