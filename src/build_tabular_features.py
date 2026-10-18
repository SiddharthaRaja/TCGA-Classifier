#!/usr/bin/env python3
"""
Build tabular (clinical + biospecimen + CNV) features for the multimodal
ablation — compared against WSI-only, this is the "side quest" track.

Usage
    python build_tabular_features.py
"""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get("TCGA_SCRATCH_ROOT", str(Path.home() / "siddhartha" / "data")))
RAW_DIR = BASE / "raw"
OUT_PATH = BASE / "tabular_features.json"

BAD_VALUES = {"", "[not evaluated]", "[not available]", "[unknown]", "[not applicable]"}

# CNV focal loci (GRCh38 coords) — chromosomal regions with known relevance
# to breast cancer subtype (matches the original pipeline's v2 CNV features)
FOCAL_LOCI = {
    "focal_her2": ("chr17", 37_800_000, 38_200_000),
    "focal_myc": ("chr8", 127_700_000, 128_100_000),
    "focal_rb1": ("chr13", 48_900_000, 49_100_000),
    "focal_pten": ("chr10", 89_600_000, 89_900_000),
    "focal_cdh1": ("chr16", 68_700_000, 68_900_000),
}
CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX"]


def local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


# ------------------------------------------------------------------ clinical
def parse_clinical(clinical_dir: Path) -> dict:
    """Age, stage, histological type, radiation, survival — one-hot friendly raw values."""
    fields = {}
    for xml_file in clinical_dir.glob("*.xml"):
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue
        for elem in root.iter():
            name = local_name(elem.tag).lower()
            text = (elem.text or "").strip()
            if text.lower() in BAD_VALUES:
                continue
            if name == "age_at_initial_pathologic_diagnosis" and "age" not in fields:
                try:
                    fields["age"] = float(text)
                except ValueError:
                    pass
            elif name == "pathologic_stage" and "stage" not in fields:
                fields["stage"] = text
            elif name == "histological_type" and "histology" not in fields:
                fields["histology"] = text
            elif name == "radiation_therapy" and "radiation" not in fields:
                fields["radiation"] = text
            elif name == "vital_status" and "vital_status" not in fields:
                fields["vital_status"] = text
            elif name in ("days_to_death", "days_to_last_followup") and "survival_days" not in fields:
                try:
                    fields["survival_days"] = float(text)
                except ValueError:
                    pass
    return fields


# ------------------------------------------------------------------ CNV
def parse_cnv(cnv_dir: Path) -> dict:
    """Global instability stats, per-chromosome means, and focal-locus features."""
    seg_files = sorted(cnv_dir.glob("*.txt"), key=lambda f: f.stat().st_size, reverse=True)
    if not seg_files:
        return {}
    tumor_file = seg_files[0]  # larger file = tumor (more segments), per original pipeline

    segments = []
    with open(tumor_file) as fh:
        header = fh.readline().strip().split("\t")
        try:
            chr_i = header.index("Chromosome")
            start_i = header.index("Start")
            end_i = header.index("End")
            mean_i = header.index("Segment_Mean")
        except ValueError:
            return {}
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) <= max(chr_i, start_i, end_i, mean_i):
                continue
            try:
                chrom = parts[chr_i]
                if not chrom.startswith("chr"):
                    chrom = "chr" + chrom
                segments.append({
                    "chr": chrom, "start": int(parts[start_i]),
                    "end": int(parts[end_i]), "mean": float(parts[mean_i]),
                })
            except (ValueError, IndexError):
                continue

    if not segments:
        return {}

    fields = {"n_segments": len(segments)}

    lengths = np.array([s["end"] - s["start"] for s in segments])
    means = np.array([s["mean"] for s in segments])
    total_len = lengths.sum() or 1

    fields["pct_gained"] = float(lengths[means > 0.2].sum() / total_len)
    fields["pct_lost"] = float(lengths[means < -0.2].sum() / total_len)
    fields["fga"] = float(lengths[np.abs(means) > 0.2].sum() / total_len)  # fraction genome altered

    for chrom in CHROMS:
        chrom_segs = [s for s in segments if s["chr"] == chrom]
        if chrom_segs:
            c_lengths = np.array([s["end"] - s["start"] for s in chrom_segs])
            c_means = np.array([s["mean"] for s in chrom_segs])
            fields[f"chr_mean_{chrom}"] = float(np.average(c_means, weights=c_lengths))
        else:
            fields[f"chr_mean_{chrom}"] = 0.0

    for name, (chrom, lo, hi) in FOCAL_LOCI.items():
        overlapping = [s for s in segments if s["chr"] == chrom
                       and s["start"] < hi and s["end"] > lo]
        if overlapping:
            ov_lengths = np.array([min(s["end"], hi) - max(s["start"], lo) for s in overlapping])
            ov_means = np.array([s["mean"] for s in overlapping])
            fields[name] = float(np.average(ov_means, weights=ov_lengths))
        else:
            fields[name] = 0.0

    fields["arm_17q"] = fields.get("chr_mean_chr17", 0.0)  # simplified: whole-chr proxy for arm
    fields["her2_arm_asymmetry"] = fields["focal_her2"] - fields["arm_17q"]

    return fields


# ------------------------------------------------------------------ one-hot encoding
def one_hot_encode(all_patient_fields: dict) -> dict:
    """Turns categorical string fields into one-hot columns; keeps numeric fields as-is."""
    categorical_keys = {"stage", "histology", "radiation", "vital_status"}
    numeric_keys = {"age", "survival_days", "n_segments", "pct_gained", "pct_lost", "fga",
                    "arm_17q", "her2_arm_asymmetry"}
    numeric_keys |= {f"chr_mean_{c}" for c in CHROMS}
    numeric_keys |= set(FOCAL_LOCI.keys())

    categories = {k: set() for k in categorical_keys}
    for fields in all_patient_fields.values():
        for k in categorical_keys:
            if k in fields:
                categories[k].add(fields[k])

    encoded = {}
    for pid, fields in all_patient_fields.items():
        row = {}
        for k in numeric_keys:
            row[k] = fields.get(k, 0.0)
        for k in categorical_keys:
            val = fields.get(k)
            for cat_val in sorted(categories[k]):
                row[f"{k}_{cat_val}"] = 1.0 if val == cat_val else 0.0
        encoded[pid] = row
    return encoded


def main():
    if not RAW_DIR.exists():
        print(f"ERROR: {RAW_DIR} does not exist.")
        return

    patients = sorted(p for p in RAW_DIR.iterdir() if p.is_dir())
    print(f"  scanning {len(patients)} patient folders...")

    all_fields = {}
    for pdir in patients:
        fields = {}
        clinical_dir = pdir / "Clinical"
        if clinical_dir.exists():
            fields.update(parse_clinical(clinical_dir))
        cnv_dir = pdir / "Copy_Number_Variation"
        if cnv_dir.exists():
            fields.update(parse_cnv(cnv_dir))
        if fields:
            all_fields[pdir.name] = fields

    encoded = one_hot_encode(all_fields)
    n_cols = len(next(iter(encoded.values()))) if encoded else 0

    OUT_PATH.write_text(json.dumps(encoded, indent=2))
    print(f"\n  patients with tabular features: {len(encoded)}")
    print(f"  feature columns: {n_cols}")
    print(f"  -> {OUT_PATH}")


if __name__ == "__main__":
    main()
