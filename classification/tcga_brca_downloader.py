#!/usr/bin/env python3
"""
TCGA-BRCA Multi-Modality Downloader
====================================
Downloads data for 5 random TCGA-BRCA patients across 4 modalities:
  1. Biospecimen  (XML supplement → patient-specific)
  2. Clinical     (XML supplement → patient-specific)
  3. Copy Number Variation  (TSV segment files, one per patient)
  4. WSI          (Whole Slide Images – Diagnostic Slide SVS, one per patient)

Folder structure produced:
  TCGA_BRCA_data/
  └── <patient_id>/
      ├── Biospecimen/
      │   └── *.xml
      ├── Clinical/
      │   └── *.xml
      ├── Copy_Number_Variation/
      │   └── *.txt / *.tsv
      └── WSI/
          └── *.svs

Requirements:
  pip install requests tqdm

Notes:
  • All data downloaded here is **open-access** – no GDC token needed.
  • WSI (SVS) files are large (hundreds of MB–several GB each).
    Set DOWNLOAD_WSI = False to skip if you only want metadata.
  • CNV: uses "Masked Copy Number Segment" (open-access, per-aliquot TXT).
  • Biospecimen & Clinical: per-case XML supplements (already patient-scoped).
"""

import os
import json
import random
import requests
import sys
from pathlib import Path
from tqdm import tqdm

# ─── Configuration ────────────────────────────────────────────────────────────
GDC_API   = "https://api.gdc.cancer.gov"
PROJECT   = "TCGA-BRCA"
N_PATIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUTPUT_DIR = Path("TCGA_BRCA_data")
DOWNLOAD_WSI = True       # Set False to skip large SVS files
RANDOM_SEED  = int(sys.argv[2]) if len(sys.argv) > 2 else 42
# ──────────────────────────────────────────────────────────────────────────────

MODALITIES = ["Biospecimen", "Clinical", "Copy_Number_Variation", "WSI"]

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})


# ─── Helper: pretty-print JSON ────────────────────────────────────────────────
def gdc_get(endpoint, params=None):
    r = session.get(f"{GDC_API}/{endpoint}", params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def gdc_post(endpoint, payload):
    r = session.post(f"{GDC_API}/{endpoint}", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()


# ─── Step 1: Pick 5 random TCGA-BRCA cases ───────────────────────────────────
def get_random_cases(n=5, seed=42):
    print(f"\n[1/5] Fetching TCGA-BRCA cases …")
    payload = {
        "filters": {
            "op": "=",
            "content": {"field": "project.project_id", "value": PROJECT}
        },
        "fields": "case_id,submitter_id",
        "size": 1000,
        "format": "json"
    }
    data = gdc_post("cases", payload)
    hits = data["data"]["hits"]
    print(f"      Found {len(hits)} cases in {PROJECT}.")

    rng = random.Random(seed)
    chosen = rng.sample(hits, n)
    cases = [{"case_id": c["case_id"], "submitter_id": c["submitter_id"]} for c in chosen]
    for c in cases:
        print(f"      • {c['submitter_id']}  ({c['case_id']})")
    return cases


# ─── Step 2: Query file IDs for each modality ─────────────────────────────────
def query_files_for_case(case_id, data_category, data_type, access="open"):
    """Return list of {file_id, file_name} for a given case + modality."""
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.case_id",       "value": case_id}},
                {"op": "=", "content": {"field": "data_category",        "value": data_category}},
                {"op": "=", "content": {"field": "data_type",            "value": data_type}},
                {"op": "=", "content": {"field": "access",               "value": access}},
            ]
        },
        "fields": "file_id,file_name,file_size",
        "size": 50,
        "format": "json"
    }
    data = gdc_post("files", payload)
    return data["data"]["hits"]


# ─── Step 3: Download a file by file_id ───────────────────────────────────────
def download_file(file_id, dest_path: Path, file_name: str):
    dest_path.mkdir(parents=True, exist_ok=True)
    out_file = dest_path / file_name

    if out_file.exists():
        print(f"        ↳ Already exists, skipping: {file_name}")
        return

    url = f"{GDC_API}/data/{file_id}"
    with session.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_file, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"        {file_name[:50]}", leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))
    print(f"        ✓ {file_name}")


# ─── Modality-specific handlers ───────────────────────────────────────────────

def download_biospecimen(case_id, submitter_id, out_dir):
    """
    Downloads Biospecimen Supplement XML for the case.
    These are case-scoped XML files that contain only that patient's biospecimen tree.
    """
    files = query_files_for_case(
        case_id,
        data_category="Biospecimen",
        data_type="Biospecimen Supplement"
    )
    if not files:
        print(f"        ⚠ No open Biospecimen Supplement found for {submitter_id}")
        return
    for f in files:
        download_file(f["file_id"], out_dir, f["file_name"])


def download_clinical(case_id, submitter_id, out_dir):
    """
    Downloads Clinical Supplement XML for the case.
    These are case-scoped XML files containing only that patient's clinical data.
    """
    files = query_files_for_case(
        case_id,
        data_category="Clinical",
        data_type="Clinical Supplement"
    )
    if not files:
        print(f"        ⚠ No open Clinical Supplement found for {submitter_id}")
        return
    for f in files:
        download_file(f["file_id"], out_dir, f["file_name"])


def download_cnv(case_id, submitter_id, out_dir):
    """
    Downloads Masked Copy Number Segment (open-access) files.
    These are per-aliquot TXT files already scoped to that patient's sample.
    """
    files = query_files_for_case(
        case_id,
        data_category="Copy Number Variation",
        data_type="Masked Copy Number Segment"
    )
    if not files:
        # Fallback: try Copy Number Segment (also open-access)
        files = query_files_for_case(
            case_id,
            data_category="Copy Number Variation",
            data_type="Copy Number Segment"
        )
    if not files:
        print(f"        ⚠ No open CNV files found for {submitter_id}")
        return
    for f in files:
        download_file(f["file_id"], out_dir, f["file_name"])


def download_wsi(case_id, submitter_id, out_dir):
    """
    Downloads Diagnostic Slide WSI (SVS) for the case.
    Slide Image files in GDC are under data_category="Biospecimen",
    data_type="Slide Image", experimental_strategy="Diagnostic Slide".
    These are case-scoped (the file name contains the patient barcode).
    """
    if not DOWNLOAD_WSI:
        print(f"        ⚠ WSI download skipped (DOWNLOAD_WSI=False).")
        _write_wsi_manifest(case_id, submitter_id, out_dir)
        return

    files = _get_wsi_files(case_id)
    if not files:
        print(f"        ⚠ No open WSI files found for {submitter_id}")
        return
    for f in files:
        download_file(f["file_id"], out_dir, f["file_name"])


def _get_wsi_files(case_id):
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.case_id",            "value": case_id}},
                {"op": "=", "content": {"field": "data_category",             "value": "Biospecimen"}},
                {"op": "=", "content": {"field": "data_type",                 "value": "Slide Image"}},
                {"op": "=", "content": {"field": "experimental_strategy",     "value": "Diagnostic Slide"}},
                {"op": "=", "content": {"field": "access",                    "value": "open"}},
            ]
        },
        "fields": "file_id,file_name,file_size",
        "size": 20,
        "format": "json"
    }
    data = gdc_post("files", payload)
    return data["data"]["hits"]


def _write_wsi_manifest(case_id, submitter_id, out_dir):
    """If WSI download is skipped, write a manifest with download URLs."""
    files = _get_wsi_files(case_id)
    if not files:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "wsi_download_manifest.txt"
    with open(manifest_path, "w") as mf:
        mf.write("# WSI files skipped (DOWNLOAD_WSI=False)\n")
        mf.write("# To download manually, run:\n")
        mf.write(f"#   curl -O https://api.gdc.cancer.gov/data/<file_id>\n\n")
        for f in files:
            size_mb = f.get("file_size", 0) / (1024 ** 2)
            mf.write(f"{f['file_id']}\t{f['file_name']}\t{size_mb:.1f} MB\n")
            mf.write(f"  URL: {GDC_API}/data/{f['file_id']}\n\n")
    print(f"        ✓ WSI manifest written → {manifest_path}")


# ─── Step 4: Write per-patient summary ────────────────────────────────────────
def write_patient_summary(case, modality_files, out_dir):
    summary = {
        "patient_id": case["submitter_id"],
        "case_id": case["case_id"],
        "modalities": modality_files
    }
    with open(out_dir / "summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)


# ─── Main orchestrator ────────────────────────────────────────────────────────
def main():
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    OUTPUT_DIR.mkdir(exist_ok=True)
    cases = get_random_cases(N_PATIENTS, RANDOM_SEED)

    for idx, case in enumerate(cases, 1):
        pid  = case["submitter_id"]   # e.g. TCGA-A1-A0SB
        cid  = case["case_id"]        # UUID
        base = OUTPUT_DIR / pid

        print(f"\n{'='*60}")
        print(f"[Patient {idx}/{N_PATIENTS}]  {pid}")
        print(f"{'='*60}")

        modality_files = {}

        # ── Biospecimen ──────────────────────────────────────
        print(f"\n  [Biospecimen]")
        bio_dir = base / "Biospecimen"
        download_biospecimen(cid, pid, bio_dir)
        modality_files["Biospecimen"] = str(bio_dir)

        # ── Clinical ─────────────────────────────────────────
        print(f"\n  [Clinical]")
        clin_dir = base / "Clinical"
        download_clinical(cid, pid, clin_dir)
        modality_files["Clinical"] = str(clin_dir)

        # ── Copy Number Variation ────────────────────────────
        print(f"\n  [Copy Number Variation]")
        cnv_dir = base / "Copy_Number_Variation"
        download_cnv(cid, pid, cnv_dir)
        modality_files["Copy_Number_Variation"] = str(cnv_dir)

        # ── WSI ──────────────────────────────────────────────
        print(f"\n  [WSI]")
        wsi_dir = base / "WSI"
        download_wsi(cid, pid, wsi_dir)
        modality_files["WSI"] = str(wsi_dir)

        write_patient_summary(case, modality_files, base)
        print(f"\n  Summary saved → {base / 'summary.json'}")

    # ── Final tree overview ──────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Download complete!  Output directory: {OUTPUT_DIR.resolve()}")
    print(f"{'='*60}\n")
    _print_tree(OUTPUT_DIR, max_depth=3)


def _print_tree(path: Path, prefix="", max_depth=3, depth=0):
    if depth > max_depth:
        return
    items = sorted(path.iterdir()) if path.is_dir() else []
    for i, item in enumerate(items):
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item.name)
        if item.is_dir() and depth < max_depth:
            extension = "    " if i == len(items) - 1 else "│   "
            _print_tree(item, prefix + extension, max_depth, depth + 1)


if __name__ == "__main__":
    main()
