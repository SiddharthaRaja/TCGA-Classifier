#!/usr/bin/env python3
"""
Derive IHC-based subtype labels from TCGA's clinical biotab file.

    ER-, PR-, HER2-        -> Triple_Negative
    HER2+                  -> HER2_Positive
    (ER+ or PR+), HER2-    -> Luminal_A
    anything else          -> skipped

Usage
    python build_labels.py --status   # report counts, write nothing
    python build_labels.py            # write labels.json

Why the biotab file and not the XML
-----------------------------------
The previous version searched each patient's clinical XML for tags named
er_status_by_ihc / pr_status_by_ihc / her2_status_by_ihc. Those names exist
only in the biotab TSV; the BRCA XML does not contain them under any spelling,
so every patient came back "no xml" and zero labels were produced.

nationwidechildrens.org_clinical_patient_brca.txt carries all ~1098 patients in
one table and is downloaded into every patient folder, so one parse covers the
whole cohort.

HER2 resolution order
---------------------
IHC status first; if it is equivocal or missing, FISH; only then the raw IHC
score. The old code let her2_ihc_score win on a first-match basis, and that
field holds "2+", not "positive"/"negative" -- so unambiguous Luminal_A cases
were being silently dropped depending on field order.
"""

import argparse
import csv
import glob
import json
import os
from collections import Counter
from pathlib import Path

BASE = Path(os.environ.get("TCGA_DATA_ROOT",
            os.environ.get("TCGA_SCRATCH_ROOT", str(Path.home() / "siddhartha" / "data"))))
STATE = Path(os.environ.get("TCGA_STATE_ROOT", str(Path.home() / "siddhartha" / "state")))
RAW_DIR = BASE / "raw"
OUT_PATH = STATE / "labels.json"

BIOTAB = "nationwidechildrens.org_clinical_patient_brca.txt"
ID_COL = "bcr_patient_barcode"

BAD = {"", "[not evaluated]", "[not available]", "[unknown]", "[not applicable]",
       "[discrepancy]", "indeterminate", "equivocal", "na", "null"}
POS, NEG = "positive", "negative"


def clean(v):
    v = (v or "").strip().lower()
    return "" if v in BAD else v


def find_biotab():
    hits = sorted(glob.glob(str(RAW_DIR / "*" / "Clinical" / BIOTAB)))
    return hits[0] if hits else None


def read_biotab(path):
    """TCGA biotab files carry three header rows: field names, display names,
    then CDE_ID:. Data starts on row 4."""
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    header = rows[0]
    body = [r for r in rows[3:] if len(r) == len(header) and r[0].strip()]
    return header, body


def resolve_her2(rec):
    """IHC status -> FISH -> IHC score. First definitive answer wins."""
    ihc = clean(rec.get("her2_status_by_ihc"))
    if ihc in (POS, NEG):
        return ihc
    fish = clean(rec.get("her2_fish_status"))
    if fish in (POS, NEG):
        return fish
    score = clean(rec.get("her2_ihc_score"))
    if score in ("0", "1+"):
        return NEG
    if score == "3+":
        return POS
    return ""  # 2+ without FISH is genuinely undetermined


def derive(rec):
    er = clean(rec.get("er_status_by_ihc"))
    pr = clean(rec.get("pr_status_by_ihc"))
    her2 = resolve_her2(rec)

    if er == NEG and pr == NEG and her2 == NEG:
        return "Triple_Negative"
    if her2 == POS:
        return "HER2_Positive"
    if (er == POS or pr == POS) and her2 == NEG:
        return "Luminal_A"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true", help="report counts, write nothing")
    ap.add_argument("--all", action="store_true",
                    help="label every patient in the biotab, not just downloaded ones")
    args = ap.parse_args()

    path = find_biotab()
    if not path:
        raise SystemExit(f"ERROR: no {BIOTAB} found under {RAW_DIR}")
    print(f"  biotab: {path}")

    header, body = read_biotab(path)
    if ID_COL not in header:
        raise SystemExit(f"ERROR: column {ID_COL} not in header")
    idx = {c: i for i, c in enumerate(header)}

    downloaded = {p.name for p in RAW_DIR.iterdir() if p.is_dir()} if RAW_DIR.exists() else set()
    print(f"  biotab rows: {len(body)}   patient folders on disk: {len(downloaded)}")

    labels, skipped, missing_all = {}, Counter(), 0
    for row in body:
        pid = row[idx[ID_COL]].strip()
        if not args.all and downloaded and pid not in downloaded:
            continue
        rec = {c: row[idx[c]] for c in header}
        er, pr = clean(rec.get("er_status_by_ihc")), clean(rec.get("pr_status_by_ihc"))
        her2 = resolve_her2(rec)
        if not er and not pr and not her2:
            missing_all += 1
            continue
        lab = derive(rec)
        if lab is None:
            skipped[f"ER={er or '?'} PR={pr or '?'} HER2={her2 or '?'}"] += 1
            continue
        labels[pid] = lab

    print(f"\n  labeled              : {len(labels)}")
    print(f"  skipped (no receptor): {missing_all}")
    print(f"  skipped (ambiguous)  : {sum(skipped.values())}")
    for combo, n in skipped.most_common(5):
        print(f"      {combo}: {n}")
    print(f"  class distribution   : {dict(Counter(labels.values()))}")

    if not args.status and labels:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(labels, indent=2))
        print(f"\n  -> {OUT_PATH}  ({len(labels)} labeled patients)")


if __name__ == "__main__":
    main()
