#!/usr/bin/env python3
"""
Build ONE canonical, patient-level stratified 5-fold split, reused by every
backbone/pooling/modality experiment so comparisons are fair and leakage-safe.

Usage
    python build_splits.py
"""

import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE = Path(os.environ.get("TCGA_SCRATCH_ROOT", str(Path.home() / "siddhartha" / "data")))
LABELS_PATH = BASE / "labels.json"
OUT_PATH = BASE / "splits.json"
N_FOLDS = 5
SEED = 42


def main():
    if not LABELS_PATH.exists():
        print(f"ERROR: {LABELS_PATH} not found — run build_labels.py first.")
        return

    labels = json.loads(LABELS_PATH.read_text())
    patient_ids = sorted(labels.keys())
    y = np.array([labels[p] for p in patient_ids])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_ids, y)):
        folds.append({
            "fold": fold_idx,
            "train": [patient_ids[i] for i in train_idx],
            "val": [patient_ids[i] for i in val_idx],
        })
        print(f"  fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")

    OUT_PATH.write_text(json.dumps({"n_folds": N_FOLDS, "seed": SEED, "folds": folds}, indent=2))
    print(f"\n  -> {OUT_PATH}  ({len(patient_ids)} patients, {N_FOLDS} folds)")


if __name__ == "__main__":
    main()
