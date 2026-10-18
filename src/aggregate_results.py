#!/usr/bin/env python3
"""
Collect every results/<backbone>_<pooling>.json file into one comparison CSV.

Usage
    python aggregate_results.py --out results/comparison_table.csv
"""

import argparse
import csv
import json
import os
from pathlib import Path

RESULTS_DIR = Path(os.environ.get("TCGA_RESULTS_ROOT", str(Path.home() / "siddhartha" / "results")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "comparison_table.csv":
            continue
        rows.append(json.loads(f.read_text()))

    if not rows:
        print("No result files found yet.")
        return

    rows.sort(key=lambda r: r["f1_macro_mean"], reverse=True)

    all_keys = []
    for r in rows:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print("\nRanked by F1 macro:")
    for r in rows:
        print(f"  {r['backbone']:<14} {r['pooling']:<10} "
              f"F1={r['f1_macro_mean']:.3f}±{r['f1_macro_std']:.3f}  "
              f"AUC={r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f}")


if __name__ == "__main__":
    main()
