#!/usr/bin/env python3
"""
Download TCGA-BRCA data from the GDC API, one patient at a time.

Sequential and resumable by design. Kill it at any point and resubmit; it picks
up where it left off and never redoes finished work.

Usage
    python download_tcga.py --manifest                # build the case list (run once)
    python download_tcga.py                           # download everything, sequentially
    python download_tcga.py --time-budget-hours 9     # stop cleanly before the wall clock
    python download_tcga.py --status                  # progress report, safe anytime
    python download_tcga.py --limit 5                 # smoke test on 5 patients

What changed vs the previous version, and why
---------------------------------------------
* A patient is "downloaded" only when a marker says so. Previously the script
  inferred it from "is there a .svs on disk", which broke the moment extraction
  deleted the slide: the file was gone, so it was re-downloaded, so it was
  re-extracted, forever. Extraction now writes its own marker, and this script
  treats an extracted patient as finished and never touches it again.

* The marker records every expected file with its exact byte size, including
  the WSI file list. Extraction refuses to start until every file in that list
  is present at the right size, which closes the race where a patient with two
  diagnostic slides got extracted from slide 1 while slide 2 was still in
  flight, then marked done forever.

* The marker is written last and atomically, after all verification passes. A
  kill mid-download leaves no marker, so the patient is simply retried.

* Small artifacts (clinical / biospecimen / CNV, all KBs) are copied into
  $HOME state as soon as they land, so a /scratch wipe costs only WSI bytes.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import requests

import tcga_common as C

# ---------------------------------------------------------------- config
PROJECT = "TCGA-BRCA"
MODALITIES = {"clinical": True, "biospecimen": True, "cnv": True, "wsi": True}
N_PATIENTS_CAP = None  # int to limit total patients, None for all (~1098)

GDC = os.environ.get("GDC_API", "https://api.gdc.cancer.gov")
# Overridable so the test harness can exercise the retry paths without waiting
# out real backoff. Leave them alone in production.
MAX_RETRIES = int(os.environ.get("GDC_MAX_RETRIES", "5"))
BACKOFF = float(os.environ.get("GDC_BACKOFF", "4"))  # seconds, doubled each retry

session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

MODALITY_SPEC = {
    "clinical": ("Clinical", "Clinical Supplement", "Clinical", None),
    "biospecimen": ("Biospecimen", "Biospecimen Supplement", "Biospecimen", None),
    "cnv": ("Copy Number Variation", "Masked Copy Number Segment",
            "Copy_Number_Variation", None),
    "wsi": ("Biospecimen", "Slide Image", "WSI",
            [{"op": "=", "content": {"field": "experimental_strategy",
                                     "value": "Diagnostic Slide"}}]),
}
SMALL_MODALITY_DIRS = ("Clinical", "Biospecimen", "Copy_Number_Variation")


def _post(endpoint, payload, timeout=120):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.post(f"{GDC}/{endpoint}", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            wait = BACKOFF * (2 ** attempt)
            print(f"      GDC {endpoint} failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"GDC {endpoint} failed after {MAX_RETRIES} attempts: {last}")


# ------------------------------------------------------------------ manifest
def build_manifest() -> list:
    print(f"  querying all {PROJECT} cases ...", flush=True)
    cases, page, size = [], 0, 500
    while True:
        data = _post("cases", {
            "filters": {"op": "=", "content": {
                "field": "project.project_id", "value": PROJECT}},
            "fields": "case_id,submitter_id",
            "size": size, "from": page, "format": "json",
        })
        hits = data["data"]["hits"]
        cases.extend({"case_id": h["case_id"], "submitter_id": h["submitter_id"]}
                     for h in hits)
        total = data["data"]["pagination"]["total"]
        page += size
        print(f"    {len(cases)}/{total}", flush=True)
        if page >= total:
            break

    # Sorted so the manifest order is deterministic. Everything downstream
    # partitions work by position in THIS list, so it must never be reshuffled.
    cases.sort(key=lambda c: c["submitter_id"])
    if N_PATIENTS_CAP:
        cases = cases[:int(N_PATIENTS_CAP)]
        print(f"  capped at n_patients={N_PATIENTS_CAP}")

    C.MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    C.MANIFEST.write_text(json.dumps(cases, indent=2))
    print(f"  -> {C.MANIFEST}  ({len(cases)} cases)", flush=True)
    return cases


# ------------------------------------------------------------------ files
def query_files(case_id, data_category, data_type, extra=None):
    content = [
        {"op": "=", "content": {"field": "cases.case_id", "value": case_id}},
        {"op": "=", "content": {"field": "data_category", "value": data_category}},
        {"op": "=", "content": {"field": "data_type", "value": data_type}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
    if extra:
        content.extend(extra)
    data = _post("files", {
        "filters": {"op": "and", "content": content},
        "fields": "file_id,file_name,file_size",
        "size": 100, "format": "json",
    })
    return data["data"]["hits"]


def download_file(file_id, dest_dir: Path, file_name: str, expected_size=None) -> str:
    """Download to <name>.part, verify the size, then rename into place.

    The rename is atomic, so a file at its final path is always complete. A
    kill mid-transfer leaves only a .part, which is removed and retried.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / file_name
    part = dest_dir / (file_name + ".part")

    if out.exists():
        if expected_size and out.stat().st_size != expected_size:
            print(f"        size mismatch, re-downloading {file_name}", flush=True)
            out.unlink()
        else:
            return "skip"

    for attempt in range(MAX_RETRIES):
        try:
            with session.get(f"{GDC}/data/{file_id}", stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            if expected_size and part.stat().st_size != expected_size:
                raise IOError(f"incomplete: {part.stat().st_size} != {expected_size}")
            os.replace(part, out)
            return "ok"
        except Exception as e:
            part.unlink(missing_ok=True)
            wait = BACKOFF * (2 ** attempt)
            print(f"        {file_name}: {e}; retry in {wait}s", flush=True)
            time.sleep(wait)
    print(f"        GAVE UP on {file_name}", flush=True)
    return "fail"


def copy_small_artifacts(pid):
    """Mirror the KB-sized modalities into $HOME state immediately.

    These are the irreplaceable-but-tiny parts. Copying them the moment they
    land means a /scratch wipe costs only re-downloadable WSI bytes.
    """
    src = C.RAW_DIR / pid
    dst = C.patient_state_dir(pid)
    for modality in SMALL_MODALITY_DIRS:
        src_mod = src / modality
        if not src_mod.exists():
            continue
        dst_mod = dst / modality
        dst_mod.mkdir(parents=True, exist_ok=True)
        for f in src_mod.iterdir():
            if f.is_file() and not f.name.endswith(".part"):
                shutil.copy2(f, dst_mod / f.name)


def download_case(case) -> tuple:
    """Returns (stats, manifest_of_files, ok). ok is False if anything failed,
    in which case NO marker is written and the patient is retried next run."""
    pid = case["submitter_id"]
    base = C.RAW_DIR / pid
    stats, file_manifest, all_ok = {}, {}, True

    for mod, enabled in MODALITIES.items():
        if not enabled or mod not in MODALITY_SPEC:
            continue
        cat, dtype, folder, extra = MODALITY_SPEC[mod]
        try:
            files = query_files(case["case_id"], cat, dtype, extra)
        except Exception as e:
            print(f"      {mod}: query failed — {e}", flush=True)
            stats[mod] = "queryfail"
            all_ok = False
            continue

        if not files and mod == "cnv":
            files = query_files(case["case_id"], cat, "Copy Number Segment")

        if not files:
            # A genuine absence, not a failure. Some cases really have no CNV
            # or no diagnostic slide; recording an empty list keeps that
            # distinguishable from "we never looked".
            stats[mod] = "none"
            file_manifest[mod] = []
            continue

        entries, results = [], []
        for f in files:
            res = download_file(f["file_id"], base / folder, f["file_name"],
                                f.get("file_size"))
            results.append(res)
            if res in ("ok", "skip"):
                entries.append({"name": f["file_name"], "size": f.get("file_size")})
        file_manifest[mod] = entries
        if "fail" in results:
            all_ok = False
        stats[mod] = (f"{results.count('ok')}ok/{results.count('skip')}skip"
                      + (f"/{results.count('fail')}fail" if "fail" in results else ""))

    return stats, file_manifest, all_ok


def verify_on_disk(pid, file_manifest) -> bool:
    """Confirm every file we think we downloaded is really there at the right
    size, before we commit to a marker that says so."""
    for mod, entries in file_manifest.items():
        folder = MODALITY_SPEC[mod][2]
        for e in entries:
            p = C.RAW_DIR / pid / folder / e["name"]
            if not p.exists():
                print(f"      verify failed: missing {folder}/{e['name']}", flush=True)
                return False
            if e["size"] and p.stat().st_size != e["size"]:
                print(f"      verify failed: wrong size {folder}/{e['name']}", flush=True)
                return False
    return True


# ------------------------------------------------------------------ status
def status(cases):
    n_dl = n_ex = 0
    n_wsi = 0
    for c in cases:
        pid = c["submitter_id"]
        dl = C.read_marker(pid, C.DOWNLOAD_MARKER)
        if dl:
            n_dl += 1
            if dl.get("files", {}).get("wsi"):
                n_wsi += 1
        if C.is_extracted(pid):
            n_ex += 1
    total = len(cases)
    print(f"  cases in manifest        : {total}")
    print(f"  downloaded (marker)      : {n_dl}")
    print(f"    ...of which have WSI   : {n_wsi}")
    print(f"  features extracted       : {n_ex}")
    print(f"  remaining to download    : {total - n_dl}")
    print(f"  remaining to extract     : {n_wsi - n_ex}  (of downloaded-with-WSI)")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true", help="rebuild the case list and exit")
    ap.add_argument("--status", action="store_true", help="report progress and exit")
    ap.add_argument("--limit", type=int, default=None, help="stop after N new patients")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="stop starting new patients after this long (set below SBATCH --time)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1,
                    help="optional stable manifest-based split; default 1 = sequential")
    args = ap.parse_args()

    C.ensure_dirs()
    C.banner(f"GDC download — {PROJECT}")
    C.describe_roots()

    if args.manifest:
        build_manifest()
        return

    cases = C.load_manifest()
    if args.status:
        status(cases)
        return

    mine = C.select_patients(cases, args.shard, args.n_shards)
    ctl = C.RunController(args.time_budget_hours)
    if args.n_shards > 1:
        print(f"  shard {args.shard}/{args.n_shards}: {len(mine)} of {len(cases)} cases")
    print(f"  modalities: {[k for k, v in MODALITIES.items() if v]}\n", flush=True)

    done = skipped = failed = 0
    for i, case in enumerate(mine, 1):
        pid = case["submitter_id"]

        stop, why = ctl.should_stop()
        if stop:
            print(f"\n  stopping: {why}. Resubmit to continue.", flush=True)
            break
        if args.limit and done >= args.limit:
            print(f"\n  reached --limit {args.limit}.", flush=True)
            break

        # An extracted patient is finished forever. Its WSI was deliberately
        # deleted to reclaim space; do not resurrect it.
        if C.is_extracted(pid):
            skipped += 1
            continue
        if C.is_downloaded(pid):
            skipped += 1
            continue

        with C.PatientLock(pid) as lk:
            if not lk.acquired:
                print(f"  [{i}/{len(mine)}] {pid} — locked by another job, skipping", flush=True)
                continue

            print(f"  [{i}/{len(mine)}] {pid}", flush=True)
            t0 = time.time()
            try:
                stats, file_manifest, ok = download_case(case)
                copy_small_artifacts(pid)

                if not ok or not verify_on_disk(pid, file_manifest):
                    print("      INCOMPLETE — no marker written, will retry next run",
                          flush=True)
                    failed += 1
                    continue

                # Marker last, and only now: its existence is the promise that
                # every file listed inside it is on disk at the stated size.
                C.write_marker(pid, C.DOWNLOAD_MARKER, {
                    "patient_id": pid,
                    "case_id": case["case_id"],
                    "modality_stats": stats,
                    "files": file_manifest,
                })
                done += 1
                n_wsi = len(file_manifest.get("wsi", []))
                print(f"      {stats} — {n_wsi} slide(s) — {time.time() - t0:.1f}s",
                      flush=True)
            except Exception as e:
                print(f"      CASE FAILED — {e}", flush=True)
                failed += 1

    print(f"\n  new: {done}   already done: {skipped}   failed (will retry): {failed}")
    print(f"  elapsed: {ctl.elapsed_h():.2f}h")
    print("  run  python download_tcga.py --status  for the overall picture.", flush=True)


if __name__ == "__main__":
    main()
