# TCGA-BRCA — download + feature extraction

Scope of this drop: get five backbones' patch-level feature vectors for every
patient. Nothing downstream (labels, splits, pooling, classifiers) is included.

## Layout

    ~/siddhartha/
      src/     tcga_common.py  download_tcga.py  extract_all_backbones.py
      jobs/    download.slurm  extract.slurm
      logs/    (created by the jobs)
      data/    raw/<PATIENT>/{WSI,Clinical,Biospecimen,Copy_Number_Variation}/
      state/   case_manifest.json
               patients/<PATIENT>/{download_complete.json,extract_complete.json,...}
               features_patch/<backbone>/<PATIENT>.npz     <- the deliverable
               features_pooled/<backbone>/<PATIENT>.npz
               thumbnails/<PATIENT>.jpg
               locks/

`data/` is bulky and disposable. `state/` is small and irreplaceable. Point
`TCGA_DATA_ROOT` at scratch if you want; keep `TCGA_STATE_ROOT` on `$HOME`.

## Install

    mkdir -p ~/siddhartha/{src,jobs,logs}
    # copy the three .py files into src/, the two .slurm files into jobs/
    cd ~/siddhartha/src
    for f in *.py; do python -m py_compile "$f" && echo "$f OK"; done

## Run

    cd ~/siddhartha/src
    export TCGA_DATA_ROOT=$HOME/siddhartha/data
    export TCGA_STATE_ROOT=$HOME/siddhartha/state

    python extract_all_backbones.py --selftest      # no GPU needed, ~1s
    python download_tcga.py --manifest              # once

    # smoke test on the login node before queueing anything
    python download_tcga.py --limit 2
    python extract_all_backbones.py --limit 1 --keep-wsi

    sbatch ~/siddhartha/jobs/download.slurm
    sbatch ~/siddhartha/jobs/extract.slurm          # safe to run at the same time

    python download_tcga.py --status
    python extract_all_backbones.py --status

Both jobs are sequential and resumable. When one hits its time limit, just
`sbatch` it again — it skips everything already finished. Nothing is ever
processed twice.

## How "don't repeat work" actually works

Two marker files per patient, under `state/patients/<PATIENT>/`:

| marker | written when | means |
|---|---|---|
| `download_complete.json` | every file is on disk at the byte size GDC reported | safe to extract |
| `extract_complete.json` | every requested backbone has saved an `.npz` | finished; slides deleted |

Both are written atomically (temp file + rename) and only after the work is
verified, so a job killed at any instant leaves no half-truth behind. Markers
live on `$HOME`, so a `/scratch` wipe costs re-download time but never progress
tracking.

Consequences worth knowing:

* **Extraction never touches an un-marked patient**, so a patient whose second
  slide is still transferring cannot be extracted from slide 1 alone.
* **Download never re-fetches an extracted patient.** Previously the missing
  `.svs` (deleted on purpose) looked like "needs downloading", and the two jobs
  fed each other in a loop forever.
* **If raw data disappears after download**, extraction notices the size
  mismatch, clears the download marker, and lets the download job re-fetch —
  rather than extracting a partial patient.
* **Per-patient lock files** mean two jobs can never work the same patient. A
  lock older than 6 hours is assumed to be from a killed job and taken over.

## Restarting from scratch, or redoing one patient

    rm -rf ~/siddhartha/state/patients/TCGA-XX-XXXX        # redo one patient
    rm ~/siddhartha/state/features_patch/*/TCGA-XX-XXXX.npz

Deleting `state/patients/` entirely re-downloads everything. Deleting only
`features_patch/<backbone>/` re-extracts just that backbone.

## Adding a backbone later (e.g. UNI)

Add it to `ALL_BACKBONES` and `EXPECTED_DIMS`, then run with
`--backbones uni`. Patients whose slides are already deleted cannot be redone
without re-downloading, so decide on the backbone list **before** the first
full run, or pass `--keep-wsi` while you are still deciding.

## Things to check before the full run

* `--selftest` passes on the GPU node.
* CTransPath loads. It is the one backbone I could not execute here (no
  torch/timm in my sandbox). Test it alone first:
  `python extract_all_backbones.py --backbones ctranspath --limit 1 --keep-wsi`
* Disk. At ~7,000 patches/patient and 5,376 dims across five backbones,
  float16 works out around 75 GB for all ~1,060 WSI patients (float32 would be
  ~150 GB). Both fit your 500 GB, but check `lfs quota` before committing.
* Time. Tiling dominates, not the GPU. Budget roughly 3-8 min/patient, so a
  full sequential pass is on the order of 60-140 hours across several
  resubmissions. If that is too slow, `--n-shards N --shard i` splits work by
  position in the **manifest** (stable, disjoint) rather than by directory
  contents (which is what made the old shards overlap and drop patients).

## Tests

`_test_download.py` and `_test_extract.py` run offline against a fake GDC
server and a synthetic slide — no network, no GPU, no real data:

    python _test_download.py && python _test_extract.py

They cover resume, truncated transfers, locks, missing raw data, the
no-diagnostic-slide case, two-slide patients, single-backbone failure and
partial retry. Worth re-running after any edit.
