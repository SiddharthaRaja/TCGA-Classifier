# Changelog

Every version is a git tag. To browse an old one, use `git checkout v1.0`, or
pick the tag on GitHub.

## v2.0 — October 2026

v2.0 is a ground-up rebuild. It is WSI-first and uses multiple backbones, and
it was hardened after repeated job kills and data wipes on Sharanga.

**Model**
- ResNet50 has been replaced by five feature extractors: DenseNet121,
  ConvNeXt-Tiny, Phikon, CTransPath and RetCCL. Each patch is tiled once and
  fed through all five.
- Every patch is read at a true 20x magnification. v1 read 40x slides at native
  resolution while treating them as 20x, so patients scanned at 40x got half the
  tissue context.
- Patch-level features are saved per patient. That lets **mean pooling** and
  **attention MIL** (gated, plain and top-K variants) be compared on identical
  features.
- Slides with two diagnostic images now use both of them.
- The multimodal track (clinical, biospecimen and CNV) is kept as an ablation,
  not as the main model.

**Labels**
- `build_labels.py` now reads the clinical biotab TSV. Before this change it
  searched the XML for tags that don't exist and produced zero labels.
- HER2 is resolved in this order: IHC status first, then FISH, then IHC score.
  Before this change, ER+/PR+ cases with an IHC score of "2+" were silently
  dropped.

**Robustness**
- Per-patient completion markers, written atomically, track pipeline state.
  Directory contents are no longer used to infer progress.
- Small, precious state (markers, features, clinical files, thumbnails) is kept
  separate from bulky raw slides, which are disposable. These are
  `TCGA_STATE_ROOT` and `TCGA_DATA_ROOT`.
- SIGTERM-safe: a job finishes its current patient and exits cleanly, so a kill
  costs at most one patient.
- Shards are split by position in the fixed manifest, and per-patient locks
  stop two workers taking the same patient. The old directory-listing shards
  overlapped and dropped patients.
- A slide is deleted only after all five backbones have saved its features. No
  code path deletes a directory.
- Includes offline tests (a fake GDC server and a synthetic slide) and a
  geometry selftest.

**Infrastructure**
- SLURM jobs: 6-shard downloads on `compute`, extraction on `gpu_v100_2`, and 16
  CPU shards inside one `big_compute` job.
- Fixed a conda activation failure in batch jobs (the spack-hash module name
  plus sourcing `conda.sh` directly) and the `PS1` / `set -u` conflict.

**Known gaps**
- The downstream scripts still read `$TCGA_SCRATCH_ROOT`. See the README.

## v1.1 — August 2026

A clean restart after the HPC scratch wipe. It keeps one version of each script
(no `_v2` / `_v3` duplicates), drives every path from `config.yaml`, keeps
scripts in `$HOME` and data in `$SCRATCH`, and adds a staged SLURM workflow
(`jobs/01–04`).

## v1.0 — mid 2026

The original multimodal pipeline. It combined clinical, CNV, biospecimen and
WSI data: ResNet50 mean-pooled embeddings, PCA/LDA and mutual-information
feature selection, then XGBoost + SMOTE.

v1.0 reported 86% accuracy and a 0.95 ROC-AUC. **Those numbers were inflated by
a feature-leakage bug.** After the fix, the honest ROC-AUC was **0.76**, which
is the baseline v2.0 is measured against.
