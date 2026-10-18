# TCGA-BRCA Molecular Subtype Classification — v2.0

> **v2.0 (Oct 2026)** — a rebuild around a multi-backbone, WSI-first pipeline
> that survives job kills. Earlier versions are preserved as git tags:
> [`v1.0`](../../tree/v1.0) (original multimodal XGBoost) and
> [`v1.1`](../../tree/v1.1) (Aug 2026 clean restart). See [CHANGELOG.md](CHANGELOG.md).

Classifies breast cancer patients from TCGA-BRCA into IHC-derived subtypes
(**Triple Negative**, **HER2 Positive**, **Luminal A**) from H&E whole-slide
images (WSIs). Coursework project CS F377, BITS Pilani Hyderabad, running on
the Sharanga HPC cluster (SLURM).

The pipeline extracts patch-level features from every diagnostic slide with
five backbones, then compares **mean pooling** against **attention MIL** for
slide-level classification. An optional multimodal track adds clinical,
biospecimen and CNV features.

## Background

v1.0 used ResNet50 embeddings with XGBoost + SMOTE. It reported
86% accuracy, which turned out to come from a feature-leakage bug. After the fix
it scored an honest ROC-AUC of **0.76**. v2.0 replaces that design with
a multi-backbone, WSI-first pipeline. It is built to survive the repeated
job kills and data wipes seen on the cluster.

## Pipeline

```
GDC API ──► download_tcga.py ──► data/raw/<patient>/{Clinical,Biospecimen,CNV,WSI}
                │                         │
                │ marker + small files    │ slides (disposable)
                ▼                         ▼
          state/patients/        extract_all_backbones.py
                                   tile once at 20x, 256px, tissue-filtered
                                   → 5 backbones per patch
                                   → slide deleted only after all 5 succeed
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
      state/features_patch/<bb>/<pid>.npz       state/features_pooled/<bb>/<pid>.npz
      (per-patch, for attention MIL)            (mean vector, for pooled classifiers)
                     │                                         │
                     └──────────► build_labels → build_splits ◄┘
                                        │
                              train_pooled_classifier.py
                                        │
                               aggregate_results.py → comparison_table.csv
```

| Backbone | Pretraining | Dim |
|---|---|---|
| DenseNet121 | ImageNet | 1024 |
| ConvNeXt-Tiny | ImageNet | 768 |
| Phikon (`owkin/phikon`) | Pathology ViT | 768 |
| CTransPath (`1aurent/swin_tiny_patch4_window7_224.CTransPath`) | Pathology Swin | 768 |
| RetCCL (`jamesdolezal/RetCCL`) | Pathology ResNet50 | 2048 |

### Design rules (why the code looks the way it does)

- **State lives in `$HOME`, data may not.** `TCGA_STATE_ROOT` holds markers,
  features, small clinical files and thumbnails. `TCGA_DATA_ROOT` holds only
  raw slides, which are disposable.
- **Markers are the state machine.** A patient goes from *nothing* to
  `download_complete.json` to `extract_complete.json`. Every marker is written
  atomically, with a temp file and then `os.replace`.
- **Kill-safe.** SIGTERM finishes the current patient and then exits cleanly.
  A kill at any instruction loses at most one patient. Resubmitting resumes
  where the last run stopped.
- **Stable sharding.** `--shard i --n-shards N` splits work by position in the
  fixed manifest, so shards never overlap. Per-patient `O_EXCL` locks act as a
  backstop.
- **Nothing deletes a directory.** The only deletes are single slide files
  (`Path.unlink`), and only after every backbone has saved its features.

## Repository layout

```
src/
  tcga_common.py                shared roots, markers, locks, shutdown handling
  download_tcga.py              GDC download, resumable, size-verified
  extract_all_backbones.py      tiling + 5-backbone feature extraction
  build_labels.py               IHC subtype labels from the clinical biotab TSV
  build_splits.py               stratified CV splits
  build_tabular_features.py     clinical / biospecimen / CNV features (multimodal track)
  train_pooled_classifier.py    mean-pooling and attention-MIL classifiers
  aggregate_results.py          collects results into a comparison table
  generate_attention_heatmap.py attention overlays on slide thumbnails
  Snakefile                     end-to-end orchestration
  _test_*.py                    offline tests (fake GDC server + synthetic slide)
jobs/
  download.slurm                CPU download, 6 shards on `compute`
  extract.slurm                 GPU extraction on `gpu_v100_2`
  extract_cpu.slurm             CPU extraction on `compute`
  extract_big.slurm             16 CPU extraction shards in one `big_compute` job
bundle.sh                       tars state/ (pooled features, markers, thumbnails) for backup
docs/RUN.md                     detailed run notes
```

## Setup (Sharanga)

```bash
module load anaconda3-2021.05-gcc-8.5.0-i5w2nbn
source /apps/spack-stable/opt/spack/linux-rocky8-zen/gcc-8.5.0/anaconda3-2021.05-i5w2nbnp5mvg3rvzxg4vzoylnbulv3fo/etc/profile.d/conda.sh
conda activate tcga_env

# in ~/.bashrc — and remove any old TCGA_SCRATCH_ROOT line
export TCGA_DATA_ROOT=$HOME/siddhartha/data
export TCGA_STATE_ROOT=$HOME/siddhartha/state
```

The conda module name needs its spack hash suffix. In batch jobs, source
`conda.sh` directly, since plain `conda activate` fails silently there. Don't
use `set -u` in job scripts either: it breaks on conda's unset `PS1`.

Key dependencies: `torch`, `torchvision`, `timm`, `transformers`,
`huggingface_hub`, `openslide-python`, `numpy`, `requests`, `scikit-learn`,
`xgboost`, `snakemake`.

## Running

```bash
cd ~/siddhartha/src

# sanity checks (no GPU, no network)
python extract_all_backbones.py --selftest
python _test_download.py && python _test_extract.py

# build the case list once (1098 cases)
python download_tcga.py --manifest

# smoke test
python download_tcga.py --limit 2
python extract_all_backbones.py --limit 1 --keep-wsi

# full run
for s in 0 1 2 3 4 5; do sbatch --export=ALL,SHARD=$s ~/siddhartha/jobs/download.slurm; done
sbatch ~/siddhartha/jobs/extract_big.slurm
sbatch ~/siddhartha/jobs/extract.slurm

# progress, safe anytime
python download_tcga.py --status
python extract_all_backbones.py --status
du -sh ~/siddhartha/data/raw/      # if this keeps climbing, cancel a download shard

# labels — --all gives the full-cohort class distribution before downloads finish
python build_labels.py --status --all
python build_labels.py
```

### Throughput notes

- GDC gives about 1.3 MB/s per connection, so downloads are the bottleneck.
  Six shards give about 30 patients per hour.
- Tiling (`read_region` plus decode) dominates extraction, not the GPU. A CPU
  worker takes about 30 minutes per patient uncapped, and slides average around
  38k tissue patches.
- `big_compute` requires at least 128 CPUs per job and allows 8 jobs per user.
  `compute` allows 7 jobs per user.

## Known gaps / next steps

- **Downstream path mismatch.** `build_splits`, `build_tabular_features`,
  `train_pooled_classifier`, `generate_attention_heatmap` and the `Snakefile`
  still read `$TCGA_SCRATCH_ROOT`, while features now live under
  `$TCGA_STATE_ROOT`. Until they are repointed, run them with
  `TCGA_SCRATCH_ROOT=$HOME/siddhartha/state`.
- The `Snakefile`'s extraction rule references
  `jobs/extract_all_backbones.slurm`, which has been replaced by the job files
  above. Run extraction through SLURM directly for now.
- Planned: a UNI backbone (pending gated HF access), true PAM50 RNA-seq labels,
  deep multimodal fusion, and U-Net tissue segmentation as an ablation.

## Data notes

Raw data is not in this repo: slides are about 1–7 GB per patient. Patch
features are roughly 75–170 GB at float16. Pooled features for the whole cohort
come to about 23 MB. Run `bundle.sh` to snapshot the precious, small part of
`state/`.
