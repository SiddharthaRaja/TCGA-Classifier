TCGA-BRCA Multimodal Cancer Subtype Classification

A large-scale multimodal machine learning pipeline for breast cancer subtype classification using clinical records, copy number variation (CNV), biospecimen metadata, and whole slide histopathology images (WSI) from the TCGA-BRCA dataset.

Built for scalable execution on HPC infrastructure using SLURM, GPU-accelerated WSI feature extraction, and ensemble-based machine learning workflows.

Overview

This project develops a multimodal cancer subtype classification framework using data from the TCGA-BRCA cohort available through the NCI Genomic Data Commons (GDC).

The system integrates:

Clinical metadata
Copy Number Variation (CNV) genomic features
Biospecimen metadata
Whole Slide Images (WSI)

into a unified machine learning pipeline capable of predicting breast cancer molecular subtypes.

The architecture was designed for:

large-scale reproducibility,
HPC deployment,
GPU-accelerated pathology processing,
multimodal feature fusion,
ablation studies,
and future extension toward pathology foundation models and survival prediction tasks.

The project currently achieves:

86% classification accuracy
0.78 macro F1-score
0.95 ROC-AUC

using an optimized XGBoost + SMOTE training pipeline on multimodal TCGA-BRCA data.

Scientific Motivation

Breast cancer subtypes exhibit distinct molecular, genomic, and histopathological characteristics.

Current clinical workflows determine subtype primarily using:

ER receptor status
PR receptor status
HER2 receptor status

This project investigates whether multimodal machine learning models can:

validate subtype assignments,
quantify modality contribution,
identify discriminative genomic/morphological signals,
and ultimately predict subtype directly from tissue morphology.

The long-term goal is building clinically meaningful AI systems capable of operating in low-resource environments where expensive molecular assays may not be available.

Dataset
Source
TCGA-BRCA (The Cancer Genome Atlas — Breast Invasive Carcinoma)
Accessed via:
GDC API
GDC Data Portal

Dataset includes:

Clinical XML files
Biospecimen XML files
CNV segment TSV files
Whole Slide SVS histopathology images

Classification Labels

The current implementation uses 3-class IHC-derived subtype labels:

Subtype	Definition
Luminal_A	ER+ or PR+, HER2-
HER2_Positive	HER2+
Triple_Negative	ER-, PR-, HER2-

Pipeline Architecture
                ┌────────────────────┐
                │   GDC API Download │
                └─────────┬──────────┘
                          │
        ┌───────────────────────────────────┐
        │           Raw Modalities          │
        └───────────────────────────────────┘
            │        │        │        │
            ▼        ▼        ▼        ▼
       Clinical    CNV   Biospecimen   WSI
            │        │        │        │
            ▼        ▼        ▼        ▼
       Preprocessing + Feature Engineering
                          │
                          ▼
              Multimodal Feature Fusion
                          │
                          ▼
             Dimensionality Reduction
                 PCA / LDA / MI
                          │
                          ▼
                   XGBoost Models
                          │
                          ▼
                    Evaluation
Modalities
1. Clinical Data

Clinical XML files contain:

demographics,
receptor status,
tumour pathology,
treatment history,
and survival outcomes.
Processing Includes
Namespace-agnostic XML parsing
Missing value normalization
Ordinal encoding of stage/grade
Binary receptor encoding
Survival engineering
One-hot categorical encoding
StandardScaler normalization

2. Copy Number Variation (CNV)

Masked copy number segment files are processed into:

chromosome-level instability statistics,
amplification/deletion summaries,
genomic instability scores,
chromosome-specific CNV means.
Biological Signals Captured
HER2 amplification on chromosome 17
Luminal A chr8/chr16 signatures
Basal-like chromosomal instability

3. Biospecimen Metadata

Biospecimen XML hierarchy:

Patient
Sample
Portion
Analyte
Aliquot

Features include:

tumour dimensions,
analyte counts,
FFPE processing,
specimen quality metrics,
metastatic indicators.

4. Whole Slide Images (WSI)

Gigapixel histopathology slides are processed using:

OpenSlide,
tissue segmentation,
patch extraction,
and CNN-based feature extraction.
WSI Pipeline
Tissue detection using HSV saturation + Otsu thresholding
512x512 patch extraction at 20x magnification
ResNet50 feature extraction
Mean pooling into slide-level embeddings
Export to 2048-dimensional patient vectors

Feature Engineering

Custom engineered features include:

Feature	Description
eng_genomic_instability_score	Weighted CNV instability
eng_chr17_her2_interaction	HER2 amplification marker
eng_gain_loss_ratio	CNV gain/loss balance
eng_wsi_patch_density	Morphological density
eng_age_stage_interaction	Clinical interaction term

Dimensionality Reduction

Two dimensionality reduction strategies were implemented.

PCA Pipeline

WSI features:

2048 → 50 dimensions
94.1% variance retained
LDA + Feature Selection Pipeline
WSI LDA reduction
Biospecimen PCA
Mutual information feature selection

Machine Learning Models
XGBoost

Primary classifier:

multiclass soft probability objective,
SMOTE balancing,
class weighting,
feature subsampling,
L1/L2 regularization.
MLP Ensemble

A multilayer perceptron was designed for:

nonlinear cross-modal interactions,
latent feature learning,
ensemble soft voting.

Results
Best Performing Model
Metric	Value
Accuracy	0.86
Macro F1	0.78
ROC-AUC	0.95

Per-Class Performance
Subtype	Precision	Recall	F1
HER2_Positive	0.89	0.74	0.81
Luminal_A	0.87	0.96	0.91
Triple_Negative	0.75	0.54	0.63

HPC Infrastructure

Developed and executed on:

Sharanga HPC Cluster
BITS Pilani Hyderabad Campus
Environment
Component	Value
Scheduler	SLURM
GPU Partition	gpu_h100_4
Main CPU Partition	compute
Environment	Conda
Frameworks	PyTorch, XGBoost

Repository Structure
classification/
│
├── tcga_brca_downloader.py
├── preprocess_clinical.py
├── preprocess_biospecimen.py
├── preprocess_cnv.py
├── preprocess_wsi.py
├── collect_wsi.py
├── merge_modalities.py
├── feature_engineering.py
├── dimensionality_reduction.py
├── dimensionality_reduction_v2.py
├── train_xgboost.py
├── train_xgboost_v2.py
├── train_xgboost_v3.py
├── assign_subtypes_ihc.py
│
├── download_batch2.sh
├── preprocess_job.sh
├── wsi_job.sh
├── train_job.sh
│
└── config.yaml

Installation
Clone Repository
git clone <your-repo-url>
cd classification
Create Environment
conda create -n tcga_env python=3.11
conda activate tcga_env
Install Dependencies
pip install pandas numpy scikit-learn torch torchvision timm xgboost imbalanced-learn openslide-python openslide-bin opencv-python lxml matplotlib seaborn requests tqdm scipy pillow
Running the Pipeline
1. Download TCGA Data
python tcga_brca_downloader.py

or using SLURM:

sbatch download_batch2.sh
2. Preprocess Modalities
python preprocess_clinical.py
python preprocess_biospecimen.py
python preprocess_cnv.py
3. Extract WSI Features

GPU recommended:

sbatch wsi_job.sh
4. Merge Modalities
python merge_modalities.py
5. Feature Engineering
python feature_engineering.py
6. Dimensionality Reduction
python dimensionality_reduction_v2.py
7. Train Model
python train_xgboost_v2.py
Current Project Status
Completed
Multimodal download pipeline
Clinical preprocessing
CNV preprocessing
Biospecimen preprocessing
WSI feature extraction
Feature engineering
PCA/LDA pipelines
XGBoost training
SLURM automation
HPC deployment
In Progress
Full 1000-patient pipeline
End-to-end automation
Config-driven execution
Full WSI coverage

Key Technical Challenges Solved
XML Namespace Inconsistency

Implemented namespace-agnostic parsing for TCGA XML schema variations.

CNV Tumour/Normal Separation

Designed heuristics for tumour-vs-normal CNV identification.

Gigapixel WSI Processing

GPU-accelerated patch extraction and feature pooling pipeline.

HPC Scaling

Built SLURM-based distributed processing architecture.

Feature Imbalance

Handled 2048-dimensional WSI dominance using:

feature subsampling,
PCA/LDA,
mutual information filtering.

Future Work
Planned Improvements
UNI pathology foundation model integration
CONCH / PLIP pathology transformers
Attention-based MIL aggregation
True PAM50 RNA-seq labels
Survival prediction modeling
WSI-only classifier
Intermediate multimodal fusion
Transformer-based architectures

Research Significance

This project demonstrates:

scalable multimodal cancer ML,
pathology image processing at HPC scale,
genomic + histopathological feature fusion,
interpretable biomedical ML pipelines,
and real-world HPC deployment for computational oncology.

It provides a strong foundation for:

multimodal pathology AI,
computational cancer genomics,
survival prediction,
and clinically deployable pathology models.
