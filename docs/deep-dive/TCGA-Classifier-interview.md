# TCGA-Classifier — Interview Questions & Model Answers

> Companion to [`TCGA-Classifier.md`](./TCGA-Classifier.md). Answers are in first person and sized to say out loud in about 30–90 seconds. "Deeper note" gives extra detail in case the interviewer digs in.
> Difficulty: 🟢 Easy · 🟡 Medium · 🔴 Hard

**Beginner tip:** if you only prep 10 questions, prep A1, A3, B1, C3, D1, E1, E2, G1, H1 and all the hostile follow-ups. They're the ones that decide whether the interviewer trusts you.

---

## (a) Project overview & motivation

**A1. 🟢 Walk me through the project in a minute.**
> It's a multimodal pipeline for breast-cancer subtyping on TCGA-BRCA. I download about 550 patients from the GDC API. Each patient has four data types: clinical XML, copy-number segments, biospecimen XML and diagnostic whole-slide images. I turn each one into patient-level features: parsed clinical fields, CNV summaries per chromosome, sample-hierarchy stats, and a 2048-d ResNet-50 embedding mean-pooled over tissue patches. I merge them on patient ID and train an XGBoost + MLP soft-voting ensemble with stratified 5-fold CV. Everything ran as SLURM jobs on our H100 cluster.

**A2. 🟢 Why breast cancer, and why multimodal?**
> Breast cancer subtypes drive treatment, and TCGA-BRCA is the largest open cohort that has slides, genomics and clinical data for the same patients. Each modality sees a different layer: CNV sees the genome, slides see morphology, clinical data gives context. The hypothesis was that combining them beats any single one. The long-term motivation is low-resource settings: if H&E morphology can predict molecular subtype, you could skip an expensive assay.

**A3. 🟡 What exactly are you predicting?**
> A subtype label derived from ER, PR and HER2 receptor status. It's an IHC surrogate for PAM50: triple-negative maps to "Basal-like", HER2+ with ER/PR− to "HER2-enriched", ER/PR+ HER2− to "Luminal A", and ER/PR+ HER2+ to "Luminal B". I want to be upfront that this isn't true PAM50, which needs RNA expression. That distinction matters a lot for how you read the results.
>
> *Deeper note:* `classification/assign_subtypes.py:7-24`. The handover doc says so too: "This is a clinical approximation."

**A4. 🟢 Who is the user of this system?**
> Right now it's a research pipeline for me and my collaborators. Notice it writes formatted Excel copies of every CSV, which were for non-coding teammates. If it ever became a product, the user would be a pathologist in a hospital that has H&E slides but no molecular testing.

**A5. 🟡 What part are you proudest of?**
> The WSI pipeline: going from a gigapixel `.svs` to a vector with OpenSlide pyramid reads, an Otsu tissue mask on the saturation channel, grid tiling with a tissue-fraction filter, and batched GPU inference. It's the most engineering-heavy part, and it's what made the HPC necessary.

**A6. 🟡 What did you learn from it?**
> Mostly about evaluation hygiene. The plumbing worked, but I learned how easy it is to build a pipeline that produces impressive numbers for the wrong reason. Labels derived from inputs, metrics computed in-sample. If I did it again, I'd write the leakage tests before writing the model.

---

## (b) Architecture & design

**B1. 🟡 Describe the architecture.**
> It's a linear chain of scripts that hand off CSVs. Download → one preprocessing script per modality → merge → label assignment → feature engineering → train/eval. The merge prefixes columns with `clin_`, `cnv_`, `bio_` and `wsi_` so you always know which modality a feature came from. Clinical, CNV and biospecimen are inner-joined. WSI is left-joined because not every slide finished. SLURM scripts wrap each heavy stage.

**B2. 🟡 Why early fusion (concatenation) rather than late or intermediate fusion?**
> Simplicity and sample size. With about 500 patients, training separate deep encoders per modality would overfit, and concatenation lets any tabular model consume everything. The cost is that the 2048 WSI columns dominate, and missing modalities have to be imputed. Late fusion, meaning one model per modality and then averaging probabilities, would handle missing slides more gracefully. That's what I'd try next.

**B3. 🟢 Why separate scripts instead of one program?**
> Each stage has very different resource needs. Downloading is I/O-bound for days, WSI needs a GPU for hours, and tabular steps take seconds. Separate scripts map cleanly onto separate SLURM jobs and partitions, and a failure in one doesn't force re-running the others. The downside is there's no dependency graph or resume logic. Snakemake would fix that.

**B4. 🟡 How do you handle patients missing a modality?**
> Clinical, CNV and biospecimen are required (inner join). For WSI I use a left join and then `fillna(0)`. Honestly, that's the weakest choice in the merge. Zero isn't "missing", and because other features are standardised, zero means "average". The right approach is a missing indicator plus imputation inside the CV pipeline, or late fusion.

**B5. 🔴 How would you re-architect it for 10,000 patients?**
> I'd move to a workflow engine (Snakemake or Nextflow) with per-patient tasks, so WSI extraction is embarrassingly parallel across GPUs and resumable. I'd store patch features in Zarr or HDF5 keyed by patient, and keep a manifest with GDC MD5s. Features would go into a feature store or Parquet with a schema. Training would be an sklearn/PyTorch pipeline with config files, experiment tracking (MLflow or W&B), and the data hash logged with every run.

**B6. 🟡 Why save raw patch features as `.npy` if you only use the mean?**
> So I can change the aggregation without re-running the expensive CNN pass. The `.npy` files are N×2048 per patient, so attention-MIL or max-pooling can be trained later from them directly. That was a deliberate future-proofing choice (`preprocess_wsi.py:281-284`).

---

## (c) Code-level / implementation

**C1. 🟡 How does your XML parsing deal with TCGA's namespace mess?**
> TCGA clinical XML uses version-specific namespace URIs, so namespaced XPath breaks across files. My `_find_tag` walks every element and compares only the local tag name, splitting off the `{namespace}` part, and returns the first non-empty match. It also takes several aliases per field, e.g. `breast_carcinoma_estrogen_receptor_status` or `er_status_by_ihc`.
>
> *Deeper note:* `preprocess_clinical.py:48-55`. Trade-off: first match wins, so if a tag appears in a follow-up block too, you might get the wrong instance.

**C2. 🟡 Explain the tissue detection.**
> I read the lowest-resolution pyramid level as a thumbnail, convert it to HSV and apply Otsu on the saturation channel. Stained tissue is saturated pink/purple, and glass is white with low saturation. Then a 5×5 close and open to clean it up. For each candidate patch on a level-0 grid, I map it to thumbnail coordinates and keep it if at least 50% of it is tissue.

**C3. 🔴 There's a magnification subtlety in `get_level_for_mag`. What is it?**
> I compute downsample = native/20 and call `get_best_level_for_downsample`. TCGA pyramids usually go 1×, 4×, 16×. For a 40× slide I ask for 2× and OpenSlide returns level 0, which is still 40×. So 40× slides get 512-px patches at 40× (half the field of view), while 20× slides get them at 20×. Magnification is inconsistent across the cohort. The fix is to read at level 0 with a patch size scaled by native/target and resize, or use the microns-per-pixel metadata.

**C4. 🟢 Why `num_classes=0` in `timm.create_model`?**
> It removes the classification head, so the forward pass returns the 2048-d globally-pooled features instead of 1000 ImageNet logits.

**C5. 🟡 How do you decide which CNV file is tumour vs normal?**
> The current code sorts by file size and assumes the larger one is tumour, because tumours have more segments. It's a heuristic and I wouldn't defend it as final. The correct way is the TCGA barcode sample-type code: positions 14–15, `01` is primary tumour and `10`/`11` are normal. Or query `cases.samples.sample_type` from GDC.

**C6. 🟡 What does the leak filter do, and where does it fail?**
> It drops any column whose name contains `er_status`, `pr_status`, `her2`, `receptor`, `pam50`, `subtype` and so on. It fails on two engineered features, `eng_is_triple_negative` and `eng_is_luminal`. They *are* the label definitions, but their names don't contain any keyword. A deny-list by name is the wrong tool. You need an explicit allow-list of features, plus a test that flags any feature that predicts the label almost perfectly on its own.

**C7. 🟢 How are downloads made restartable?**
> `download_file` skips any file that already exists, so you just resubmit the SLURM job after a network error. The weakness is that a partially written file also "exists". I should download to a temp name, check the GDC MD5 and then rename.

**C8. 🟡 Walk me through the XGBoost hyperparameters.**
> 200 trees at learning rate 0.05 with depth 3. That's shallow trees and slow learning for about 500 rows. Subsample 0.8 and colsample_bytree 0.3, so each tree sees 30% of features, which stops WSI columns monopolising splits. L1 = 1, L2 = 2 and min_child_weight 5 regularise leaf weights. Objective `multi:softprob` gives calibrated-ish class probabilities for soft voting.

**C9. 🟡 What's a bug in how the MLP is configured?**
> Its size and early stopping are keyed off an `N_PATIENTS` environment variable that the SLURM training script never sets. So it always defaults to 10, giving the small `(256,128,64)` network with early stopping off, regardless of the real cohort size. It should just use `len(X)`.

---

## (d) Underlying ML / CS theory

**D1. 🟡 What is target leakage, and how is it different from train/test contamination?**
> Target leakage is when a feature contains information about the label that wouldn't be available at prediction time, or is literally derived from it. Here that's receptor flags when the label *is* a receptor rule. Train/test contamination is when information from evaluation rows reaches training, for example fitting a scaler or PCA on all rows before CV. The first can make a useless model look perfect. The second usually inflates scores a bit. This project has both, and the first is the serious one.

**D2. 🟡 Explain gradient boosting in your own words.**
> You build an additive model one tree at a time. Each new tree is fit to the negative gradient of the loss with respect to the current predictions, which for squared loss is just the residuals. XGBoost uses a second-order Taylor expansion, so each split is scored with gradients *and* Hessians. It adds L1/L2 penalties on leaf weights and a per-leaf cost γ. The optimal leaf weight is −Σg / (Σh + λ).

**D3. 🟡 Why might mean pooling patch embeddings be a poor choice?**
> It assumes every patch is equally informative. In a slide, maybe 10% of patches show the diagnostic tumour morphology and the rest are stroma, fat or artefacts. Averaging dilutes that signal. Attention-based MIL learns a weight per patch, a softmax over a small network's scores, so informative patches dominate, and it gives you a heatmap for interpretability.

**D4. 🟡 Why ImageNet features for histology? Is that reasonable?**
> As a baseline, yes. Low-level filters for edges, textures and colour blobs transfer. But there's a big domain gap: H&E has no objects, the colour statistics are different, and the important cues are nuclear shape and tissue architecture. Pathology foundation models trained self-supervised on millions of H&E patches (UNI, CTransPath, CONCH) consistently beat ImageNet features on these tasks.

**D5. 🟢 What is Otsu's method?**
> It picks the threshold that maximises the between-class variance, ω₀ω₁(μ₀−μ₁)², of the two groups it creates. Equivalently, it minimises within-class variance. It's ideal when the histogram is bimodal, like tissue vs background saturation.

**D6. 🟡 Why must you standardise before PCA?**
> PCA finds directions of maximum variance, so a column measured in pixels (tens of thousands) has a far larger variance than a 0/1 flag. It will dominate the first components purely because of its units. Standardising puts every feature on unit variance, so PCA reflects correlation structure. My training script applies PCA to a mix of scaled and unscaled columns, which is a bug.

**D7. 🟡 What does a CNV segment mean value represent?**
> It's log2 of the copy ratio relative to the diploid reference. 0 means two copies, about +0.58 is three copies, −1 is one copy. Segmentation (CBS) merges neighbouring probes with similar values into segments. I call gain or loss at ±0.2.

**D8. 🔴 Soft vs hard voting: when does soft voting help?**
> Hard voting takes the majority class. Soft voting averages predicted probabilities. Soft voting helps when the models are reasonably calibrated and make *different* errors. A confident correct model can override an unsure wrong one. With two models, hard voting can't break ties sensibly, so soft is the only real option. It doesn't help if one model is badly calibrated, e.g. an over-confident MLP.

---

## (e) Evaluation & metrics

**E1. 🔴 What are your results?**
> I need to be careful here. The README reports 86% accuracy, 0.78 macro-F1 and 0.95 ROC-AUC on a 3-class version with SMOTE. But the script that produced those isn't in the repo. The committed trainer is 5-class without SMOTE. And when I audited it, I found two problems: two label-defining features leak through the filter, and the ROC-AUC is computed on training data. So I don't treat those numbers as validated. What I can defend is the pipeline itself and the evaluation protocol I'd use to get real numbers.
>
> *Deeper note:* say this proactively. It's far better than the interviewer discovering it.

**E2. 🟡 Why is your ROC-AUC not trustworthy?**
> In `evaluate_model` I call `model.fit(X, y)` and then `predict_proba(X)` on the same data, so it's training-set AUC. Boosted trees nearly memorise training data, so it's inflated. The fix is `cross_val_predict(..., method="predict_proba")` and computing AUC on the out-of-fold probabilities.

**E3. 🟡 Why macro-F1 rather than accuracy?**
> Luminal A dominates breast cancer, so a model that always says Luminal A gets high accuracy. Macro-F1 averages F1 across classes equally, so poor recall on triple-negative, the clinically critical minority, actually shows up.

**E4. 🟡 What baseline would you compare against?**
> A majority-class predictor first. Then single-modality models: clinical-only (excluding receptors), CNV-only, WSI-only. Those tell you what each modality adds. And honestly, the "rule baseline": since the label *is* an IHC rule, IHC alone gets 100%. That shows why you shouldn't give the model IHC inputs.

**E5. 🔴 How would you validate that the model isn't using batch effects?**
> TCGA samples come from many tissue-source sites with different scanners and staining protocols. Slide colour and biospecimen processing stats can identify the site, and sites differ in subtype mix. I'd do grouped CV, holding out whole sites, and compare against random CV. A big drop means shortcut learning. I'd also apply stain normalisation (Macenko) and drop process variables like plate or concentration stats.

**E6. 🟡 Is 5-fold CV on 500 patients enough?**
> It's reasonable, but the variance is high, especially for small classes like HER2-enriched. I'd use repeated stratified CV (5×5 or 10×5) and report mean ± 95% CI. For final claims, a held-out test set fixed before any modelling, or external validation on CPTAC-BRCA.

**E7. 🟡 Your preprocessing is fit before CV. How much does that matter?**
> For a median imputer or a scaler, usually a little, because the statistics barely change between folds. For PCA it's more, since the components are shaped by the test rows. And for anything supervised, like the README's mutual-information feature selection, it can matter a lot. The clean solution is an sklearn `Pipeline` inside `cross_validate`, so every fitted step only sees training folds.

---

## (f) Scaling, production & system design

**F1. 🟡 What were the HPC constraints and how did you work within them?**
> Three main ones: a 40 GB home quota that was nearly full, so packages went to scratch via `pip --target`; scratch auto-deletion after 15 days, so outputs had to be backed up; and a 12-hour GPU wall-time. I split work by partition: downloads and tabular steps on the compute partition, WSI on `gpu_h100_4`. The download script was batch-seeded so I could run many 50-patient jobs.

**F2. 🔴 WSI extraction for 553 slides under a 12-hour limit: does it fit?**
> Not reliably, and that's a real flaw. That's about 78 seconds per slide, and big slides have tens of thousands of patches. The script only writes the combined CSV at the very end and doesn't skip patients already done. A killed job loses the CSV, though the per-patient `.npy` files survive. I'd add skip-if-exists, write results per patient, and use a SLURM job array, so each task handles a slice and they run in parallel across GPUs.

**F3. 🟡 How would you speed up patch extraction?**
> Most of the time goes to OpenSlide reads and JPEG decoding on CPU, not the GPU. I'd use a PyTorch `DataLoader` with several workers doing the reads in parallel, a bigger batch size (16 is tiny for an H100), mixed precision, and possibly cuCIM for GPU-side decoding. And I'd read at the right level to avoid reading 40× data and then downsizing.

**F4. 🟡 How would you serve this model?**
> At inference you'd need the same preprocessing artefacts: the scalers, PCA and feature column order. Right now these are separate pickles and the column order is implicit, which is fragile. I'd package one sklearn `Pipeline` with an explicit feature schema, version it with MLflow, and put the WSI encoder in a separate GPU service that returns embeddings.

**F5. 🟢 What about storage?**
> Raw data was about 200 GB or more, mostly slides. Patch features are much smaller (N×2048 floats per slide), so caching them is the key optimisation. Keeping only embeddings would let you delete raw slides after extraction.

---

## (g) Failure modes & debugging

**G1. 🔴 Suppose your model got 99% accuracy. What would you do?**
> Assume leakage until proven otherwise. First, feature importance: if `eng_is_triple_negative` or `eng_is_luminal` sit at the top, that's the answer. Second, train single-feature models and flag any feature that alone predicts the label almost perfectly. Third, check correlations between every feature and the label. In this pipeline, that check would have caught both leaky features immediately.

**G2. 🟡 A patient's WSI vector is all zeros. What happened?**
> Either every patch failed to read, in which case `extract_features` returns `np.zeros((1, 2048))`, or the patient had no slide and the merge's `fillna(0)` filled it in. Both are silent. I'd log them explicitly and add a `has_wsi` flag.

**G3. 🟡 The downloader keeps "skipping" a file that's corrupt. Why?**
> The skip logic only checks that the file exists. If a previous run was killed mid-download, a truncated file exists and is never re-fetched. The fix is temp-file + rename, plus an MD5 check against GDC metadata.

**G4. 🟡 `train_classifier.py` ran and wrote results, but the labels look perfectly balanced across 5 classes. Why?**
> It probably went into dry-run mode. If `clin_pam50_subtype` is missing or empty, it assigns round-robin dummy labels and carries on. That happens if `assign_subtypes.py` wasn't run, or if `feature_engineering.py` was run before it. That should be a hard failure, not a warning.

**G5. 🟡 Re-running the pipeline gives different results. Why?**
> A few reasons. `assign_subtypes.py` overwrites `multimodal_dataset.csv` in place, so order matters. `find_wsi_files` picks `svs_files[0]` from an unordered glob. And there's no pinned environment. I'd make every step write a new file, sort the globs, and add a lock file.

**G6. 🔴 What happens if a patient has missing HER2 status?**
> `impute_numeric` fills `her2_status_binary` with the cohort median, almost certainly 0, so that patient is treated as HER2-negative. The label rule then calls them Luminal A or Basal-like. So label noise is introduced *before* modelling. Patients with missing receptor status should be excluded from labelling.

---

## (h) "Why didn't you…" / alternatives

**H1. 🟡 Why didn't you use real PAM50 labels?**
> GDC's case API doesn't expose PAM50. I tried, and that's what `fetch_pam50.py` was for. So we used the IHC surrogate. But TCGA PAM50 calls exist in the PanCancer Atlas supplementary data and through TCGAbiolinks. Integrating those is the single most important fix, because then the task becomes predicting a molecular label from non-molecular inputs, which is the interesting question.

**H2. 🟡 Why not fine-tune the CNN end-to-end?**
> With about 500 slide-level labels and tens of thousands of patches per slide, end-to-end fine-tuning would massively overfit and needs MIL anyway, since labels are per slide, not per patch. Frozen features plus a light aggregator is the standard approach at this scale.

**H3. 🟢 Why XGBoost and not a deep model for the tabular part?**
> On small tabular data with mixed types, gradient-boosted trees are consistently as good as or better than neural nets, need less tuning, and give feature importances. The MLP was added for diversity in the ensemble.

**H4. 🟡 Why not use RNA-seq as a modality?**
> It would be the strongest predictor, since PAM50 is literally defined on RNA. But that would also make the problem trivial for PAM50 labels. The scientific goal was morphology and genomics *without* expensive assays. RNA is better used to make the labels than as an input.

**H5. 🟡 Why not stain normalisation?**
> I should have. TCGA slides come from many sites with different H&E protocols, so colour differs systematically by site, which is a batch-effect shortcut. Macenko or Vahadane normalisation, or colour augmentation, before feature extraction is standard.

**H6. 🟡 Why `gdc-client` wasn't used for downloads?**
> I wanted per-patient, per-modality folder structure and the ability to sample cases randomly in Python. The REST API with JSON filters gave me that directly. For bulk transfer of hundreds of GB, `gdc-client` with a manifest is faster and more robust.

---

## (i) Behavioural (STAR format)

**I1. 🟡 Tell me about a time you found a flaw in your own work.**
> **S:** While preparing to present this pipeline, I reviewed the evaluation code end to end. **T:** I wanted to be sure every number was defensible. **A:** I traced where the labels come from and found they're a rule on ER/PR/HER2. Then I checked the leak filter against every engineered column name, and found two features that *are* the label but slip past the keyword list. I also noticed ROC-AUC was computed on training data. **R:** I documented both issues, stopped quoting the headline numbers, and planned the fixes: an explicit feature allow-list, a leakage unit test, real PAM50 labels and out-of-fold AUC. The lesson was to write leakage tests before models.

**I2. 🟡 Tell me about working under infrastructure constraints.**
> **S:** The HPC home quota was 98% full, scratch auto-deleted after 15 days, and GPU jobs were capped at 12 hours. **T:** I had to download about 200 GB and run a GPU feature-extraction pass. **A:** I installed packages to scratch with `pip --target` and a `PYTHONPATH` export. I made downloads idempotent so failed jobs could simply be resubmitted, split downloads into seeded 50-patient batches, and put WSI on the H100 partition. **R:** We got 553 patients downloaded and clinical, CNV and biospecimen processing completed. WSI was the bottleneck, and the lesson I took is that long jobs need checkpoint/resume from day one.

**I3. 🟢 Tell me about handing work over to someone else.**
> **S:** The project had to be handed over mid-way, with WSI extraction still running. **T:** Make sure someone else could finish it without me. **A:** I wrote a handover document covering how to activate the environment, the directory layout, the status of every stage, exact next commands, SLURM gotchas and known caveats, like the labels being a clinical approximation. **R:** It's in the repo as `TCGA_BRCA_Handover.docx`. In hindsight, a README plus a workflow file would have been even better than a Word doc.

**I4. 🟡 Tell me about a technical decision you'd reverse.**
> Concatenating everything and filling missing slides with zero. It was the quickest way to get a model running, but it mixes "missing" with "average". It also lets 2048 image columns swamp the clinical and genomic ones. I'd switch to late fusion with per-modality models, which handles missing modalities naturally and also gives you the modality ablation for free.

**I5. 🟡 How did you prioritise what to build first?**
> Data access and preprocessing first, because nothing works without clean per-patient tables, and those steps were cheap to iterate on. WSI was last, since it was the most expensive and the riskiest. In hindsight I should have pulled the evaluation and label design forward too. Getting labels right is cheaper than any model.

---

## Hostile follow-ups (with honest answers)

**X1. 🔴 "So your 86% accuracy is meaningless?"**
> Not necessarily meaningless, but unverified. That number came from a training script that isn't in the repo. The committed trainer has a leakage bug I can show you. Interestingly, the reported triple-negative recall is only 0.54, which suggests that run *didn't* have the leaky triple-negative flag, since that would give near-perfect recall. But I can't prove that from the repo, so I don't stand behind the number until I re-run with the fixes.

**X2. 🔴 "You're predicting a label that's a deterministic function of your inputs. What's the point?"**
> Fair. With IHC in the inputs, there's no point: the rule is the model. The point only exists once you exclude receptor fields and ask whether CNV and morphology can recover the subtype. Even then, IHC-surrogate labels are noisy. The right framing is predicting PAM50 from H&E and CNV, and that's my top fix.

**X3. 🔴 "Your ROC-AUC is on training data. Did you know that when you put it in the README?"**
> The plot in the committed code is literally titled "train set", so the code was honest about it, but the README presents 0.95 as a headline metric without that context. That's a reporting mistake. The fix is trivial: out-of-fold probabilities from `cross_val_predict`. I'd rather fix it than defend it.

**X4. 🔴 "The README lists files that don't exist. How do I know any of this ran?"**
> The committed scripts are the ones that ran. The handover doc lists their real outputs and sizes: 532 clinical rows, 542 biospecimen, 525 CNV, and 553 patient folders. The `_v2` and `_v3` scripts the README mentions weren't committed. That's a repo-hygiene failure on my part, and the right response is to commit them or remove the claims.

**X5. 🔴 "Your 'genomic instability score' is a weighted sum of z-scores with made-up weights. Justify it."**
> I can't justify the specific weights. They were hand-picked, not fitted or taken from the literature. Also the inputs were already standardised, so ratios like gain/loss divide by values near zero. A tree model would learn those combinations itself, so these features add noise and risk. I'd remove them, or replace them with established measures like fraction of genome altered computed on raw values.

**X6. 🔴 "Your 'tumour vs normal' CNV logic is 'bigger file wins'. Seriously?"**
> Yes, it's a heuristic, and it's wrong in edge cases: several aliquots, or a quiet tumour genome. The TCGA barcode encodes sample type at positions 14–15, so it's a one-line fix to do it properly. I'd also question the per-chromosome "baseline subtraction", since masked segment files are already relative to a reference.

**X7. 🔴 "Did you actually write this code? The first commit is by 'Antigravity'."**
> *(Answer truthfully about your actual role.)* A good honest template: "I used an AI coding assistant to scaffold scripts, which is why that's the commit author. The design decisions are mine: which modalities, how to aggregate WSI, the SLURM layout, label strategy. I ran it on the cluster and debugged it there. And the audit I just walked you through, finding the leakage and the AUC issue, is exactly the kind of review I do on generated code."

**X8. 🔴 "Why should I believe the WSI features are even at 20×?"**
> You shouldn't, for 40× slides. `get_best_level_for_downsample(2)` returns level 0 on a 1/4/16 pyramid, so those slides are tiled at 40×. Magnification varies across the cohort, which adds noise and possibly a scanner/site confound. The fix is to compute the patch size from objective power or MPP at level 0 and resize.

**X9. 🔴 "Biospecimen features like aliquot concentration and plate: biology or batch effect?"**
> Mostly batch effect, honestly. They describe how the lab processed the sample, not the tumour. If processing differed by site and sites differ in subtype mix, the model can learn the site. I'd drop them or test with site-grouped CV. If performance drops a lot, they were a shortcut.

**X10. 🟡 "What would you do in your first week if this were your job?"**
> Day 1: an allow-list of features, a leakage test, and out-of-fold AUC. Day 2: real PAM50 labels from the PanCancer Atlas. Day 3: a sklearn Pipeline inside repeated CV, plus baselines and per-modality ablations. Days 4–5: fix WSI magnification and resumability, and site-grouped CV. Then, and only then, try a pathology foundation model and attention-MIL.
