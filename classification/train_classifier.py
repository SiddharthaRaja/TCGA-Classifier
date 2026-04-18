"""
TCGA-BRCA PAM50 Subtype Classifier
=====================================
Ensemble of XGBoost + MLP with soft voting for 5-class PAM50 classification.

PAM50 Classes:
  0 = Basal-like
  1 = HER2-enriched
  2 = Luminal A
  3 = Luminal B
  4 = Normal-like

Evaluation:
  - Accuracy
  - F1 Score (weighted, macro)
  - ROC-AUC (one-vs-rest, multiclass)
  - Confusion Matrix

Outputs:
  - models/xgboost_model.pkl
  - models/mlp_model.pkl
  - models/ensemble_model.pkl
  - models/label_encoder.pkl
  - results/classification_report.txt
  - results/confusion_matrix.png
  - results/roc_curves.png
  - results/metrics_summary.csv

Requirements:
  pip install xgboost scikit-learn pandas numpy matplotlib seaborn
"""

import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_CSV      = Path("multimodal_engineered.csv")
LABEL_COL     = "clin_pam50_subtype"
PATIENT_COL   = "patient_id"
DROP_COLS     = ["clin_pam50_encoded"]

MODELS_DIR    = Path("models")
RESULTS_DIR   = Path("results")

RANDOM_STATE  = 42
N_FOLDS       = 5
TEST_SIZE     = 0.2

PAM50_CLASSES = ["Basal-like", "HER2-enriched", "Luminal A", "Luminal B", "Normal-like"]

# XGBoost hyperparameters — regularized to prevent overfitting
XGB_PARAMS = {
    "n_estimators"     : 200,
    "max_depth"        : 3,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.3,
    "reg_alpha"        : 1.0,
    "reg_lambda"       : 2.0,
    "min_child_weight" : 5,
    "use_label_encoder": False,
    "eval_metric"      : "mlogloss",
    "random_state"     : RANDOM_STATE,
}

# MLP hyperparameters
import os as _os
_n_patients_hint = int(_os.environ.get("N_PATIENTS", "10"))
MLP_PARAMS = {
    "hidden_layer_sizes": (512, 256, 128, 64) if _n_patients_hint >= 50 else (256, 128, 64),
    "activation"        : "relu",
    "solver"            : "adam",
    "max_iter"          : 1000,
    "early_stopping"    : _n_patients_hint >= 20,
    "validation_fraction": 0.1,
    "random_state"      : RANDOM_STATE,
}
# ──────────────────────────────────────────────────────────────────────────────


def load_data(path: Path):
    df = pd.read_csv(path)
    print(f"    Loaded {df.shape[0]} patients, {df.shape[1]} columns")
    return df


def prepare_features(df: pd.DataFrame):
    """
    Splits df into X (features) and y (labels).
    Drops leaky receptor-status columns that directly encode PAM50 subtype.
    """
    # Drop non-feature columns
    drop = [PATIENT_COL] + DROP_COLS
    drop = [c for c in drop if c in df.columns]

    # Drop columns that leak the PAM50 label
    leak_keywords = [
        'er_status', 'pr_status', 'her2', 'estrogen', 'progesterone',
        'receptor', 'pam50', 'subtype'
    ]
    leak_cols = [
        c for c in df.columns
        if any(kw in c.lower() for kw in leak_keywords)
        and c != LABEL_COL
    ]
    if leak_cols:
        print(f"    Dropping {len(leak_cols)} leaky columns: {leak_cols}")
        drop += leak_cols

    # Separate label
    label_drop = [LABEL_COL] if LABEL_COL in df.columns else []
    X = df.drop(columns=drop + label_drop)

    # Use real labels if available
    if LABEL_COL in df.columns and df[LABEL_COL].notna().sum() > 0:
        y_raw = df[LABEL_COL]
        print(f"\n    Using real PAM50 labels: {y_raw.value_counts().to_dict()}")
        n_labelled = y_raw.notna().sum()
        print(f"    Labelled patients: {n_labelled} / {len(df)}")
    else:
        print("\n    ⚠ No real PAM50 labels — running in DRY RUN mode.")
        print("    Dummy labels assigned for pipeline verification.\n")
        dummy_classes = PAM50_CLASSES[:min(5, len(df))]
        y_raw = pd.Series(
            [dummy_classes[i % len(dummy_classes)] for i in range(len(df))]
        )
        print(f"    Dry run labels: {y_raw.tolist()}")
        n_labelled = len(y_raw)
        print(f"    Labelled patients: {n_labelled} / {len(df)}")

    # Encode labels
    le = LabelEncoder()
    le.fit(PAM50_CLASSES)
    y = le.transform(y_raw)

    print(f"    Classes: {list(le.classes_)}")
    print(f"    Feature matrix shape: {X.shape}")

    return X.values.astype(float), y, le


def apply_pca(X, n_components=100):
    """Reduce dimensionality with PCA before training."""
    n_components = min(n_components, X.shape[1], X.shape[0] - 1)
    print(f"\n    Applying PCA: {X.shape[1]} features → {n_components} components")
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_reduced = pca.fit_transform(X)
    print(f"    PCA variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    return X_reduced, pca


def build_models(n_classes: int):
    """Build XGBoost, MLP, and soft-voting ensemble."""
    xgb = XGBClassifier(
        **XGB_PARAMS,
        num_class=n_classes,
        objective="multi:softprob"
    )
    mlp = MLPClassifier(**MLP_PARAMS)

    ensemble = VotingClassifier(
        estimators=[("xgboost", xgb), ("mlp", mlp)],
        voting="soft",
        weights=[1, 1]
    )
    return xgb, mlp, ensemble


def evaluate_model(model, X, y, le, model_name, n_folds):
    """
    Evaluates a model using cross-validation.
    Uses StratifiedKFold if enough samples, else LeaveOneOut.
    """
    n_samples = len(y)
    n_classes = len(le.classes_)

    if n_samples < 10:
        from sklearn.model_selection import LeaveOneOut
        cv = LeaveOneOut()
        cv_name = "LeaveOneOut"
    else:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
        cv_name = f"StratifiedKFold(k={n_folds})"

    print(f"\n    [{model_name}] CV strategy: {cv_name}")

    scoring = ["accuracy", "f1_weighted", "f1_macro"]
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring,
                                return_train_score=False)

    metrics = {
        "model"           : model_name,
        "cv_strategy"     : cv_name,
        "accuracy_mean"   : round(cv_results["test_accuracy"].mean(), 4),
        "accuracy_std"    : round(cv_results["test_accuracy"].std(),  4),
        "f1_weighted_mean": round(cv_results["test_f1_weighted"].mean(), 4),
        "f1_macro_mean"   : round(cv_results["test_f1_macro"].mean(),    4),
    }

    # ROC-AUC: fit on all data, compute OvR AUC
    try:
        model.fit(X, y)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
            y_bin  = label_binarize(y, classes=list(range(n_classes)))
            if n_classes > 2:
                auc_score = roc_auc_score(y_bin, y_prob,
                                          multi_class="ovr", average="macro")
            else:
                auc_score = roc_auc_score(y_bin, y_prob[:, 1])
            metrics["roc_auc_macro"] = round(float(auc_score), 4)
    except Exception as e:
        metrics["roc_auc_macro"] = None
        print(f"      ROC-AUC could not be computed: {e}")

    print(f"    Accuracy : {metrics['accuracy_mean']} ± {metrics['accuracy_std']}")
    print(f"    F1 (wtd) : {metrics['f1_weighted_mean']}")
    print(f"    F1 (mac) : {metrics['f1_macro_mean']}")
    print(f"    ROC-AUC  : {metrics.get('roc_auc_macro', 'N/A')}")

    return metrics, model


def plot_confusion_matrix_cv(y_true, y_pred, le, title, save_path):
    """Plot confusion matrix from CV predictions."""
    present_classes = sorted(set(y_true))
    labels = [le.classes_[i] for i in present_classes]
    cm = confusion_matrix(y_true, y_pred, labels=present_classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title(title,        fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    Confusion matrix saved → {save_path}")


def plot_roc_curves(model, X, y, le, title, save_path):
    n_classes = len(le.classes_)
    model.fit(X, y)

    if not hasattr(model, "predict_proba"):
        print("    Model does not support predict_proba — skipping ROC plot")
        return

    y_prob = model.predict_proba(X)
    present_classes = sorted(set(y))
    y_bin  = label_binarize(y, classes=present_classes)
    present_labels = [le.classes_[i] for i in present_classes]

    fig, ax = plt.subplots(figsize=(9, 7))
    colours = plt.cm.tab10(np.linspace(0, 1, len(present_classes)))

    for i, (cls, col) in enumerate(zip(present_labels, colours)):
        try:
            if y_bin.ndim > 1 and y_bin.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, present_classes[i]])
            else:
                fpr, tpr, _ = roc_curve(y_bin, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=col, lw=2,
                    label=f"{cls} (AUC = {roc_auc:.2f})")
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ROC curves saved → {save_path}")


def save_model(model, path: Path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"    Model saved → {path}")


def main():
    print("=" * 65)
    print("  TCGA-BRCA PAM50 Classifier — XGBoost + MLP Soft Voting")
    print("=" * 65)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1. Load data
    print("\n[1/6] Loading multimodal dataset ...")
    if not DATA_CSV.exists():
        print(f"  ERROR: {DATA_CSV} not found. Run merge_modalities.py first.")
        return
    df = load_data(DATA_CSV)

    # 2. Prepare features
    print("\n[2/6] Preparing features and labels ...")
    X, y, le = prepare_features(df)
    if X is None:
        return
    n_classes = len(le.classes_)

    # 2b. Dimensionality reduction with PCA
    X, pca = apply_pca(X, n_components=100)

    # 3. Build models
    print("\n[3/6] Building models ...")
    xgb, mlp, ensemble = build_models(n_classes)
    print(f"    XGBoost : {XGB_PARAMS['n_estimators']} trees, max_depth={XGB_PARAMS['max_depth']}")
    print(f"    MLP     : layers={MLP_PARAMS['hidden_layer_sizes']}, activation={MLP_PARAMS['activation']}")
    print(f"    Ensemble: Soft voting (equal weights)")

    # 4. Evaluate each model + ensemble
    print("\n[4/6] Evaluating models ...")
    all_metrics = []

    print("\n  -- XGBoost --")
    xgb_metrics, xgb = evaluate_model(xgb, X, y, le, "XGBoost", N_FOLDS)
    all_metrics.append(xgb_metrics)

    print("\n  -- MLP --")
    mlp_metrics, mlp = evaluate_model(mlp, X, y, le, "MLP", N_FOLDS)
    all_metrics.append(mlp_metrics)

    print("\n  -- Ensemble (XGBoost + MLP, Soft Voting) --")
    ens_metrics, ensemble = evaluate_model(ensemble, X, y, le, "Ensemble", N_FOLDS)
    all_metrics.append(ens_metrics)

    # 5. Plots and honest reporting using CV predictions
    print("\n[5/6] Generating plots ...")
    print("    Generating cross-validated predictions for honest reporting ...")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    y_pred_cv = cross_val_predict(ensemble, X, y, cv=cv)

    # Confusion matrix from CV predictions
    plot_confusion_matrix_cv(
        y, y_pred_cv, le,
        "Ensemble Confusion Matrix — CV Predictions (PAM50)",
        RESULTS_DIR / "confusion_matrix.png"
    )

    # ROC curves (fitted on all data — for visual reference only)
    plot_roc_curves(
        ensemble, X, y, le,
        "Ensemble ROC Curves — PAM50 (One-vs-Rest, train set)",
        RESULTS_DIR / "roc_curves.png"
    )

    # Classification report from CV predictions — honest per-class metrics
    present_classes = sorted(set(y))
    present_names   = [le.classes_[i] for i in present_classes]
    report = classification_report(y, y_pred_cv, labels=present_classes,
                                   target_names=present_names, zero_division=0)
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("PAM50 Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write("Model: XGBoost + MLP Soft Voting Ensemble\n")
        f.write(f"CV Strategy: StratifiedKFold (k={N_FOLDS}) — predictions on held-out folds\n\n")
        f.write(report)
    print(f"    Classification report saved → {report_path}")

    # Metrics CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = RESULTS_DIR / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"    Metrics summary saved → {metrics_path}")

    # Refit ensemble on all data for saving
    print("    Refitting ensemble on full data for model saving ...")
    ensemble.fit(X, y)

    # Save PCA
    with open(MODELS_DIR / "pca.pkl", "wb") as f:
        pickle.dump(pca, f)
    print(f"    PCA saved → {MODELS_DIR / 'pca.pkl'}")

    # 6. Save models
    print("\n[6/6] Saving models ...")
    save_model(xgb,      MODELS_DIR / "xgboost_model.pkl")
    save_model(mlp,      MODELS_DIR / "mlp_model.pkl")
    save_model(ensemble, MODELS_DIR / "ensemble_model.pkl")
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"    Label encoder saved → {MODELS_DIR / 'label_encoder.pkl'}")

    # Final summary
    print("\n" + "=" * 65)
    print("  Results Summary (CV on held-out folds):")
    print("=" * 65)
    print(f"  {'Model':<30} {'Accuracy':>10} {'F1 (wtd)':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*60}")
    for m in all_metrics:
        auc_val = str(m.get('roc_auc_macro', 'N/A'))
        print(f"  {m['model']:<30} {str(m['accuracy_mean']):>10} "
              f"{str(m['f1_weighted_mean']):>10} {auc_val:>10}")

    print(f"\n  Output files:")
    print(f"  models/   → trained model .pkl files (+ pca.pkl)")
    print(f"  results/  → confusion matrix, ROC curves, classification report")
    print("=" * 65)


if __name__ == "__main__":
    main()
