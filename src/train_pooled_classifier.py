#!/usr/bin/env python3
"""
Pool patch-level features into patient-level predictions.

Two pooling strategies:
  --pooling mean       -> average all patches, then LogReg/MLP on top
  --pooling attention  -> trainable attention MIL, weighted sum, then MLP

Usage
    python train_pooled_classifier.py --backbone densenet121 --pooling mean
    python train_pooled_classifier.py --backbone phikon --pooling attention
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

BASE = Path(os.environ.get("TCGA_SCRATCH_ROOT", str(Path.home() / "siddhartha" / "data")))
FEATURES_ROOT = BASE / "features_patch"
LABELS_PATH = BASE / "labels.json"
ATTENTION_ROOT = BASE / "attention_weights"
TABULAR_PATH = BASE / "tabular_features.json"
# expects: {"TCGA-XX-XXXX": "Luminal_A", ...} — built from build_labels.py output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------ data
def load_patient_features(backbone):
    """Returns dict: patient_id -> (N_patches, D) array."""
    out = {}
    for f in (FEATURES_ROOT / backbone).glob("*.npz"):
        pid = f.stem
        data = np.load(f)
        out[pid] = data["features"]
    return out


def load_patient_coords_and_slides(backbone, patient_id):
    """Returns (coords, slide_files). coords is (N_patches, 3): slide_idx, x, y
    (level-0 pixel coords, indexed into slide_files since multi-slide patients
    have their patches merged from >1 file)."""
    f = FEATURES_ROOT / backbone / f"{patient_id}.npz"
    data = np.load(f)
    return data["coords"], data.get("slide_files", np.array([]))


def load_tabular_features():
    """Returns (dict: patient_id -> dict of feature values, sorted list of all feature names)."""
    if not TABULAR_PATH.exists():
        return {}, []
    data = json.loads(TABULAR_PATH.read_text())
    all_keys = sorted({k for v in data.values() for k in v.keys()})
    return data, all_keys


def tabular_to_matrix(tabular_data, all_keys, patient_ids):
    """Builds a (len(patient_ids), len(all_keys)) matrix, missing values -> 0.0."""
    rows = []
    for pid in patient_ids:
        row = [float(tabular_data.get(pid, {}).get(k, 0.0)) for k in all_keys]
        rows.append(row)
    return np.array(rows, dtype=np.float32)


SPLITS_PATH = BASE / "splits.json"


def load_splits():
    with open(SPLITS_PATH) as fh:
        return json.load(fh)


def load_labels():
    with open(LABELS_PATH) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ mean pooling
def mean_pool(feat_dict):
    """dict of (N, D) -> dict of (D,)"""
    return {pid: feats.mean(axis=0) for pid, feats in feat_dict.items()}


# ------------------------------------------------------------------ k selection (EPV rule)
def compute_k_values(labels, patient_ids):
    """
    Events-Per-Variable rule: k should scale with the SMALLEST class size,
    not total N, since minority-class instability is what actually breaks
    a model. Returns (k_conservative, k_strict) = (min_class/10, min_class/20).
    """
    from collections import Counter
    class_counts = Counter(labels[p] for p in patient_ids)
    min_class = min(class_counts.values())
    k_conservative = max(5, min_class // 10)
    k_strict = max(5, min_class // 20)
    print(f"  class counts: {dict(class_counts)}")
    print(f"  smallest class: {min_class} -> k_conservative={k_conservative}, "
          f"k_strict={k_strict} (EPV 10:1 / 20:1 rule)")
    return k_conservative, k_strict


# ------------------------------------------------------------------ feature selection
def select_features(method, X_train, y_train, X_val, k):
    """Fit selector on TRAIN ONLY, apply to both — never let validation
    data influence which features get kept."""
    if method == "none":
        return X_train, X_val

    if method == "selectk":
        selector = SelectKBest(f_classif, k=min(k, X_train.shape[1]))
        X_train_sel = selector.fit_transform(X_train, y_train)
        X_val_sel = selector.transform(X_val)
        return X_train_sel, X_val_sel

    if method == "l1":
        # L1 zeroes out uninformative coefficients entirely (unlike L2) —
        # rank features by max |coef| across classes, keep top k.
        l1 = LogisticRegression(penalty="l1", solver="saga", max_iter=3000,
                                class_weight="balanced", C=0.1)
        l1.fit(X_train, y_train)
        importance = np.abs(l1.coef_).max(axis=0)
        top_k = np.argsort(importance)[::-1][:k]
        return X_train[:, top_k], X_val[:, top_k]

    if method == "xgb":
        booster = xgb.XGBClassifier(n_estimators=100, max_depth=3,
                                    learning_rate=0.1, objective="multi:softprob",
                                    num_class=len(set(y_train)), eval_metric="mlogloss")
        booster.fit(X_train, y_train)
        top_k = np.argsort(booster.feature_importances_)[::-1][:k]
        return X_train[:, top_k], X_val[:, top_k]

    raise ValueError(f"unknown selection method: {method}")


def make_classifier(kind):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, class_weight="balanced")
    if kind == "mlp":
        return MLPClassifier(hidden_layer_sizes=(64,), alpha=0.01,
                             early_stopping=True, max_iter=500, random_state=42)
    if kind == "xgb":
        return xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                 objective="multi:softprob", eval_metric="mlogloss")
    raise ValueError(f"unknown classifier: {kind}")


# ------------------------------------------------------------------ attention MIL
class AttentionMIL(nn.Module):
    """Attention pooling + classifier head. Three variants via constructor args:
      gated=True,  topk=None  -> Gated attention MIL (Ilse et al., 2018) — default
      gated=False, topk=None  -> Plain attention MIL (no sigmoid gate)
      gated=True,  topk=N     -> Gated attention, but only the top-N highest-
                                  scoring patches are kept before the weighted sum
    """
    def __init__(self, in_dim, hidden_dim=128, n_classes=3, gated=True, topk=None):
        super().__init__()
        self.gated = gated
        self.topk = topk
        self.attn_V = nn.Linear(in_dim, hidden_dim)
        if gated:
            self.attn_U = nn.Linear(in_dim, hidden_dim)
        self.attn_w = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: (N_patches, D) — one bag (one patient)
        A_V = torch.tanh(self.attn_V(x))
        if self.gated:
            A_U = torch.sigmoid(self.attn_U(x))
            combined = A_V * A_U
        else:
            combined = A_V
        scores = self.attn_w(combined)  # (N, 1)

        if self.topk is not None and self.topk < x.shape[0]:
            top_idx = torch.topk(scores.squeeze(-1), k=self.topk).indices
            scores = scores[top_idx]
            x = x[top_idx]

        weights = torch.softmax(scores, dim=0)    # (N, 1)
        bag_repr = (weights * x).sum(dim=0)       # (D,)
        logits = self.classifier(bag_repr)
        return logits, weights.squeeze(-1)


def train_attention_mil(train_feats, train_labels, val_feats, val_labels,
                        label_to_idx, in_dim, n_classes, gated=True, topk=None, epochs=30):
    model = AttentionMIL(in_dim, n_classes=n_classes, gated=gated, topk=topk).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # class-balanced loss weights (same spirit as your compute_sample_weight)
    counts = np.bincount([label_to_idx[l] for l in train_labels], minlength=n_classes)
    class_weights = torch.tensor(1.0 / np.maximum(counts, 1), dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_ids = list(train_feats.keys())
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for pid in train_ids:
            x = torch.tensor(train_feats[pid], dtype=torch.float32).to(DEVICE)
            y = torch.tensor([label_to_idx[train_labels[pid]]]).to(DEVICE)
            opt.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits.unsqueeze(0), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1}/{epochs}  loss={total_loss/len(train_ids):.4f}")

    return model


def predict_attention_mil(model, feats_dict, label_to_idx):
    model.eval()
    preds, probs, weights_out = {}, {}, {}
    with torch.no_grad():
        for pid, feats in feats_dict.items():
            x = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
            logits, weights = model(x)
            p = torch.softmax(logits, dim=0).cpu().numpy()
            preds[pid] = p.argmax()
            probs[pid] = p
            weights_out[pid] = weights.cpu().numpy()
    return preds, probs, weights_out


# ------------------------------------------------------------------ evaluation
def evaluate(y_true, y_pred, y_proba, n_classes):
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")
    return f1, auc


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--pooling", required=True, choices=["mean", "attention"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--selection", choices=["none", "selectk", "l1", "xgb"], default="none",
                    help="feature selection method (mean pooling only)")
    ap.add_argument("--k-mode", choices=["conservative", "strict"], default="conservative",
                    help="which EPV-rule k to use when --selection is not 'none'")
    ap.add_argument("--classifier", choices=["logreg", "mlp", "xgb"], default="logreg",
                    help="final classifier (mean pooling only; attention pooling has its own head)")
    ap.add_argument("--attention-variant", choices=["gated", "plain", "topk"], default="gated",
                    help="attention MIL variant (attention pooling only)")
    ap.add_argument("--topk", type=int, default=50,
                    help="patches to keep when --attention-variant topk")
    ap.add_argument("--save-attention", action="store_true",
                    help="save per-patient attention weights for heatmap generation (attention pooling only)")
    ap.add_argument("--modalities", choices=["wsi", "wsi+tabular"], default="wsi",
                    help="wsi+tabular concatenates clinical/biospecimen/CNV features (mean pooling only)")
    ap.add_argument("--out", type=str, default=None, help="path to save results as JSON")
    args = ap.parse_args()

    if args.pooling == "mean" and args.classifier == "mlp" and args.selection == "none":
        print("ERROR: --classifier mlp with --selection none feeds an MLP the full "
              f"{'768-2048'}-dim pooled vector against a small patient count — high overfitting "
              "risk (this exact failure showed up in the original pipeline: MLP F1~0.35 vs "
              "XGBoost F1~0.49 on similar-scale data). Pick a --selection method first "
              "(selectk / l1 / xgb) to cut dimensionality before using --classifier mlp.")
        sys.exit(1)

    if args.modalities == "wsi+tabular" and args.pooling == "attention":
        print("ERROR: --modalities wsi+tabular is only supported with --pooling mean "
              "(attention's classifier head has a fixed WSI-only input size).")
        sys.exit(1)

    print(f"Loading {args.backbone} features + labels + shared splits...")
    feat_dict = load_patient_features(args.backbone)
    labels = load_labels()
    splits = load_splits()

    classes = sorted(set(labels.values()))
    label_to_idx = {c: i for i, c in enumerate(classes)}

    all_patient_ids = sorted(set(feat_dict) & set(labels))
    k_conservative, k_strict = compute_k_values(labels, all_patient_ids)
    k = k_conservative if args.k_mode == "conservative" else k_strict

    tabular_data, tabular_keys = ({}, [])
    if args.modalities == "wsi+tabular":
        tabular_data, tabular_keys = load_tabular_features()
        if not tabular_keys:
            print("ERROR: --modalities wsi+tabular but tabular_features.json not found/empty — "
                  "run build_tabular_features.py first.")
            sys.exit(1)
        print(f"  tabular features: {len(tabular_keys)} columns")

    fold_f1, fold_auc = [], []

    for fold_info in splits["folds"]:
        fold = fold_info["fold"]
        train_ids = [p for p in fold_info["train"] if p in feat_dict]
        val_ids = [p for p in fold_info["val"] if p in feat_dict]
        print(f"\n  fold {fold+1}/{splits['n_folds']}: {len(train_ids)} train, {len(val_ids)} val "
              f"(of {len(fold_info['train'])}/{len(fold_info['val'])} in shared split — "
              f"some may still lack extracted features)")

        if args.pooling == "mean":
            pooled = mean_pool(feat_dict)
            X_train = np.stack([pooled[p] for p in train_ids])
            X_val = np.stack([pooled[p] for p in val_ids])
            y_train = np.array([label_to_idx[labels[p]] for p in train_ids])
            y_val = np.array([label_to_idx[labels[p]] for p in val_ids])

            if args.modalities == "wsi+tabular":
                tab_train = tabular_to_matrix(tabular_data, tabular_keys, train_ids)
                tab_val = tabular_to_matrix(tabular_data, tabular_keys, val_ids)
                X_train = np.concatenate([X_train, tab_train], axis=1)
                X_val = np.concatenate([X_val, tab_val], axis=1)

            X_train, X_val = select_features(args.selection, X_train, y_train, X_val, k)

            clf = make_classifier(args.classifier)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            y_proba = clf.predict_proba(X_val)

        else:  # attention
            train_feats = {p: feat_dict[p] for p in train_ids}
            val_feats = {p: feat_dict[p] for p in val_ids}
            train_labels = {p: labels[p] for p in train_ids}

            in_dim = next(iter(feat_dict.values())).shape[1]
            gated = args.attention_variant in ("gated", "topk")
            topk = args.topk if args.attention_variant == "topk" else None
            model = train_attention_mil(train_feats, train_labels, val_feats, None,
                                        label_to_idx, in_dim, len(classes),
                                        gated=gated, topk=topk)
            preds, probs, weights_out = predict_attention_mil(model, val_feats, label_to_idx)
            y_val = np.array([label_to_idx[labels[p]] for p in val_ids])
            y_pred = np.array([preds[p] for p in val_ids])
            y_proba = np.stack([probs[p] for p in val_ids])

            if args.save_attention:
                out_dir = ATTENTION_ROOT / f"{args.backbone}_{args.attention_variant}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for pid in val_ids:
                    coords, slide_files = load_patient_coords_and_slides(args.backbone, pid)
                    np.savez_compressed(out_dir / f"{pid}.npz",
                                        weights=weights_out[pid], coords=coords,
                                        slide_files=slide_files)
                print(f"    saved attention weights for {len(val_ids)} patients -> {out_dir}")

        f1, auc = evaluate(y_val, y_pred, y_proba, len(classes))
        print(f"    F1 macro={f1:.3f}  ROC-AUC={auc:.3f}")
        fold_f1.append(f1)
        fold_auc.append(auc)

    print(f"\n{'='*50}")
    variant_note = (f" + {args.attention_variant}"
                    f"{f'(k={args.topk})' if args.attention_variant == 'topk' else ''}"
                    if args.pooling == "attention" else "")
    print(f"  {args.backbone} + {args.pooling}{variant_note}"
          f"{' + ' + args.selection + f'(k={k})' if args.pooling == 'mean' and args.selection != 'none' else ''}"
          f"{' + ' + args.classifier if args.pooling == 'mean' else ''}")
    print(f"  folds used : {splits['n_folds']} (shared canonical split)")
    print(f"  F1 macro   : {np.mean(fold_f1):.3f} +/- {np.std(fold_f1):.3f}")
    print(f"  ROC-AUC    : {np.mean(fold_auc):.3f} +/- {np.std(fold_auc):.3f}")

    if args.out:
        result = {
            "backbone": args.backbone,
            "pooling": args.pooling,
            "modalities": args.modalities,
            "attention_variant": args.attention_variant if args.pooling == "attention" else "n/a",
            "topk": args.topk if (args.pooling == "attention" and args.attention_variant == "topk") else None,
            "selection": args.selection if args.pooling == "mean" else "n/a (joint attention)",
            "k": k if (args.pooling == "mean" and args.selection != "none") else None,
            "classifier": args.classifier if args.pooling == "mean" else "attention_head",
            "n_folds": splits["n_folds"],
            "f1_macro_mean": float(np.mean(fold_f1)),
            "f1_macro_std": float(np.std(fold_f1)),
            "roc_auc_mean": float(np.mean(fold_auc)),
            "roc_auc_std": float(np.std(fold_auc)),
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"  -> saved to {args.out}")


if __name__ == "__main__":
    main()
