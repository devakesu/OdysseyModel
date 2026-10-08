#!/usr/bin/env python3
"""
starter/train_person1.py
========================
ML Hackathon - Network Intrusion Detection
Comprehensive Training & Tuning Script for Person 1 (Modeling & Features)

Features:
- Seamless integration with starter/features.py (NetworkFeatureExtractor).
- Supports HistGradientBoosting, LightGBM, and Weighted Probability Ensemble.
- Asymmetric cost optimization (Cost = w * FN + FP) with class-weighting and threshold tuning.
- Detailed evaluation report including per-attack-category recall breakdown.
- Automatically exports model package for Person 3 and submission predictions.

Usage:
------
# 1. Fast prototyping run (150k training rows, ~3 seconds):
python train_person1.py --sample-train 150000 --model-type lgbm

# 2. Full run with HistGradientBoosting on 100% of data:
python train_person1.py --model-type histgb

# 3. Train high-performing ensemble (HistGB + LightGBM):
python train_person1.py --model-type ensemble

# 4. Retrain at Hour 3 with newly announced cost weight (e.g. w=50):
python train_person1.py --cost-weight 50.0 --model-type ensemble
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import sys
import argparse
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, confusion_matrix
)

# Optional LightGBM import
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from features import NetworkFeatureExtractor, EnsembleModel, FEATURE_NAMES, CATEGORICAL_COLS

TRAIN_VAL_COLS = FEATURE_NAMES + ["attack_cat", "Label"]



def resolve_path(candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return candidates[0] if candidates else ""


def load_dataset(filepath, nrows=None):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    print(f"[*] Loading dataset: {filepath} ...")
    df = pd.read_csv(
        filepath,
        header=None,
        names=TRAIN_VAL_COLS,
        dtype=str,
        keep_default_na=False,
        nrows=nrows
    )
    df["attack_cat"] = df["attack_cat"].str.strip().replace({"": "Normal"})
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    print(f"    Loaded {len(df):,} records ({df.shape[1]} columns).")
    return df


def evaluate_metrics(y_true, y_prob, threshold=0.5, cost_weight=20.0):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    cost = cost_weight * fn + fp
    return {
        "threshold": threshold,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "cost": int(cost)
    }


def find_cost_optimal_threshold(y_true, y_prob, cost_weight=20.0, num_steps=250):
    """Fine-grained search for decision threshold minimizing asymmetric loss."""
    thresholds = np.linspace(0.002, 0.80, num_steps)
    best_th = 0.5
    min_cost = float("inf")
    best_m = None

    for th in thresholds:
        m = evaluate_metrics(y_true, y_prob, threshold=th, cost_weight=cost_weight)
        if m["cost"] < min_cost:
            min_cost = m["cost"]
            best_th = th
            best_m = m

    return best_th, best_m


def print_report(title, metrics, cost_weight):
    print(f"\n--- {title} ---")
    print(f"  Threshold:    {metrics['threshold']:.4f}")
    print(f"  F1-Score:     {metrics['f1']:.4f}")
    print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  Confusion:    TP={metrics['tp']:,} | FP={metrics['fp']:,} | TN={metrics['tn']:,} | FN={metrics['fn']:,}")
    print(f"  Total Cost:   {metrics['cost']:,}  (Cost = {cost_weight:.0f}*FN + FP)")


def print_category_breakdown(attack_cats, y_true, y_pred):
    print("\n" + "-" * 65)
    print(f"  {'Category':<16} {'Count':>8}   {'Metric':<20} {'Score':>8}")
    print("  " + "-" * 55)
    for cat in sorted(np.unique(attack_cats)):
        mask = (attack_cats == cat)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        preds = y_pred[mask]
        if cat == "Normal":
            spec = float(np.mean(preds == 0))
            print(f"  {cat:<16} {cnt:>8,}   Specificity (Passed)   {spec:>7.2%}")
        else:
            rec = float(np.mean(preds == 1))
            print(f"  {cat:<16} {cnt:>8,}   Recall (Detected)      {rec:>7.2%}")
    print("-" * 65)


def build_model(model_type, cat_indices, cost_weight=20.0, use_class_weight=False):
    """Construct gradient boosting model."""
    # Scale positive weight roughly proportional to sqrt(cost_weight) to avoid extreme FP inflation
    pos_weight = np.sqrt(cost_weight) if use_class_weight else 1.0

    if model_type == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=45,
            min_samples_leaf=30,
            categorical_features=cat_indices,
            random_state=42
        )

    elif model_type == "lgbm":
        if not HAS_LIGHTGBM:
            raise RuntimeError("LightGBM is not installed. Use --model-type histgb or install lightgbm.")
        return lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.06,
            num_leaves=45,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    elif model_type == "ensemble":
        m1 = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=40,
            categorical_features=cat_indices,
            random_state=42
        )
        if HAS_LIGHTGBM:
            m2 = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.06,
                num_leaves=45,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            return [m1, m2]
        else:
            # Fallback to 2 HistGB with different tree structures
            m2 = HistGradientBoostingClassifier(
                max_iter=160,
                learning_rate=0.05,
                max_leaf_nodes=60,
                categorical_features=cat_indices,
                random_state=123
            )
            return [m1, m2]

    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Person 1: Model Training & Tuning Pipeline")
    parser.add_argument("--train", type=str, default="data/train.csv", help="Path to train.csv")
    parser.add_argument("--val", type=str, default="data/validation.csv", help="Path to validation.csv")
    parser.add_argument("--sample-train", type=int, default=None, help="Sample N train rows for fast iteration")
    parser.add_argument("--sample-val", type=int, default=None, help="Sample N val rows for fast verification")
    parser.add_argument("--model-type", type=str, default="ensemble", choices=["histgb", "lgbm", "ensemble"], help="Model family")
    parser.add_argument("--cost-weight", type=float, default=20.0, help="FN penalty weight w (default: 20.0)")
    parser.add_argument("--use-class-weight", action="store_true", help="Apply cost-weighted positive class scale")
    parser.add_argument("--save-model", type=str, default="person1_model.joblib", help="Output path for saved model artifact")
    parser.add_argument("--output-sub", type=str, default="submission_person1.csv", help="Output path for validation predictions")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 70)
    print(" PERSON 1 - INTRUSION DETECTION MODELING PIPELINE")
    print("=" * 70)
    print(f"[*] Model Family:       {args.model_type.upper()}")
    print(f"[*] Cost Weight w:      {args.cost_weight:.1f}")
    print(f"[*] Class Weighting:    {args.use_class_weight}")
    if args.sample_train:
        print(f"[!] Fast Mode:          Sampling {args.sample_train:,} training records.")
    print("-" * 70)

    # 1. Load Data
    train_path = resolve_path([args.train, "data/train.csv", "../data/train.csv"])
    val_path = resolve_path([args.val, "data/validation.csv", "../data/validation.csv"])

    train_df = load_dataset(train_path, nrows=args.sample_train)
    val_df = load_dataset(val_path, nrows=args.sample_val)

    y_train = train_df["Label"].values
    y_val = val_df["Label"].values

    # 2. Extract Features
    print("\n[*] Fitting Feature Extractor & Transforming Data ...")
    t_feat = time.time()
    extractor = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True)
    X_train = extractor.fit_transform(train_df)
    X_val = extractor.transform(val_df)
    print(f"    Features generated: {X_train.shape[1]} features in {time.time() - t_feat:.2f}s.")

    cat_indices = extractor.categorical_indices_

    # 3. Model Training
    print(f"\n[*] Training {args.model_type.upper()} model ...")
    t_train = time.time()
    model_obj = build_model(args.model_type, cat_indices, cost_weight=args.cost_weight, use_class_weight=args.use_class_weight)

    if args.model_type == "ensemble":
        trained_submodels = []
        for i, m in enumerate(model_obj):
            print(f"    Fitting ensemble component {i + 1}/{len(model_obj)} ({type(m).__name__}) ...")
            m.fit(X_train, y_train)
            trained_submodels.append(m)
        final_model = EnsembleModel(trained_submodels)
    else:
        model_obj.fit(X_train, y_train)
        final_model = model_obj

    print(f"    Training completed in {time.time() - t_train:.2f}s.")

    # 4. Evaluation & Optimization
    print("\n" + "=" * 70)
    print(" VALIDATION EVALUATION RESULTS")
    print("=" * 70)
    y_val_probs = final_model.predict_proba(X_val)[:, 1]

    # Standard T = 0.50
    m_default = evaluate_metrics(y_val, y_val_probs, threshold=0.5, cost_weight=args.cost_weight)
    print_report("Standard Threshold (T = 0.50)", m_default, args.cost_weight)

    # Cost-Optimized Threshold
    best_thresh, m_opt = find_cost_optimal_threshold(y_val, y_val_probs, cost_weight=args.cost_weight)
    cost_reduction = (1.0 - m_opt["cost"] / m_default["cost"]) * 100.0
    print_report(f"Cost-Optimized Threshold (T* = {best_thresh:.4f})", m_opt, args.cost_weight)
    print(f"  >>> Operational Cost Reduced by {cost_reduction:.1f}%")

    # Category Breakdown
    print_category_breakdown(val_df["attack_cat"].values, y_val, (y_val_probs >= best_thresh).astype(int))

    # 5. Save Artifact for Person 3
    model_package = {
        "model": final_model,
        "feature_extractor": extractor,
        "best_threshold": float(best_thresh),
        "cost_weight": float(args.cost_weight),
        "feature_names": extractor.feature_names_,
        "metrics": {
            "f1": float(m_opt["f1"]),
            "pr_auc": float(m_opt["pr_auc"]),
            "recall": float(m_opt["recall"]),
            "precision": float(m_opt["precision"]),
            "cost": int(m_opt["cost"])
        }
    }
    joblib.dump(model_package, args.save_model)
    print(f"\n[+] Saved full model package to: {args.save_model}")

    # 6. Generate Validation Submission
    val_preds = (y_val_probs >= best_thresh).astype(int)
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(val_preds) + 1),
        "prediction": val_preds,
        "probability": np.round(y_val_probs, 6)
    })
    sub_df.to_csv(args.output_sub, index=False)
    print(f"[+] Saved submission predictions to: {args.output_sub} ({len(sub_df):,} rows)")
    print(f"\nTotal elapsed time: {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
