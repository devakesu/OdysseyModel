#!/usr/bin/env python3
"""
starter/decisive_a_vs_d.py
==========================
ML Hackathon - Network Intrusion Detection
The Decisive Gate Experiment: Model A vs Model D on Full 1.52M Data

Trains:
1. Model A (Current Champion): Clean 1.52M train rows, full 61 features.
2. Model B (Shift-Augmented): 1.52M train rows + 35% realistic perturbations.
3. Model C (Ratio-Centric): Clean 1.52M train rows on 47 scale-invariant features.
4. Model D (Multi-View Tri-Ensemble): 0.35 * A + 0.35 * B + 0.30 * C.

Evaluates on ALL 381,007 validation flows:
- PR-AUC
- F1 at T=0.50
- Cost-optimal T* and Cost at w=20
- FN, FP, Attack Detection Recall
- Attack Category Breakdown (Analysis, Exploits, Fuzzers, Generic, Normal, Worms)
- Environmental Shift Stress Tests (Latency 2x, Bandwidth 3x, Throttle 0.3x, Combined)

If Model D matches or beats Model A on clean validation while preserving robustness,
it automatically promotes Model D to team_model/ and updates metadata.json.
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import argparse
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    f1_score, average_precision_score, recall_score,
    precision_score, confusion_matrix
)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from features import (
    NetworkFeatureExtractor, EnsembleModel, ColumnSubsetClassifier,
    FEATURE_NAMES, CATEGORICAL_COLS
)
from baseline import load_dataset
from robust_training_v2 import augment_traffic_batch, perturb_cols, perturb_combined


def build_base_ensemble(X_tr, y_tr, cat_indices, max_iter=120, learning_rate=0.07):
    """Fits HistGB + LightGBM on the provided feature matrix."""
    m1 = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=40,
        categorical_features=cat_indices, random_state=42
    )
    m1.fit(X_tr, y_tr)

    if HAS_LIGHTGBM:
        m2 = lgb.LGBMClassifier(
            n_estimators=max_iter + 50, learning_rate=learning_rate * 0.85,
            num_leaves=42, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
        m2.fit(X_tr, y_tr)
        return EnsembleModel([m1, m2])
    return m1


def compute_metrics(y_true, y_prob, threshold=0.5, cost_weight=20.0):
    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    f1 = f1_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    cost = cost_weight * fn + fp
    return {
        "threshold": threshold,
        "f1": f1,
        "pr_auc": pr_auc,
        "recall": rec,
        "precision": prec,
        "cost": int(cost),
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn)
    }


def find_optimal_threshold(y_true, y_prob, cost_weight=20.0):
    best_th = 0.5
    min_cost = float("inf")
    best_m = None
    for th in np.linspace(0.01, 0.60, 200):
        m = compute_metrics(y_true, y_prob, threshold=th, cost_weight=cost_weight)
        if m["cost"] < min_cost:
            min_cost = m["cost"]
            best_th = th
            best_m = m
    return best_th, best_m


def evaluate_stress(predict_fn, val_df, y_val, threshold=0.0405):
    scenarios = {
        "Latency Spike (2x)": lambda df: perturb_cols(df, ["dur", "tcprtt", "synack", "ackdat", "Sjit", "Djit"], 2.0),
        "Bandwidth Surge (3x)": lambda df: perturb_cols(df, ["sbytes", "dbytes", "Sload", "Dload", "res_bdy_len"], 3.0),
        "Bandwidth Throttle (0.3x)": lambda df: perturb_cols(df, ["sbytes", "dbytes", "Sload", "Dload"], 0.3),
        "Combined Stress": lambda df: perturb_combined(df)
    }
    recalls = {}
    for name, p_fn in scenarios.items():
        df_mod = p_fn(val_df.copy())
        probs = predict_fn(df_mod)
        preds = (probs >= threshold).astype(int)
        recalls[name] = float(np.mean(preds[y_val == 1] == 1))
    return recalls


def main():
    parser = argparse.ArgumentParser(description="Decisive A vs D Full Data Experiment")
    parser.add_argument("--cost-weight", type=float, default=20.0, help="Cost weight w (default: 20.0)")
    parser.add_argument("--save-winner", action="store_true", default=True, help="Promote winner to team_model")
    args = parser.parse_args()

    t_global = time.time()
    print("=" * 76)
    print(" THE DECISIVE GATE EXPERIMENT: MODEL A vs MODEL D (FULL 1.52M ROWS)")
    print("=" * 76)
    print(f"Cost Weight w: {args.cost_weight:.1f}")
    print("-" * 76)

    # 1. Load Datasets
    t0 = time.time()
    train_df = load_dataset("data/train.csv")
    val_df = load_dataset("data/validation.csv")
    y_train = train_df["Label"].values
    y_val = val_df["Label"].values
    print(f"Loaded datasets in {time.time() - t0:.1f}s.")

    # 2. Extract Primary Features (Model A & Extractor)
    print("\n[*] Fitting primary NetworkFeatureExtractor (61 features) ...")
    t0 = time.time()
    ext_primary = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=False)
    X_train_a = ext_primary.fit_transform(train_df)
    X_val = ext_primary.transform(val_df)
    cat_indices = ext_primary.categorical_indices_
    print(f"    Primary features extracted in {time.time() - t0:.1f}s.")

    # Determine invariant column indices for Model C
    magnitude_cols = [
        "dur", "sbytes", "dbytes", "Spkts", "Dpkts", "Sload", "Dload", "res_bdy_len",
        "log_dur", "log_sbytes", "log_dbytes", "log_Sload", "log_Dload", "log_res_bdy_len"
    ]
    invariant_indices = [i for i, col in enumerate(ext_primary.feature_names_) if col not in magnitude_cols]
    inv_cat_indices = [invariant_indices.index(c) for c in cat_indices if c in invariant_indices]
    print(f"    Scale-invariant features: {len(invariant_indices)} columns identified.")

    # 3. Train Model A (Clean Champion)
    print("\n[*] [1/3] Training Model A (Clean Data, Full 61 Features) on 1.52M rows ...")
    t0 = time.time()
    model_a = build_base_ensemble(X_train_a, y_train, cat_indices, max_iter=120)
    print(f"    Model A training complete in {time.time() - t0:.1f}s.")

    # 4. Train Model B (Shift-Augmented)
    print("\n[*] [2/3] Generating realistic perturbations and training Model B ...")
    t0 = time.time()
    train_df_aug = augment_traffic_batch(train_df, augment_ratio=0.35, seed=42)
    X_train_b = ext_primary.transform(train_df_aug)
    y_train_b = train_df_aug["Label"].values
    model_b = build_base_ensemble(X_train_b, y_train_b, cat_indices, max_iter=120)
    print(f"    Model B training complete on {len(train_df_aug):,} rows in {time.time() - t0:.1f}s.")

    # 5. Train Model C (Ratio-Centric Invariant)
    print("\n[*] [3/3] Training Model C on 47 scale-invariant features ...")
    t0 = time.time()
    X_train_c = X_train_a.iloc[:, invariant_indices]
    model_c_base = build_base_ensemble(X_train_c, y_train, inv_cat_indices, max_iter=100)
    model_c = ColumnSubsetClassifier(model_c_base, invariant_indices)
    print(f"    Model C training complete in {time.time() - t0:.1f}s.")

    # 6. Assemble Model D (Multi-View Tri-Ensemble)
    print("\n[*] Assembling Model D (0.35 * A + 0.35 * B + 0.30 * C) ...")
    model_d = EnsembleModel([model_a, model_b, model_c], weights=[0.35, 0.35, 0.30])

    # -----------------------------------------------------------------------
    # EVALUATION ON ALL 381,007 VALIDATION ROWS
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("              FULL VALIDATION EVALUATION (381,007 ROWS)")
    print("=" * 76)

    # Probabilities
    probs_a = model_a.predict_proba(X_val)[:, 1]
    probs_d = model_d.predict_proba(X_val)[:, 1]

    # Metrics at T=0.50
    m_a_50 = compute_metrics(y_val, probs_a, threshold=0.5, cost_weight=args.cost_weight)
    m_d_50 = compute_metrics(y_val, probs_d, threshold=0.5, cost_weight=args.cost_weight)

    # Cost-optimal threshold
    th_a_opt, m_a_opt = find_optimal_threshold(y_val, probs_a, cost_weight=args.cost_weight)
    th_d_opt, m_d_opt = find_optimal_threshold(y_val, probs_d, cost_weight=args.cost_weight)

    # Environmental stress tests
    p_fn_a = lambda df: model_a.predict_proba(ext_primary.transform(df))[:, 1]
    p_fn_d = lambda df: model_d.predict_proba(ext_primary.transform(df))[:, 1]
    stress_a = evaluate_stress(p_fn_a, val_df, y_val, threshold=th_a_opt)
    stress_d = evaluate_stress(p_fn_d, val_df, y_val, threshold=th_d_opt)

    # Per-category recall
    preds_a = (probs_a >= th_a_opt).astype(int)
    preds_d = (probs_d >= th_d_opt).astype(int)
    cats = sorted(np.unique(val_df["attack_cat"]))

    cat_rec_a = {}
    cat_rec_d = {}
    for c in cats:
        mask = (val_df["attack_cat"] == c)
        if c == "Normal":
            cat_rec_a[c] = float(np.mean(preds_a[mask] == 0))
            cat_rec_d[c] = float(np.mean(preds_d[mask] == 0))
        else:
            cat_rec_a[c] = float(np.mean(preds_a[mask] == 1))
            cat_rec_d[c] = float(np.mean(preds_d[mask] == 1))

    # -----------------------------------------------------------------------
    # COMPARISON TABLE
    # -----------------------------------------------------------------------
    print(f"\n{'Metric / Scenario':<32} {'Model A (Champion)':>20} {'Model D (Tri-Ensemble)':>20} {'Delta (D - A)':>12}")
    print("-" * 88)
    print(f"{'PR-AUC (Primary Metric)':<32} {m_a_opt['pr_auc']:>20.4f} {m_d_opt['pr_auc']:>20.4f} {m_d_opt['pr_auc'] - m_a_opt['pr_auc']:>+12.4f}")
    print(f"{'Standard F1 (T = 0.50)':<32} {m_a_50['f1']:>20.4f} {m_d_50['f1']:>20.4f} {m_d_50['f1'] - m_a_50['f1']:>+12.4f}")
    print(f"{'Cost-Optimal F1 (T*)':<32} {m_a_opt['f1']:>20.4f} {m_d_opt['f1']:>20.4f} {m_d_opt['f1'] - m_a_opt['f1']:>+12.4f}")
    print(f"{'Cost-Optimal Threshold T*':<32} {th_a_opt:>20.4f} {th_d_opt:>20.4f} {th_d_opt - th_a_opt:>+12.4f}")
    print(f"{'Attack Recall (Catch Rate)':<32} {m_a_opt['recall']:>19.2%} {m_d_opt['recall']:>19.2%} {m_d_opt['recall'] - m_a_opt['recall']:>+11.2%}")
    print(f"{'Missed Intrusions (FN)':<32} {m_a_opt['fn']:>20,d} {m_d_opt['fn']:>20,d} {m_d_opt['fn'] - m_a_opt['fn']:>+12,d}")
    print(f"{'False Alarms (FP)':<32} {m_a_opt['fp']:>20,d} {m_d_opt['fp']:>20,d} {m_d_opt['fp'] - m_a_opt['fp']:>+12,d}")
    print(f"{'Total Cost Score (w = 20)':<32} {m_a_opt['cost']:>20,d} {m_d_opt['cost']:>20,d} {m_d_opt['cost'] - m_a_opt['cost']:>+12,d}")
    print("-" * 88)
    print("PER-ATTACK CATEGORY DETECTION RECALL:")
    for c in cats:
        lbl = f"  {c} (Specificity)" if c == "Normal" else f"  {c} (Recall)"
        print(f"{lbl:<32} {cat_rec_a[c]:>19.2%} {cat_rec_d[c]:>19.2%} {cat_rec_d[c] - cat_rec_a[c]:>+11.2%}")
    print("-" * 88)
    print("ENVIRONMENTAL STRESS TEST RECALL:")
    for sc, rec_a in stress_a.items():
        rec_d = stress_d[sc]
        print(f"  {sc:<30} {rec_a:>19.2%} {rec_d:>19.2%} {rec_d - rec_a:>+11.2%}")
    print("=" * 88)

    # -----------------------------------------------------------------------
    # DECISION GATE
    # -----------------------------------------------------------------------
    d_is_winner = (
        (m_d_opt["pr_auc"] >= m_a_opt["pr_auc"] - 0.001) and
        (m_d_opt["cost"] <= m_a_opt["cost"] + 150) and
        (stress_d["Combined Stress"] >= stress_a["Combined Stress"])
    )

    if d_is_winner:
        print("\n[+] VERDICT: MODEL D (TRI-ENSEMBLE) CONFIRMED AS NEW CHAMPION!")
        print("    Model D preserves/improves clean validation metrics while offering superior shift resilience.")
        chosen_model = model_d
        chosen_m = m_d_opt
        chosen_th = th_d_opt
        arch_name = "Multi-View Tri-Ensemble (Clean + Shift-Augmented + Invariant Ratios)"
    else:
        print("\n[*] VERDICT: MODEL A MAINTAINS CHAMPION STATUS ON CLEAN COST/F1.")
        chosen_model = model_a
        chosen_m = m_a_opt
        chosen_th = th_a_opt
        arch_name = "Ensemble (HistGradientBoosting + LightGBM) with Domain Feature Engineering"

    # Package update if requested
    if args.save_winner:
        print(f"\n[*] Packaging winning model ({arch_name}) into team_model/ and odyssey_model/ ...")
        package = {
            "model": chosen_model,
            "feature_extractor": ext_primary,
            "best_threshold": float(chosen_th),
            "cost_weight": float(args.cost_weight),
            "feature_names": ext_primary.feature_names_,
            "metrics": {
                "f1": float(chosen_m["f1"]),
                "pr_auc": float(chosen_m["pr_auc"]),
                "recall": float(chosen_m["recall"]),
                "precision": float(chosen_m["precision"]),
                "cost": int(chosen_m["cost"])
            }
        }
        for dest in ["team_model/model.joblib", "odyssey_model/model.joblib", "person1_model.joblib"]:
            joblib.dump(package, dest)
            print(f"    Saved artifact to: {dest}")

        meta = {
            "team_name": "Odyssey",
            "model_architecture": arch_name,
            "training_rows": len(train_df),
            "validation_rows": len(val_df),
            "decision_threshold": float(chosen_th),
            "cost_weight": float(args.cost_weight),
            "num_features": len(ext_primary.feature_names_),
            "validation_metrics": {
                "f1_score": float(chosen_m["f1"]),
                "pr_auc": float(chosen_m["pr_auc"]),
                "recall": float(chosen_m["recall"]),
                "precision": float(chosen_m["precision"]),
                "false_negatives": int(chosen_m["fn"]),
                "false_positives": int(chosen_m["fp"]),
                "operational_cost": int(chosen_m["cost"])
            }
        }
        for dest in ["team_model/metadata.json", "odyssey_model/metadata.json"]:
            import json
            with open(dest, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    Updated metadata to: {dest}")

    print(f"\nTotal Decisive Experiment Run Time: {time.time() - t_global:.2f}s.")


if __name__ == "__main__":
    main()
