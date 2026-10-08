#!/usr/bin/env python3
"""
starter/compare_models.py
=========================
ML Hackathon - Network Intrusion Detection
Quick Leaderboard Benchmark for Person 1

Compares multiple modeling approaches side-by-side on the exact same validation data:
1. Raw Baseline (HistGradientBoosting on raw 38 features)
2. Feature Engineered HistGradientBoosting
3. Feature Engineered LightGBM
4. Feature Engineered Ensemble (HistGB + LightGBM soft blend)

Usage:
------
python compare_models.py --sample-train 100000
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import argparse
import time
import numpy as np
import pandas as pd
from tabulate import tabulate

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, average_precision_score, recall_score, precision_score, confusion_matrix

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from features import NetworkFeatureExtractor, EnsembleModel
from baseline import load_dataset, preprocess_features, CATEGORICAL_COLS as BASELINE_CATS


def compute_metrics(y_true, y_prob, cost_weight=20.0):
    pr_auc = average_precision_score(y_true, y_prob)
    
    # 1. Metric at 0.50
    p50 = (y_prob >= 0.5).astype(int)
    f1_50 = f1_score(y_true, p50, zero_division=0)
    cm50 = confusion_matrix(y_true, p50, labels=[0, 1])
    tn50, fp50, fn50, tp50 = cm50.ravel()
    cost_50 = cost_weight * fn50 + fp50
    
    # 2. Optimal threshold
    best_th = 0.5
    min_cost = float("inf")
    best_rec = 0.0
    best_f1 = 0.0
    for th in np.linspace(0.01, 0.70, 140):
        p = (y_prob >= th).astype(int)
        cm = confusion_matrix(y_true, p, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        c = cost_weight * fn + fp
        if c < min_cost:
            min_cost = c
            best_th = th
            best_rec = recall_score(y_true, p, zero_division=0)
            best_f1 = f1_score(y_true, p, zero_division=0)
            
    return {
        "pr_auc": pr_auc,
        "f1_50": f1_50,
        "cost_50": int(cost_50),
        "best_th": best_th,
        "best_f1": best_f1,
        "best_recall": best_rec,
        "min_cost": int(min_cost)
    }


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Benchmark")
    parser.add_argument("--sample-train", type=int, default=100000, help="Train samples")
    parser.add_argument("--sample-val", type=int, default=50000, help="Val samples")
    parser.add_argument("--cost-weight", type=float, default=20.0, help="Cost weight w")
    args = parser.parse_args()

    print("=" * 75)
    print(" PERSON 1 - MODEL & FEATURE LEADERBOARD BENCHMARK")
    print("=" * 75)
    print(f"Train Samples: {args.sample_train:,} | Val Samples: {args.sample_val:,} | Cost Weight: {args.cost_weight}")
    print("-" * 75)

    train_df = load_dataset("data/train.csv", nrows=args.sample_train)
    val_df = load_dataset("data/validation.csv", nrows=args.sample_val)

    y_train = train_df["Label"].values
    y_val = val_df["Label"].values

    results = []

    # -------------------------------------------------------------
    # 1. Baseline: Raw Features + HistGB
    # -------------------------------------------------------------
    print("[1/4] Running Raw Baseline (HistGB) ...")
    t0 = time.time()
    X_tr_base, cat_map = preprocess_features(train_df, is_train=True)
    X_val_base, _ = preprocess_features(val_df, cat_levels=cat_map, is_train=False)
    cat_idxs = [X_tr_base.columns.get_loc(c) for c in BASELINE_CATS]
    m_base = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, categorical_features=cat_idxs, random_state=42)
    m_base.fit(X_tr_base, y_train)
    p_base = m_base.predict_proba(X_val_base)[:, 1]
    res_base = compute_metrics(y_val, p_base, cost_weight=args.cost_weight)
    res_base["Model"] = "1. Raw Baseline (HistGB)"
    res_base["FitTime"] = round(time.time() - t0, 1)
    results.append(res_base)

    # -------------------------------------------------------------
    # Feature Extraction with NetworkFeatureExtractor
    # -------------------------------------------------------------
    print("[*] Extracting Domain Features ...")
    extractor = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True)
    X_tr_fe = extractor.fit_transform(train_df)
    X_val_fe = extractor.transform(val_df)
    fe_cat_idxs = extractor.categorical_indices_

    # -------------------------------------------------------------
    # 2. HistGB + Domain Features
    # -------------------------------------------------------------
    print("[2/4] Running HistGB + Domain Features ...")
    t0 = time.time()
    m_fe_hgb = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=40, categorical_features=fe_cat_idxs, random_state=42)
    m_fe_hgb.fit(X_tr_fe, y_train)
    p_fe_hgb = m_fe_hgb.predict_proba(X_val_fe)[:, 1]
    res_fe_hgb = compute_metrics(y_val, p_fe_hgb, cost_weight=args.cost_weight)
    res_fe_hgb["Model"] = "2. HistGB + Domain Features"
    res_fe_hgb["FitTime"] = round(time.time() - t0, 1)
    results.append(res_fe_hgb)

    # -------------------------------------------------------------
    # 3. LightGBM + Domain Features
    # -------------------------------------------------------------
    if HAS_LIGHTGBM:
        print("[3/4] Running LightGBM + Domain Features ...")
        t0 = time.time()
        m_lgb = lgb.LGBMClassifier(n_estimators=180, learning_rate=0.06, num_leaves=40, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        m_lgb.fit(X_tr_fe, y_train)
        p_lgb = m_lgb.predict_proba(X_val_fe)[:, 1]
        res_lgb = compute_metrics(y_val, p_lgb, cost_weight=args.cost_weight)
        res_lgb["Model"] = "3. LightGBM + Domain Features"
        res_lgb["FitTime"] = round(time.time() - t0, 1)
        results.append(res_lgb)

        # -------------------------------------------------------------
        # 4. Ensemble: HistGB + LightGBM
        # -------------------------------------------------------------
        print("[4/4] Running Ensemble (HistGB + LightGBM) ...")
        t0 = time.time()
        ens = EnsembleModel([m_fe_hgb, m_lgb])
        p_ens = ens.predict_proba(X_val_fe)[:, 1]
        res_ens = compute_metrics(y_val, p_ens, cost_weight=args.cost_weight)
        res_ens["Model"] = "4. Ensemble (HistGB + LGBM)"
        res_ens["FitTime"] = round(res_fe_hgb["FitTime"] + res_lgb["FitTime"], 1)
        results.append(res_ens)

    # -------------------------------------------------------------
    # Format Leaderboard Table
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("                        LEADERBOARD RESULTS")
    print("=" * 75)
    
    rows = []
    for r in results:
        rows.append([
            r["Model"],
            f"{r['pr_auc']:.4f}",
            f"{r['f1_50']:.4f}",
            f"{r['cost_50']:,}",
            f"{r['best_th']:.3f}",
            f"{r['best_f1']:.4f}",
            f"{r['best_recall']:.2%}",
            f"{r['min_cost']:,}",
            f"{r['FitTime']}s"
        ])
        
    headers = ["Model", "PR-AUC", "F1 (T=0.5)", "Cost (T=0.5)", "T*", "F1 (T*)", "Recall (T*)", "Cost (T*)", "FitTime"]
    df_out = pd.DataFrame(rows, columns=headers)
    print(df_out.to_string(index=False))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
