#!/usr/bin/env python3
"""
starter/hour3_tune_w.py
=======================
ML Hackathon - Network Intrusion Detection
Hour 3 Cost Inject: Instant Re-Optimization Tool

When organizers announce the true cost weight w at Hour 3 (e.g. w=35, w=50, w=15):
1. Loads the champion Multi-View Tri-Ensemble from team_model/model.joblib.
2. Computes/uses validation probabilities across all 381,007 validation flows.
3. Rapidly scans decision thresholds to find the exact T* minimizing Cost = w * FN + FP.
4. Compares against standard T=0.50 and reports cost reduction percentage.
5. Updates best_threshold and cost_weight in both team_model/ and odyssey_model/ packages.
6. Updates metadata.json in both directories with the new operational metrics.

Usage:
------
python hour3_tune_w.py --cost-weight 35.0
python hour3_tune_w.py --cost-weight 50.0
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import sys
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, average_precision_score

from features import NetworkFeatureExtractor, EnsembleModel, ColumnSubsetClassifier, FEATURE_NAMES
from baseline import load_dataset


def evaluate_cost_threshold(y_true, y_prob, threshold, cost_weight):
    preds = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cost = cost_weight * fn + fp
    return {
        "threshold": threshold,
        "cost": int(cost),
        "fn": int(fn),
        "fp": int(fp),
        "tp": int(tp),
        "tn": int(tn),
        "recall": recall_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0)
    }


def main():
    parser = argparse.ArgumentParser(description="Hour 3 Cost Re-Optimization")
    parser.add_argument("--cost-weight", "-w", type=float, required=True, help="Newly announced cost weight w (e.g. 35.0, 50.0)")
    parser.add_argument("--model", "-m", type=str, default="team_model/model.joblib", help="Path to champion model artifact")
    parser.add_argument("--val", type=str, default="data/validation.csv", help="Path to validation ground truth")
    args = parser.parse_args()

    t_start = time.time()
    w = args.cost_weight

    print("=" * 72)
    print(" HOUR 3 COST RE-OPTIMIZATION ENGINE")
    print("=" * 72)
    print(f"[*] Newly Announced Cost Penalty (w): {w:.1f}")
    print(f"[*] Cost Objective Function:           Cost = {w:.0f} * FN + FP")
    print(f"[*] Theoretical Bayes Optimal:        T_Bayes = 1 / ({w:.0f} + 1) = {1.0 / (w + 1.0):.4f}")
    print("-" * 72)

    # 1. Load Model Package
    if not os.path.exists(args.model):
        print(f"[!] ERROR: Model artifact not found at {args.model}")
        sys.exit(1)
    
    print(f"[*] Loading champion model package: {args.model} ...")
    package = joblib.load(args.model)
    model = package["model"]
    feature_extractor = package["feature_extractor"]
    old_th = package.get("best_threshold", 0.0604)
    old_w = package.get("cost_weight", 20.0)

    # 2. Load Validation Set
    print(f"[*] Loading validation dataset: {args.val} ...")
    val_df = load_dataset(args.val)
    y_val = val_df["Label"].values

    # 3. Compute Probabilities
    print("[*] Running feature extraction and probability inference ...")
    t_inf = time.time()
    X_val = feature_extractor.transform(val_df)
    probs = model.predict_proba(X_val)[:, 1]
    pr_auc = average_precision_score(y_val, probs)
    print(f"    Inference complete in {time.time() - t_inf:.2f}s across {len(val_df):,} flows (PR-AUC: {pr_auc:.4f}).")

    # 4. Threshold Optimization Scan
    print(f"\n[*] Scanning decision thresholds [0.002 to 0.700] for w = {w:.1f} ...")
    best_th = 0.5
    min_cost = float("inf")
    best_res = None

    threshold_grid = np.linspace(0.002, 0.70, 350)
    for th in threshold_grid:
        res = evaluate_cost_threshold(y_val, probs, th, cost_weight=w)
        if res["cost"] < min_cost:
            min_cost = res["cost"]
            best_th = th
            best_res = res

    # Evaluate at default T=0.50 for comparison
    res_default = evaluate_cost_threshold(y_val, probs, 0.50, cost_weight=w)
    cost_saving = (1.0 - min_cost / res_default["cost"]) * 100.0

    print("\n" + "=" * 72)
    print("                      RE-OPTIMIZATION RESULTS")
    print("=" * 72)
    print(f"Default Threshold (T = 0.50):")
    print(f"  Cost:        {res_default['cost']:,}  (FN = {res_default['fn']:,} * {w:.0f} + FP = {res_default['fp']:,})")
    print(f"  Recall:      {res_default['recall']:.2%} | F1: {res_default['f1']:.4f}")
    print(f"\nOptimal Threshold for w = {w:.1f} (T* = {best_th:.4f}):")
    print(f"  Cost:        {best_res['cost']:,}  (FN = {best_res['fn']:,} * {w:.0f} + FP = {best_res['fp']:,})")
    print(f"  Recall:      {best_res['recall']:.2%} (Detected {best_res['tp']:,} / {best_res['tp'] + best_res['fn']:,} attacks)")
    print(f"  Precision:   {best_res['precision']:.4f} | F1: {best_res['f1']:.4f}")
    print(f"  >>> Operational Risk Reduced by {cost_saving:.1f}%")

    # 5. Update Packages
    print("\n[*] Updating serialized model packages with new calibrated T* ...")
    package["best_threshold"] = float(best_th)
    package["cost_weight"] = float(w)
    package["metrics"]["cost"] = int(best_res["cost"])
    package["metrics"]["recall"] = float(best_res["recall"])
    package["metrics"]["precision"] = float(best_res["precision"])
    package["metrics"]["f1"] = float(best_res["f1"])

    destinations = [
        "team_model/model.joblib",
        "odyssey_model/model.joblib",
        "person1_model.joblib"
    ]
    for dest in destinations:
        if os.path.exists(os.path.dirname(dest)) or os.path.dirname(dest) == "":
            joblib.dump(package, dest)
            print(f"    [OK] Updated: {dest}")

    # 6. Update metadata.json
    for meta_dest in ["team_model/metadata.json", "odyssey_model/metadata.json"]:
        if os.path.exists(meta_dest):
            with open(meta_dest, "r") as f:
                meta = json.load(f)
            meta["decision_threshold"] = float(best_th)
            meta["cost_weight"] = float(w)
            meta["validation_metrics"]["decision_threshold"] = float(best_th)
            meta["validation_metrics"]["operational_cost"] = int(best_res["cost"])
            meta["validation_metrics"]["false_negatives"] = int(best_res["fn"])
            meta["validation_metrics"]["false_positives"] = int(best_res["fp"])
            meta["validation_metrics"]["recall"] = float(best_res["recall"])
            meta["validation_metrics"]["f1_score"] = float(best_res["f1"])
            with open(meta_dest, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    [OK] Updated: {meta_dest}")

    print("\n[+] HOUR 3 RE-OPTIMIZATION COMPLETE: Model package is locked and ready for Challenge data!")
    print(f"Total time elapsed: {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
