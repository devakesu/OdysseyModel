#!/usr/bin/env python3
"""
light_bulb_model/predict.py
===========================
ML Hackathon  -  Network Intrusion Detection
Team Light Bulb — Multi-View Tri-Ensemble Inference Script

Organizers: This is the official Team Light Bulb model inference script.
Organizers will test your model by running:
    python predict.py --input path/to/test_features.csv --output predictions.csv

This script:
1. Loads your saved model artifact (e.g. baseline_model.joblib).
2. Reads the unlabelled input CSV containing the 38 competition features.
3. Applies your feature preprocessing.
4. Generates predictions using your chosen decision threshold.
5. Saves the output in the required CSV format (row_id,prediction,probability).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "16")
import argparse
import joblib
import numpy as np
import pandas as pd

# Ensure local directory is on sys.path so features.py is always importable
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import features module if available
try:
    from features import NetworkFeatureExtractor, EnsembleModel
except ImportError:
    pass

FEATURE_NAMES = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]


def preprocess(df, cat_levels):
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL_COLS:
        mapping = cat_levels.get(c, {})
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    return X


def main():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input unlabelled CSV (38 feature columns)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save output prediction CSV")
    parser.add_argument("--model", "-m", type=str, default="model.joblib", help="Path to saved model artifact")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Override decision threshold")
    parser.add_argument("--cost-weight", "-w", type=float, default=None, help="Asymmetric cost penalty weight (recalculates T* = 1 / (w + 1))")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for cand in [
            os.path.join(script_dir, model_path),
            os.path.join(script_dir, "..", model_path),
            os.path.join(script_dir, "model.joblib"),
            os.path.join(script_dir, "person1_model.joblib"),
            os.path.join(script_dir, "..", "person1_model.joblib"),
            os.path.join(script_dir, "baseline_model.joblib"),
            os.path.join(script_dir, "..", "baseline_model.joblib")
        ]:
            if os.path.exists(cand):
                model_path = cand
                break

    if not os.path.exists(model_path):
        print(f"[!] Error: Model artifact not found: {args.model}")
        sys.exit(1)

    print(f"[*] Loading model artifact: {model_path} ...")
    package = joblib.load(model_path)
    model = package["model"]
    feature_extractor = package.get("feature_extractor", None)
    cat_levels = package.get("cat_levels", {})

    # Determine decision threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"[*] Decision threshold explicitly set: T = {threshold:.4f}")
    elif args.cost_weight is not None:
        if abs(package.get("cost_weight", 0) - args.cost_weight) < 1e-3:
            threshold = package.get("best_threshold", 1.0 / (args.cost_weight + 1.0))
            print(f"[*] Using calibrated threshold for w = {args.cost_weight:.1f}: T* = {threshold:.4f}")
        else:
            threshold = 1.0 / (args.cost_weight + 1.0)
            print(f"[*] Decision threshold dynamically computed for w = {args.cost_weight:.1f}: T* = {threshold:.4f}")
    else:
        threshold = package.get("best_threshold", 0.0600)
        print(f"[*] Using model package calibrated threshold: T* = {threshold:.4f}")

    print(f"[*] Reading test features: {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        first_line = f.readline().strip().lower()

    if "proto" in first_line:
        print("    Detected CSV with column headers.")
        raw_df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
        if set(FEATURE_NAMES).issubset(raw_df.columns):
            test_df = raw_df[FEATURE_NAMES]
        else:
            test_df = raw_df.iloc[:, :38]
            test_df.columns = FEATURE_NAMES
    else:
        raw_df = pd.read_csv(args.input, header=None, dtype=str, keep_default_na=False)
        if raw_df.shape[1] > 38:
            test_df = raw_df.iloc[:, :38]
        else:
            test_df = raw_df
        test_df.columns = FEATURE_NAMES

    print(f"    Loaded {len(test_df):,} records ({test_df.shape[1]} features).")

    print("[*] Running feature preprocessing ...")
    if feature_extractor is not None:
        X_test = feature_extractor.transform(test_df)
    else:
        X_test = preprocess(test_df, cat_levels)

    print(f"[*] Generating predictions (Threshold = {threshold:.4f}) ...")
    probs = model.predict_proba(X_test)[:, 1]
    probs = np.clip(probs, 0.0, 1.0)
    preds = (probs >= threshold).astype(int)

    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(preds) + 1),
        "prediction": preds,
        "probability": np.round(probs, 6)
    })
    sub_df.to_csv(args.output, index=False)
    print(f"[+] Predictions saved successfully to: {args.output}")
    print(f"    Total predictions: {len(sub_df):,} | Positive rate: {sub_df['prediction'].mean():.2%}")


if __name__ == "__main__":
    main()
