#!/usr/bin/env python3
"""
starter/submit.py
=================
ML Hackathon - Network Intrusion Detection
1-Click Automated Submission Pipeline (for Hours 4 & 5)

Performs:
1. Loads unlabelled test/challenge CSV (headerless, 38 competition features).
2. Runs feature extraction with zero training-serving skew.
3. Generates calibrated probabilities and binary predictions using optimal threshold T*.
4. Strictly validates submission requirements:
   - Header: row_id,prediction,probability
   - Exact row count check (254,005 for challenge, 381,007 for final test)
   - Sequential 1-based row_id (1 to N)
   - Strictly binary predictions (0 or 1)
   - Normalized probabilities in [0.0, 1.0]
   - No NaNs, nulls, or infs
5. Automatically executes evaluate.py --validate-only to confirm competition validity.

Usage:
------
# Hour 4 (Challenge dataset):
python submit.py --input path/to/challenge.csv --stage challenge

# Hour 5 (Final test dataset):
python submit.py --input path/to/final_test.csv --stage test

# Custom / verification run:
python submit.py --input path/to/features.csv --output custom_pred.csv
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import sys
import argparse
import time
import subprocess
import joblib
import numpy as np
import pandas as pd

from features import NetworkFeatureExtractor, EnsembleModel, FEATURE_NAMES

STAGE_ROWS = {
    "challenge": 254005,
    "test": 381007
}


def validate_dataframe(sub_df, expected_rows=None):
    """Run internal sanity assertions before writing to disk."""
    print("[*] Running pre-submission data integrity checks ...")
    
    # 1. Row count
    if expected_rows is not None and len(sub_df) != expected_rows:
        raise ValueError(f"FATAL: Row count mismatch! Expected {expected_rows:,}, got {len(sub_df):,}.")
    print(f"    [OK] Row count verified: {len(sub_df):,} records.")
    
    # 2. Columns
    expected_cols = ["row_id", "prediction", "probability"]
    if list(sub_df.columns) != expected_cols:
        raise ValueError(f"FATAL: Column header mismatch! Expected {expected_cols}, got {list(sub_df.columns)}")
    print(f"    [OK] Column schema verified: {list(sub_df.columns)}.")
    
    # 3. Nulls
    if sub_df.isnull().any().any():
        null_counts = sub_df.isnull().sum().to_dict()
        raise ValueError(f"FATAL: Missing (NaN) values detected: {null_counts}")
    print("    [OK] No missing or NaN values detected.")
    
    # 4. Binary predictions
    unique_preds = set(sub_df["prediction"].unique())
    if not unique_preds.issubset({0, 1}):
        raise ValueError(f"FATAL: Prediction column contains non-binary values: {unique_preds}")
    print(f"    [OK] Predictions are strictly binary integer (0 or 1). Positive rate: {sub_df['prediction'].mean():.2%}")
    
    # 5. Probabilities
    probs = sub_df["probability"].values
    if (probs < 0.0).any() or (probs > 1.0).any():
        raise ValueError("FATAL: Probability column contains values outside [0.0, 1.0].")
    print("    [OK] Probabilities valid in range [0.0, 1.0].")
    
    # 6. row_id sequence
    row_ids = sub_df["row_id"].values
    if not np.array_equal(row_ids, np.arange(1, len(sub_df) + 1)):
        raise ValueError("FATAL: row_id is not strictly sequential 1 to N.")
    print("    [OK] row_id is strictly sequential 1 to N.")


def main():
    parser = argparse.ArgumentParser(description="1-Click Submission Generator & Validator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input unlabelled CSV (38 features)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output submission CSV")
    parser.add_argument("--stage", "-s", type=str, choices=["challenge", "test", "custom"], default="custom",
                        help="Competition stage: 'challenge' (254,005 rows) or 'test' (381,007 rows)")
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to saved model artifact")
    parser.add_argument("--threshold", "-t", type=float, default=None, help="Override decision threshold (default: uses artifact T*)")
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 72)
    print(" 1-CLICK COMPETITION SUBMISSION GENERATOR & VALIDATOR")
    print("=" * 72)
    print(f"[*] Input file:         {args.input}")
    print(f"[*] Stage:              {args.stage.upper()}")

    # Determine default output filename
    if args.output is None:
        if args.stage == "challenge":
            args.output = "submission_challenge.csv"
        elif args.stage == "test":
            args.output = "submission_final.csv"
        else:
            args.output = "submission_output.csv"
    print(f"[*] Output destination: {args.output}")

    # Resolve model artifact
    model_candidates = [
        args.model or "",
        "team_model/model.joblib",
        "lightbulb_model/model.joblib",
        "../team_model/model.joblib",
        "../lightbulb_model/model.joblib"
    ]
    model_path = ""
    for cand in model_candidates:
        if cand and os.path.exists(cand):
            model_path = cand
            break

    if not model_path:
        print("[!] FATAL ERROR: No model artifact found. Train a model first!")
        sys.exit(1)
    print(f"[*] Model artifact:     {model_path}")

    # 1. Load Model Package
    print("\n[*] Loading serialized model artifact ...")
    package = joblib.load(model_path)
    model = package["model"]
    feature_extractor = package.get("feature_extractor", None)
    best_threshold = args.threshold if args.threshold is not None else package.get("best_threshold", 0.5)
    print(f"    Loaded model ({type(model).__name__}) with Decision Threshold T* = {best_threshold:.4f}")

    # 2. Read Test Input
    if not os.path.exists(args.input):
        print(f"[!] FATAL ERROR: Input file does not exist: {args.input}")
        sys.exit(1)

    print(f"\n[*] Reading test features: {args.input} ...")
    t_load = time.time()
    test_df = pd.read_csv(
        args.input,
        header=None,
        names=FEATURE_NAMES,
        dtype=str,
        keep_default_na=False
    )
    print(f"    Loaded {len(test_df):,} records in {time.time() - t_load:.2f}s.")

    expected_rows = STAGE_ROWS.get(args.stage, None)
    if expected_rows and len(test_df) != expected_rows:
        print(f"[!] WARNING: Input row count ({len(test_df):,}) does not match expected {expected_rows:,} rows for {args.stage} stage!")

    # 3. Transform Features
    print("\n[*] Transforming features with NetworkFeatureExtractor ...")
    t_feat = time.time()
    if feature_extractor is not None:
        X_test = feature_extractor.transform(test_df)
    else:
        # Fallback to base encoding
        cat_levels = package.get("cat_levels", {})
        X_test = pd.DataFrame(index=test_df.index)
        for c in ["proto", "state", "service"]:
            mapping = cat_levels.get(c, {})
            X_test[c] = test_df[c].astype(str).map(mapping).fillna(len(mapping)).astype(int)
        for c in [col for col in FEATURE_NAMES if col not in ["proto", "state", "service"]]:
            X_test[c] = pd.to_numeric(test_df[c], errors="coerce").fillna(0.0).astype(np.float32)
    print(f"    Transformed {X_test.shape[1]} features in {time.time() - t_feat:.2f}s.")

    # 4. Predict
    print(f"\n[*] Generating predictions (Threshold T* = {best_threshold:.4f}) ...")
    t_pred = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= best_threshold).astype(int)
    print(f"    Generated {len(preds):,} predictions in {time.time() - t_pred:.2f}s.")
    print(f"    Predicted Attacks: {int(np.sum(preds == 1)):,} ({np.mean(preds == 1):.2%}) | Normal: {int(np.sum(preds == 0)):,}")

    # 5. Format Submission
    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(preds) + 1),
        "prediction": preds,
        "probability": np.round(probs, 6)
    })

    # 6. Validate Data
    validate_dataframe(sub_df, expected_rows=expected_rows)

    # 7. Write to Disk
    sub_df.to_csv(args.output, index=False)
    print(f"\n[+] Submission CSV saved: {args.output}")

    # 8. External Verification with evaluate.py
    eval_script = "starter/evaluate.py"
    if not os.path.exists(eval_script):
        eval_script = "evaluate.py"

    if os.path.exists(eval_script):
        print(f"\n[*] Executing official validator: {eval_script} --validate-only ...")
        cmd = [
            sys.executable, eval_script,
            "--submission", args.output,
            "--validate-only"
        ]
        if expected_rows:
            cmd.extend(["--expected-rows", str(expected_rows)])
            
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"[!] Validation script returned error: {result.stderr}")
            sys.exit(1)
        else:
            print("[+] OFFICIAL VALIDATOR CONFIRMED: Submission is 100% compliant and ready!")

    print(f"First 3 rows:\n{sub_df.head(3).to_string(index=False)}")
    print(f"Last 3 rows:\n{sub_df.tail(3).to_string(index=False)}")
    print(f"\nTotal pipeline elapsed time: {time.time() - t_start:.2f}s.")


if __name__ == "__main__":
    main()
