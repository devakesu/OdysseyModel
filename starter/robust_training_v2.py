#!/usr/bin/env python3
"""
starter/robust_training_v2.py
=============================
ML Hackathon - Network Intrusion Detection
Robustness Hardening & Multi-View Training (Person 1 & Person 2)

Tests:
- Model A: Standard Champion (Clean data + Full 61 features)
- Model B: Shift-Augmented (Clean + Realistic traffic perturbations during training)
- Model C: Ratio-Centric Invariant (Clean data + Scale-invariant ratios & asymmetries only)
- Model D: Multi-View Tri-Ensemble (Soft blend of A, B, and C)

Evaluates on:
1. Standard Validation PR-AUC, F1, and Cost
2. Zero-Day LOFO Recall on Exploits, Fuzzers, Generic
3. Simulated Network Environmental Shifts (Latency, Bandwidth Surge, Throttling, Connection Flood, Combined)
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import argparse
import time
import copy
import joblib
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

from features import NetworkFeatureExtractor, EnsembleModel, FEATURE_NAMES, CATEGORICAL_COLS
from baseline import load_dataset


def augment_traffic_batch(df, augment_ratio=0.35, seed=42):
    """
    Creates realistic traffic condition perturbations during training.
    Teaches the model to recognize attack patterns across varying bandwidths,
    latencies, jitter rates, and network load conditions.
    """
    np.random.seed(seed)
    n_aug = int(len(df) * augment_ratio)
    aug_idx = np.random.choice(df.index, size=n_aug, replace=False)
    
    df_aug = df.loc[aug_idx].copy()
    
    # 1. Latency & Jitter perturbations (simulate congested routes / distant hops)
    lat_factor = np.random.uniform(0.6, 2.5, size=n_aug)
    jit_factor = np.random.uniform(0.5, 3.0, size=n_aug)
    for c in ["dur", "tcprtt", "synack", "ackdat"]:
        if c in df_aug.columns:
            vals = pd.to_numeric(df_aug[c], errors="coerce").fillna(0.0).values
            df_aug[c] = (vals * lat_factor).astype(str)
            
    for c in ["Sjit", "Djit", "Sintpkt", "Dintpkt"]:
        if c in df_aug.columns:
            vals = pd.to_numeric(df_aug[c], errors="coerce").fillna(0.0).values
            df_aug[c] = (vals * jit_factor).astype(str)
            
    # 2. Bandwidth & Load scaling (simulate gigabit vs throttled links)
    load_factor = np.random.uniform(0.4, 2.5, size=n_aug)
    for c in ["Sload", "Dload", "sbytes", "dbytes", "res_bdy_len"]:
        if c in df_aug.columns:
            vals = pd.to_numeric(df_aug[c], errors="coerce").fillna(0.0).values
            df_aug[c] = (vals * load_factor).astype(str)
            
    # 3. Connection Density variations (simulate busy enterprise switch)
    conn_factor = np.random.uniform(0.6, 2.0, size=n_aug)
    for c in ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"]:
        if c in df_aug.columns:
            vals = pd.to_numeric(df_aug[c], errors="coerce").fillna(0.0).values
            df_aug[c] = np.maximum(1.0, np.round(vals * conn_factor)).astype(str)
            
    return pd.concat([df, df_aug], ignore_index=True)


class MultiViewModel:
    """Combines predictions from multiple submodels with different feature spaces."""
    def __init__(self, submodels_with_extractors, weights=None):
        self.submodels_with_extractors = submodels_with_extractors
        n = len(submodels_with_extractors)
        self.weights = weights if weights is not None else [1.0 / n] * n

    def predict_proba(self, df_raw):
        probs = np.zeros(len(df_raw), dtype=np.float64)
        for (model, extractor), w in zip(self.submodels_with_extractors, self.weights):
            X = extractor.transform(df_raw)
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
            else:
                p = model.predict(X)
            probs += w * p
        return np.vstack([1.0 - probs, probs]).T


def train_classifier(X_tr, y_tr, cat_indices, max_iter=100, learning_rate=0.08):
    """Trains a 2-way HistGB + LightGBM base ensemble."""
    m1 = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=40,
        categorical_features=cat_indices, random_state=42
    )
    m1.fit(X_tr, y_tr)
    
    if HAS_LIGHTGBM:
        m2 = lgb.LGBMClassifier(
            n_estimators=max_iter + 40, learning_rate=learning_rate * 0.85, num_leaves=40,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1
        )
        m2.fit(X_tr, y_tr)
        return EnsembleModel([m1, m2])
    return m1


def compute_eval_metrics(y_true, y_prob, threshold=0.0405, cost_weight=20.0):
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
    return {"f1": f1, "pr_auc": pr_auc, "recall": rec, "precision": prec, "cost": int(cost), "fn": int(fn), "fp": int(fp)}


def evaluate_shift_stress(predict_fn, val_df, y_val, threshold=0.0405):
    """Evaluates a prediction function under multiple environmental perturbations."""
    stress_scenarios = {
        "Clean": lambda df: df,
        "Latency Spike (2x)": lambda df: perturb_cols(df, ["dur", "tcprtt", "synack", "ackdat", "Sjit", "Djit"], 2.0),
        "Bandwidth Surge (3x)": lambda df: perturb_cols(df, ["sbytes", "dbytes", "Sload", "Dload", "res_bdy_len"], 3.0),
        "Bandwidth Throttle (0.3x)": lambda df: perturb_cols(df, ["sbytes", "dbytes", "Sload", "Dload"], 0.3),
        "Connection Flood (2x)": lambda df: perturb_cols(df, ["ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_dport_ltm"], 2.0),
        "Combined Stress": lambda df: perturb_combined(df)
    }
    
    results = {}
    for name, p_fn in stress_scenarios.items():
        df_mod = p_fn(val_df.copy())
        probs = predict_fn(df_mod)
        m = compute_eval_metrics(y_val, probs, threshold=threshold)
        results[name] = m
    return results


def perturb_cols(df, cols, factor):
    for c in cols:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").fillna(0.0).values
            df[c] = (vals * factor).astype(str)
    return df


def perturb_combined(df):
    df = perturb_cols(df, ["dur", "tcprtt", "synack"], 2.0)
    df = perturb_cols(df, ["Sload", "Dload"], 2.5)
    df = perturb_cols(df, ["Sjit", "Djit"], 3.0)
    df = perturb_cols(df, ["ct_srv_src", "ct_dst_ltm"], 2.0)
    return df


def evaluate_lofo_recall(train_df, val_df, model_builder_fn, threshold=0.0405, families=["Exploits", "Fuzzers", "Generic"]):
    """Evaluates zero-day recall across LOFO splits."""
    lofo_scores = {}
    for fam in families:
        tr_sub = train_df[train_df["attack_cat"] != fam].copy()
        pred_fn = model_builder_fn(tr_sub)
        probs = pred_fn(val_df)
        preds = (probs >= threshold).astype(int)
        
        fam_mask = (val_df["attack_cat"] == fam)
        if np.sum(fam_mask) > 0:
            rec = float(np.mean(preds[fam_mask] == 1))
        else:
            rec = 0.0
        lofo_scores[fam] = rec
    return lofo_scores


def main():
    parser = argparse.ArgumentParser(description="Robust Training v2 Experiment")
    parser.add_argument("--sample-train", type=int, default=150000, help="Train samples for experiment")
    parser.add_argument("--sample-val", type=int, default=50000, help="Val samples for experiment")
    parser.add_argument("--save-best", action="store_true", help="Save best performing robust model to team_model")
    args = parser.parse_args()

    print("=" * 75)
    print(" ROBUST TRAINING V2: SHIFT INVARIANCE & ZERO-DAY TOURNAMENT")
    print("=" * 75)
    print(f"Sample Train: {args.sample_train:,} | Sample Val: {args.sample_val:,}")
    print("-" * 75)

    train_df = load_dataset("data/train.csv", nrows=args.sample_train)
    val_df = load_dataset("data/validation.csv", nrows=args.sample_val)
    y_val = val_df["Label"].values

    # -----------------------------------------------------------------------
    # MODEL A: Current Champion (Full 61 Features, Clean Data)
    # -----------------------------------------------------------------------
    print("\n[1/3] Training Model A (Current Champion: Clean Data + Full Features) ...")
    t0 = time.time()
    ext_a = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=False)
    X_tr_a = ext_a.fit_transform(train_df)
    model_a = train_classifier(X_tr_a, train_df["Label"].values, ext_a.categorical_indices_)
    pred_fn_a = lambda df: model_a.predict_proba(ext_a.transform(df))[:, 1]
    print(f"    Trained Model A in {time.time() - t0:.1f}s.")

    # -----------------------------------------------------------------------
    # MODEL B: Shift-Augmented (Clean + Realistic Perturbations during Training)
    # -----------------------------------------------------------------------
    print("\n[2/3] Training Model B (Shift-Augmented: Perturbation Augmented Training) ...")
    t0 = time.time()
    train_df_aug = augment_traffic_batch(train_df, augment_ratio=0.35, seed=42)
    ext_b = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=False)
    X_tr_b = ext_b.fit_transform(train_df_aug)
    model_b = train_classifier(X_tr_b, train_df_aug["Label"].values, ext_b.categorical_indices_)
    pred_fn_b = lambda df: model_b.predict_proba(ext_b.transform(df))[:, 1]
    print(f"    Trained Model B on {len(train_df_aug):,} records in {time.time() - t0:.1f}s.")

    # -----------------------------------------------------------------------
    # MODEL C: Ratio-Centric Invariant (Drops Raw Magnitude Bytes/Durations/Rates)
    # -----------------------------------------------------------------------
    print("\n[3/3] Training Model C (Ratio-Centric Invariant: Asymmetry & Ratios Only) ...")
    t0 = time.time()
    ext_c = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=True)
    X_tr_c = ext_c.fit_transform(train_df)
    model_c = train_classifier(X_tr_c, train_df["Label"].values, ext_c.categorical_indices_)
    pred_fn_c = lambda df: model_c.predict_proba(ext_c.transform(df))[:, 1]
    print(f"    Trained Model C ({X_tr_c.shape[1]} invariant features) in {time.time() - t0:.1f}s.")

    # -----------------------------------------------------------------------
    # MODEL D: Tri-Ensemble (Soft Blend of A, B, and C)
    # -----------------------------------------------------------------------
    pred_fn_d = lambda df: (0.35 * pred_fn_a(df) + 0.35 * pred_fn_b(df) + 0.30 * pred_fn_c(df))

    models_to_test = {
        "Model A (Champion)": pred_fn_a,
        "Model B (Shift-Augmented)": pred_fn_b,
        "Model C (Ratio-Centric)": pred_fn_c,
        "Model D (Tri-Ensemble)": pred_fn_d
    }

    # -----------------------------------------------------------------------
    # STRESS EVALUATION
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("                     ENVIRONMENTAL SHIFT STRESS TEST")
    print("=" * 75)

    stress_table = []
    headers = ["Model", "Clean PR-AUC", "Clean Recall", "Latency 2x Rec", "Bandwidth 3x Rec", "Throttle 0.3x Rec", "Combined Rec"]

    stress_reports = {}
    for name, p_fn in models_to_test.items():
        res = evaluate_shift_stress(p_fn, val_df, y_val, threshold=0.0405)
        stress_reports[name] = res
        stress_table.append([
            name,
            f"{res['Clean']['pr_auc']:.4f}",
            f"{res['Clean']['recall']:.2%}",
            f"{res['Latency Spike (2x)']['recall']:.2%}",
            f"{res['Bandwidth Surge (3x)']['recall']:.2%}",
            f"{res['Bandwidth Throttle (0.3x)']['recall']:.2%}",
            f"{res['Combined Stress']['recall']:.2%}"
        ])

    df_stress = pd.DataFrame(stress_table, columns=headers)
    print(df_stress.to_string(index=False))

    # -----------------------------------------------------------------------
    # ZERO-DAY LOFO TEST
    # -----------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("               ZERO-DAY LOFO GENERALIZATION BENCHMARK")
    print("=" * 75)

    def build_model_a(df_tr):
        ext = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=False)
        m = train_classifier(ext.fit_transform(df_tr), df_tr["Label"].values, ext.categorical_indices_)
        return lambda df: m.predict_proba(ext.transform(df))[:, 1]

    def build_model_b(df_tr):
        df_aug = augment_traffic_batch(df_tr, augment_ratio=0.35, seed=42)
        ext = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=False)
        m = train_classifier(ext.fit_transform(df_aug), df_aug["Label"].values, ext.categorical_indices_)
        return lambda df: m.predict_proba(ext.transform(df))[:, 1]

    def build_model_c(df_tr):
        ext = NetworkFeatureExtractor(top_k_proto=30, drop_leakage_cols=True, invariant_only=True)
        m = train_classifier(ext.fit_transform(df_tr), df_tr["Label"].values, ext.categorical_indices_)
        return lambda df: m.predict_proba(ext.transform(df))[:, 1]

    def build_model_d(df_tr):
        fn_a = build_model_a(df_tr)
        fn_b = build_model_b(df_tr)
        fn_c = build_model_c(df_tr)
        return lambda df: (0.35 * fn_a(df) + 0.35 * fn_b(df) + 0.30 * fn_c(df))

    lofo_table = []
    lofo_headers = ["Model", "Held Exploits Rec", "Held Fuzzers Rec", "Held Generic Rec", "Mean Zero-Day Rec"]

    builders = {
        "Model A (Champion)": build_model_a,
        "Model B (Shift-Augmented)": build_model_b,
        "Model C (Ratio-Centric)": build_model_c,
        "Model D (Tri-Ensemble)": build_model_d
    }

    for name, b_fn in builders.items():
        print(f"[*] Running LOFO for {name} ...")
        lofo_res = evaluate_lofo_recall(train_df, val_df, b_fn, threshold=0.0405)
        mean_rec = np.mean(list(lofo_res.values()))
        lofo_table.append([
            name,
            f"{lofo_res['Exploits']:.2%}",
            f"{lofo_res['Fuzzers']:.2%}",
            f"{lofo_res['Generic']:.2%}",
            f"{mean_rec:.2%}"
        ])

    df_lofo = pd.DataFrame(lofo_table, columns=lofo_headers)
    print("\n" + "=" * 75)
    print(df_lofo.to_string(index=False))
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
