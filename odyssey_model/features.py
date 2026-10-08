"""
starter/features.py
===================
ML Hackathon - Network Intrusion Detection
Production-Grade Feature Engineering Pipeline for Person 1 (Modeling) & Person 3 (Submission)

Features:
1. Categorical encoding with frequency thresholding and OOV handling (proto, state, service).
2. Defense against testbed leakage: Excludes brittle random TCP sequence numbers (stcpb, dtcpb).
3. Log1p transformation on heavy-tailed traffic metrics (bytes, duration, rates, jitter).
4. Domain-engineered network flow ratios to generalize to zero-day and unseen attack families
   (DoS, Reconnaissance, Exploits, Worms, etc.).
"""

import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "32")
import numpy as np
import pandas as pd

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

# Heavy-tailed columns spanning several orders of magnitude
LOG_COLS = [
    "dur", "sbytes", "dbytes", "Sload", "Dload", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "res_bdy_len", "tcprtt", "synack", "ackdat"
]

# Random sequence numbers that cause testbed shortcut memorization
DROP_COLS = ["stcpb", "dtcpb"]


class NetworkFeatureExtractor:
    """
    Self-contained feature extractor that fits on training data and transforms
    any train, validation, challenge, or final test dataset identically.
    """
    def __init__(self, top_k_proto=30, drop_leakage_cols=True):
        self.top_k_proto = top_k_proto
        self.drop_leakage_cols = drop_leakage_cols
        self.cat_levels = {}
        self.feature_names_ = []
        self.categorical_indices_ = []

    def fit(self, df):
        """Learn categorical mappings from training data."""
        self.cat_levels = {}
        for c in CATEGORICAL_COLS:
            if c == "proto":
                top_vals = df[c].astype(str).value_counts().nlargest(self.top_k_proto).index.tolist()
            else:
                top_vals = sorted(df[c].astype(str).unique())
            self.cat_levels[c] = {val: idx for idx, val in enumerate(top_vals)}
        return self

    def transform(self, df):
        """Transform raw input DataFrame into engineered numerical feature matrix."""
        X = pd.DataFrame(index=df.index)

        # 1. Categoricals with Out-of-Vocabulary (OOV) bucket
        for c in CATEGORICAL_COLS:
            mapping = self.cat_levels[c]
            oov_idx = len(mapping)
            X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)

        # 2. Base Numerics (excluding leakage columns if requested)
        for c in NUMERIC_COLS:
            if self.drop_leakage_cols and c in DROP_COLS:
                continue
            X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)

        # 3. Log1p transforms for heavy-tailed metrics
        for c in LOG_COLS:
            if c in X.columns:
                X[f"log_{c}"] = np.log1p(np.maximum(X[c], 0.0)).astype(np.float32)

        # 4. Domain-Informed Flow Ratios & Dynamics
        sbytes = X["sbytes"].values
        dbytes = X["dbytes"].values
        spkts = X["Spkts"].values
        dpkts = X["Dpkts"].values

        # Directional Asymmetry (-1.0 to +1.0)
        X["byte_ratio"] = ((sbytes - dbytes) / (sbytes + dbytes + 1.0)).astype(np.float32)
        X["pkt_ratio"] = ((spkts - dpkts) / (spkts + dpkts + 1.0)).astype(np.float32)

        # Payload Density
        X["bytes_per_spkt"] = (sbytes / (spkts + 1e-4)).astype(np.float32)
        X["bytes_per_dpkt"] = (dbytes / (dpkts + 1e-4)).astype(np.float32)

        # Congestion & Retransmission Loss Rates
        X["loss_ratio_src"] = (X["sloss"].values / (spkts + 1e-4)).astype(np.float32)
        X["loss_ratio_dst"] = (X["dloss"].values / (dpkts + 1e-4)).astype(np.float32)

        # Handshake Latency Breakdown
        tcprtt = X["tcprtt"].values
        X["rtt_synack_ratio"] = (X["synack"].values / (tcprtt + 1e-4)).astype(np.float32)
        X["rtt_ackdat_ratio"] = (X["ackdat"].values / (tcprtt + 1e-4)).astype(np.float32)

        # Traffic Rate Discrepancy
        if "log_Sload" in X.columns and "log_Dload" in X.columns:
            X["load_diff"] = (X["log_Sload"] - X["log_Dload"]).astype(np.float32)

        # Connection Density Ratios (Detecting Scanning / Reconnaissance / DoS)
        X["ct_srv_ratio"] = (X["ct_srv_src"].values / (X["ct_srv_dst"].values + 1e-4)).astype(np.float32)
        X["ct_dst_src_ratio"] = (X["ct_dst_src_ltm"].values / (X["ct_dst_ltm"].values + 1e-4)).astype(np.float32)

        # TCP Flag Probes (Half-open / unacknowledged connections)
        X["is_syn_only"] = (((X["swin"].values > 0) & (X["dwin"].values == 0))).astype(np.float32)

        self.feature_names_ = list(X.columns)
        self.categorical_indices_ = [X.columns.get_loc(c) for c in CATEGORICAL_COLS]
        return X

    def fit_transform(self, df):
        return self.fit(df).transform(df)


class EnsembleModel:
    """Combines predictions from multiple fitted base models via soft probability averaging."""
    def __init__(self, models, weights=None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict_proba(self, X):
        probs = np.zeros(len(X), dtype=np.float64)
        for model, w in zip(self.models, self.weights):
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
            else:
                p = model.predict(X)
            probs += w * p
        return np.vstack([1.0 - probs, probs]).T

