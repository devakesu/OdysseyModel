# 📊 Network Intrusion Detection: Comprehensive Model & Peer Comparison Report
**Team Odyssey | Technical Evaluation, Model Report, & Comparative Defense**

---

## Executive Summary

In enterprise Security Operations Centers (SOC), network intrusion detection operates under severe **asymmetric risk**: missing an active breach (False Negative, FN) carries catastrophic consequences compared to the investigative overhead of triaging a benign alert (False Positive, FP). In this benchmark, the operational penalty is defined as:

$$\text{Operational Cost} = 20 \times \text{False Negatives (FN)} + 1 \times \text{False Positives (FP)}$$

This document presents a comprehensive evaluation of **Team Odyssey's Multi-View Tri-Ensemble** pipeline alongside an in-depth comparative analysis against an external peer solution (a cost-optimized LightGBM pipeline). 

Both solutions were evaluated on the identical, unmanipulated validation dataset of **$381,007$ network flows** ($44,040$ malicious attack flows and $336,967$ benign traffic records).

### Primary Benchmark Takeaway
* **Peer Best Model (LightGBM + Ratios)**: Achieved an operational cost of **$3,447$** with **$7$ missed intrusions** ($-9.84\%$ cost reduction vs. baseline).
* **Team Odyssey Champion (Multi-View Tri-Ensemble)**: Achieved an operational cost of **$3,431$** with only **$6$ missed intrusions** ($-10.25\%$ cost reduction vs. baseline).
* **Generalization Advantage**: Team Odyssey achieved superior detection recall (**$99.986\%$**) while **eliminating synthetic testbed leakage features** (`stcpb` and `dtcpb`), utilizing bounded ratio math, and demonstrating validated resilience against zero-day attack families and network traffic drift.

---

## ⚡ 1. Head-to-Head Performance Benchmark

The following table summarizes validation performance on all **$381,007$ records** evaluated at cost-calibrated decision thresholds ($w = 20$):

| Model / Pipeline Configuration | Architectures & Key Enhancements | Optimal Threshold ($T^*$) | False Positives (FP) | False Negatives (FN) | Attack Detection Recall | Validation Operational Cost | Cost Reduction vs Baseline |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Competition Baseline** | Default `HistGradientBoosting` (Raw 38 features) | $0.0525$ | $3,343$ | $24$ | $99.945\%$ | $3,823$ | Reference |
| **Peer: LightGBM (State Decomposition)** | LightGBM + 14 one-hot state flags | $0.0298$ | $3,306$ | $9$ | $99.980\%$ | $3,486$ | $-8.82\%$ |
| **Peer: LightGBM (Interactions)** | LightGBM + 2 pairwise multiplicative features | $0.0199$ | $3,335$ | $7$ | $99.984\%$ | $3,475$ | $-9.10\%$ |
| **Peer: LightGBM (Ratios)** | LightGBM + 4 unbounded payload ratios | $0.0298$ | **$3,307$** | $7$ | $99.984\%$ | $3,447$ | $-9.84\%$ |
| **Team Odyssey (Clean View Branch)** | `HistGB` + `LightGBM` soft blend (Clean Data, 61 features) | $0.0405$ | $3,320$ | $10$ | $99.977\%$ | $3,520$ | $-7.93\%$ |
| **Team Odyssey (Ratio-Centric Branch)** | Scale-Invariant Ratio Model (47 Invariant features) | $0.0450$ | $3,382$ | $12$ | $99.973\%$ | $3,622$ | $-5.26\%$ |
| **Team Odyssey: Champion** | **Multi-View Tri-Ensemble** (Clean + Shift-Augmented + Invariant Ratios) | **$0.0604$** | $3,311$ | **$6$** | **$99.986\%$** | **$3,431$** | **-10.25%** |

---

## 🛠️ 2. Architectural & Feature Engineering Breakdown

### A. Feature Inventory Comparison

| Feature Category | Peer Model Inventory (58 Features) | Team Odyssey Inventory (61 Features) | Technical Rationale & Impact |
| :--- | :--- | :--- | :--- |
| **Testbed Leakage Mitigation** | **None** (Retained `stcpb` & `dtcpb`) | **Explicitly Dropped `stcpb` & `dtcpb`** | Initial 32-bit TCP sequence numbers contain PRNG seed artifacts from synthetic traffic generators. Dropping them prevents brittle testbed shortcut memorization. |
| **Heavy-Tailed Transformations** | $\log(x + 1)$ on 11 columns (`dur`, `sbytes`, `dbytes`, `Sload`, `Dload`, `Spkts`, `Dpkts`, `Sjit`, `Djit`, `Sintpkt`, `Dintpkt`) | $\log(x + 1)$ on **13 columns** (Includes the peer 11 + `res_bdy_len`, `tcprtt`, `synack`, `ackdat`) | Application body length and TCP handshake timings span up to 8 orders of magnitude; normalizing them optimizes histogram split binning. |
| **Traffic Asymmetry & Ratios** | Unbounded ratio divisions: $\frac{\text{sbytes}}{\text{dbytes} + 10^{-5}}$, $\frac{\text{Spkts}}{\text{Dpkts} + 10^{-5}}$, payload/pkt ratios | **Bounded Normalized Ratios**: $\frac{\text{sbytes} - \text{dbytes}}{\text{sbytes} + \text{dbytes} + 1.0} \in [-1, 1]$, $\frac{\text{Spkts} - \text{Dpkts}}{\text{Spkts} + \text{Dpkts} + 1.0} \in [-1, 1]$ | Bounded normalization avoids extreme numerical division spikes when destination packets/bytes are zero or near-zero. |
| **Packet Loss & Congestion** | Not modeled | `loss_ratio_src` ($\frac{\text{sloss}}{\text{Spkts} + 10^{-4}}$), `loss_ratio_dst` ($\frac{\text{dloss}}{\text{Dpkts} + 10^{-4}}$) | Detects aggressive packet-flood buffer bloat and transmission drops characteristic of volumetric attacks. |
| **TCP Protocol & Handshake** | Not modeled | `rtt_synack_ratio`, `rtt_ackdat_ratio`, `is_syn_only` | Dissects 3-way handshake round trips; `is_syn_only` flags half-open connections and SYN scans with zero ACK return. |
| **Connection Density Tracking** | 1 interaction: `ct_srv_dst_x_ct_dst_ltm` | 2 ratios: `ct_srv_ratio` ($\frac{\text{ct\_srv\_src}}{\text{ct\_srv\_dst} + 10^{-4}}$), `ct_dst_src_ratio` ($\frac{\text{ct\_dst\_src\_ltm}}{\text{ct\_dst\_ltm} + 10^{-4}}$) | Identifies port scans, lateral reconnaissance, and host-targeted sweeps over short and long temporal windows. |
| **Categorical Encodings** | 14 binary one-hot indicators for `state` (`state_is_con`, etc.) | **Top-30 Frequency bucketing on `proto`** + Native integer mapping on `state`/`service` with **Out-of-Vocabulary (`OOV`) buckets** | One-hot expansion dilutes sample count per node in tree splits. Integer encoding with frequency capping handles rare and unseen test protocols without tree fragmentation. |

---

### B. Modeling Strategy Comparison

```
                         PEER PIPELINE
┌─────────────────────────────────────────────────────────────┐
│ 38 Raw Features ──> 58 Features (One-Hot States + Div Ratios)│
│                                                             │
│       Single LightGBM Model (T* = 0.0298)                   │
│                                                             │
│       Result: Cost = 3,447 | FN = 7 | FP = 3,307            │
└─────────────────────────────────────────────────────────────┘

                      TEAM ODYSSEY PIPELINE
┌─────────────────────────────────────────────────────────────┐
│ 38 Raw Features (Drop stcpb/dtcpb) ──> 61 Domain Features   │
│                                                             │
│ ┌───────────────────────┐ ┌───────────────────────────────┐ │
│ │ Clean Traffic Branch  │ │ Shift-Augmented Branch        │ │
│ │ HistGB + LightGBM     │ │ HistGB + LightGBM on 35%      │ │
│ │ Weight: 0.35          │ │ Latency & Bandwidth Drift     │ │
│ └──────────┬────────────┘ └───────────────┬───────────────┘ │
│            │                              │                 │
│            │   ┌──────────────────────┐   │                 │
│            └──>│ Invariant Ratios Br. │<──┘                 │
│                │ 47 Scale-Free Ratios │                     │
│                │ Weight: 0.30         │                     │
│                └──────────┬───────────┘                     │
│                           │                                 │
│                           ▼                                 │
│      Multi-View Tri-Ensemble Champion (T* = 0.0604)         │
│                                                             │
│        Result: Cost = 3,431 | FN = 6 | FP = 3,311           │
└─────────────────────────────────────────────────────────────┘
```

#### 1. Peer Strategy: Single-Model LightGBM
* Employs LightGBM on hand-crafted ratios, interactions, and one-hot categorical state flags.
* Relies on threshold down-shifting ($T^* \approx 0.02 - 0.03$) to minimize false negatives under asymmetric risk.
* **Limitation**: Vulnerable to single-model inductive bias; retains testbed leakage features (`stcpb`, `dtcpb`); evaluated solely on clean in-distribution cross-validation.

#### 2. Team Odyssey Strategy: Multi-View Tri-Ensemble
* **Clean Baseline Branch (35% weight)**: Soft probability blend combining `HistGradientBoostingClassifier` and `LGBMClassifier` trained on 1.52M clean records across all 61 features. Combines scikit-learn's histogram depth splitting with LightGBM's leaf-wise tree growth for peak baseline precision.
* **Shift-Augmented Branch (35% weight)**: Trained on 1.52M records augmented with 35% realistic perturbations (jitter surges, latency doubling, bandwidth throttling) to learn invariant representations under distribution drift.
* **Scale-Invariant Ratio Branch (30% weight)**: Evaluates exclusively on the 47 scale-invariant ratio and directional features, stripping raw volume counters susceptible to network scale drift.
* **Multi-View Tri-Ensemble (Champion Architecture)**: Soft probability fusion combining all three complementary views ($0.35 \times \text{Clean} + 0.35 \times \text{Shift} + 0.30 \times \text{Ratio}$) providing maximum decision boundary smoothing, yielding the optimal **3,431 cost score**.

---

## 🛡️ 3. Robustness, Zero-Day Generalization & Stress Testing

While the peer report validated performance using 5-Fold Stratified Cross-Validation on the training distribution, Team Odyssey implemented two specialized security validation regimens to guarantee operational survivability:

### A. Leave-One-Attack-Family-Out (LOFO) Zero-Day Validation
To simulate the arrival of unseen zero-day attacks (e.g., DoS, Reconnaissance, Backdoor, Shellcode), models were trained after **completely holding out** specific attack families from the training data:

| Held-Out Attack Family | Validation Sample Count | Multi-View Ensemble Zero-Day Recall | Baseline Model Zero-Day Recall | Zero-Day Generalization Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Exploits** | 517 | **98.26%** | 98.07% | $+0.19\%$ (High baseline retention) |
| **Generic** | 1,609 | **95.34%** | 29.77% | **$+65.57\%$ massive generalization leap** |
| **Fuzzers** | 445 | **40.45%** | 38.20% | $+2.25\%$ (Transport layer limit) |
| **Mean Across Families** | 2,571 | **78.02%** | 55.35% | **$+22.67\%$ overall zero-day boost** |

> **Finding**: In the Multi-View Ensemble, unseen Generic zero-day recall surged from **29.77% to 95.34%**, and unseen Exploits maintained **98.26%**. Fuzzers (~40.5%) reflect micro-burst probes (e.g., 2 UDP packets in 3 microseconds) that exhibit zero volumetric divergence from benign dropped packets without Deep Packet Inspection (DPI).

### B. Environmental Shift & Traffic Drift Stress Testing
Simulated real-world enterprise infrastructure perturbations (link congestion, routing delays, throttling) across all 381,007 validation records:

| Network Stress Scenario | Stress Parameters Applied | Baseline F1 | Perturbed F1 | Attack Detection Recall |
| :--- | :--- | :---: | :---: | :---: |
| **Baseline (Clean Validation)** | Unmodified test flows | $0.9637$ | $0.9637$ | **$99.99\%$** (6 missed) |
| **Latency Spike** | $2\times$ RTT, $2\times$ duration, $3\times$ jitter | $0.9637$ | $0.9582$ | **$99.81\%$** |
| **Connection Density Flood** | $2\times$ connection counts (`ct_*`) | $0.9637$ | $0.9610$ | **$99.69\%$** |
| **Bandwidth Throttling** | $0.3\times$ bytes, $0.3\times$ load | $0.9637$ | $0.9495$ | **$99.54\%$** |
| **Combined Network Surge** | Latency + Density + Volume shift | $0.9637$ | $0.9451$ | **$99.38\%$** |

---

## 🔍 4. In-Depth Side-by-Side Comparison Matrix

| Evaluation Dimension | Peer Work (LightGBM Pipeline) | Team Odyssey Work (Multi-View Tri-Ensemble) | Comparative Verdict & Advantage |
| :--- | :--- | :--- | :--- |
| **Validation Cost ($w=20$)** | $3,447$ | **$3,431$** | **Team Odyssey Wins ($-16$ lower cost)**: Achieves the best recorded cost score on the full 381k benchmark. |
| **False Negatives (Missed Attacks)** | $7$ | **$6$** | **Team Odyssey Wins**: Detects $44,034$ out of $44,040$ attacks ($99.986\%$ detection rate). |
| **False Positives (False Alarms)** | **$3,307$** | $3,311$ | **Parity ($\Delta = 4$)**: Both limit false alarm overhead to $<1.0\%$ across $336,967$ normal flows. |
| **Cost Reduction vs Baseline** | $-9.84\%$ | **$-10.25\%$** | **Team Odyssey Wins**: Delivers double-digit cost reduction over the official competition baseline. |
| **Decision Threshold Stability** | $T^* = 0.0298$ | **$T^* = 0.0604$** | **Team Odyssey Wins**: Operating at $\approx 0.06$ is roughly $2\times$ higher than $\approx 0.03$, offering greater margin against probability jitter. |
| **Feature Leakage Rigor** | Kept synthetic sequence numbers `stcpb`/`dtcpb` | **Dropped `stcpb` & `dtcpb`** | **Team Odyssey Wins**: Eliminates capture seed memorization, ensuring legitimate production validity. |
| **Ratio Mathematical Formulation** | Unbounded: $\frac{A}{B + \epsilon}$ | **Bounded: $\frac{A - B}{A + B + 1} \in [-1, 1]$** | **Team Odyssey Wins**: Bounded normalization prevents numerical divergence and extreme outliers. |
| **Protocol & Handshake Modeling** | None | **3 TCP Handshake Diagnostics** (`rtt_synack_ratio`, `ackdat`, `is_syn_only`) | **Team Odyssey Wins**: Directly flags stealth scans and half-open SYN floods that payload ratios miss. |
| **Model Diversity & Ensembling** | Single LightGBM model | **Multi-View Tri-Ensemble** (HistGB + LightGBM + Drift-Augmented + Invariant Ratios) | **Team Odyssey Wins**: Multi-algorithm ensembling reduces variance and smooths probability calibration. |
| **Validation Methodology** | 5-Fold Stratified CV only | **LOFO Zero-Day Validation + 4 Network Shift Stress Tests** | **Team Odyssey Wins**: Quantifies true zero-day detection and resilience to environmental drift. |
| **Dynamic Re-Tuning Tooling** | Static threshold | **Automated Hour 3 Re-Tuner** (`hour3_tune_w.py`) | **Team Odyssey Wins**: Capable of re-optimizing $T^*$ and updating deployment metadata in $<5$ seconds upon weight announcement. |

---

## ⚡ 5. Operational Adaptability: Hour 3 Cost Weight Injection

In live hackathons and SOC deployments, the cost penalty weight $w$ can change dynamically (e.g., $w=35$, $w=50$). 

Team Odyssey implemented an instant re-tuning tool ([`starter/hour3_tune_w.py`](file:///k:/Dev/Odyssey/starter/hour3_tune_w.py)):

* **Execution Time**: Re-optimizes decision boundaries across all 381,007 records in **$<3$ seconds**.
* **Automatic Package Synchronization**: Updates `best_threshold`, `cost_weight`, and operational metrics in both [`team_model/`](file:///k:/Dev/Odyssey/team_model/) and [`odyssey_model/`](file:///k:/Dev/Odyssey/odyssey_model/).
* **Cost Trajectory Under Dynamic $w$**:
  * At $w = 20$: $T^* = 0.0604 \implies \text{Cost} = 3,431$ (FN = 6, FP = 3,311)
  * At $w = 40$ (Official Announced Weight): $T^* = 0.0600 \implies \text{Cost} = 3,552$ (FN = 6, FP = 3,312) | **92.0% cost reduction** vs default $T=0.50$ (44,226)
  * At $w = 50$: $T^* = 0.0210 \implies \text{Cost} = 3,610$ (FN = 3, FP = 3,460)

---

## 🎯 6. Conclusion & Deployment Verification

1. **Quantifiable Superiority**: Team Odyssey's pipeline achieves the lowest operational cost (**$3,431$** vs peer's **$3,447$**) and lowest missed intrusion count (**$6$** vs peer's **$7$**).
2. **Defensible Engineering**: Rather than taking shortcuts with testbed sequence numbers (`stcpb`, `dtcpb`), Team Odyssey achieved its leading score with genuine domain-engineered flow dynamics and TCP handshake metrics.
3. **Verified Reproducibility**: The model package in [`team_model/`](file:///k:/Dev/Odyssey/team_model/) is fully self-contained, tested with [`starter/evaluate.py`](file:///k:/Dev/Odyssey/starter/evaluate.py), and executes independently without external notebook dependencies.
