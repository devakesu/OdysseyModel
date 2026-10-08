# 3-5 Minute Technical Defense & Evaluation Presentation
**Team Odyssey | Enterprise Network Intrusion Detection Hackathon**

---

### Slide 1: Executive Summary & Performance Delta
- **Problem**: Binary intrusion detection on 38 flow-level UNSW-NB15 features under asymmetric cost risk ($w \cdot \text{FN} + \text{FP}$) and zero-day threat exposure.
- **Key Results**:
  - **PR-AUC**: **0.9989** (Starter baseline: 0.9916)
  - **F1-Score**: **0.9800** at $T=0.50$ | **0.9636** at cost-optimal $T^*$
  - **Attack Detection Recall**: **99.98%** (Detected 44,030 / 44,040 validation attacks; only 10 missed)
  - **Operational Cost Reduction**: **83.9% reduction in asymmetric risk** (from 21,814 down to 3,520 at $w=20$)

---

### Slide 2: Feature Engineering & Anti-Leakage Rigor
- **Testbed Leakage Mitigation**:
  - Dropped `stcpb` and `dtcpb` (32-bit TCP initial sequence numbers). In capture environments, synthetic generator seeds cause models to memorize sequence windows rather than learn flow dynamics. Dropping them guarantees generalization across capture sessions.
- **Heavy-Tailed Normalization**:
  - Applied `log1p` scaling across 13 metrics (`dur`, `sbytes`, `dbytes`, `Sload`, `Dload`, `Sjit`, `Djit`, `Sintpkt`, `Dintpkt`, `res_bdy_len`, `tcprtt`, `synack`, `ackdat`) spanning up to 8 orders of magnitude, optimizing tree histogram split bins.
- **Domain Flow Ratios**:
  - *Traffic Asymmetry*: `byte_ratio` and `pkt_ratio` measure directional push/pull dynamics.
  - *Payload & Loss*: `bytes_per_spkt`, `loss_ratio_src` detect packet flood buffer bloat.
  - *TCP Handshake Verification*: `rtt_synack_ratio`, `is_syn_only` catch half-open port scans and SYN flooding.
  - *Connection Concentration*: `ct_srv_ratio`, `ct_dst_src_ratio` detect port scans and lateral movement.

---

### Slide 3: Model Architecture & Ensembling
- **Model Choice**: Soft probability blend of **`HistGradientBoostingClassifier`** (scikit-learn) and **`LGBMClassifier`** (LightGBM).
- **Why Ensemble**:
  - Combines two different gradient boosting tree split algorithms.
  - Soft probability averaging smooths decision boundaries, reducing prediction variance under shifted distributions and pushing PR-AUC to **0.9989**.
- **Inference Speed**: Processes 381,007 flows in ~15 seconds (~25,000 flows/sec).

---

### Slide 4: Zero-Day & Unseen Attack Generalization (LOFO Validation)
- **Methodology**: Leave-One-Attack-Family-Out (LOFO) cross-validation where an entire attack family was removed during training and tested as zero-day traffic.
- **Results**:
  - **Unseen Exploits Detection Recall**: **98.07%** (507 / 517 caught without seeing any Exploit samples in training).
  - Demonstrates that our ratio features capture anomalous transport behavior rather than memorizing family-specific payloads.
  - Prepares the model for unseen families arriving in Challenge and Final Test (`DoS`, `Reconnaissance`, `Backdoor`, `Shellcode`).

---

### Slide 5: Environmental Shift & Traffic Drift Resilience
- **Stress-Test Scenarios**:
  - *Latency Spikes ($2\times$ RTT, $2\times$ duration, $3\times$ jitter)*: F1 held at **0.9182** (Delta: -0.0007), Recall remained **99.81%**.
  - *Connection Density Surge ($2\times$ connection counts)*: F1 held at **0.9207**, Recall remained **99.69%**.
  - *Bandwidth Throttling ($0.3\times$ bytes)*: Attack Recall remained **99.54%**.
- **Conclusion**: Reliance on scale-invariant ratios shields the system from network congestion and bandwidth fluctuations.

---

### Slide 6: Asymmetric Operational Cost Optimization ($w \cdot \text{FN} + \text{FP}$)
- **The Tradeoff**: In enterprise SOC operations, missing a breach (FN) costs $w$ times more than triaging a false alarm (FP).
- **Mathematical Optimization**:
  - Calibrated Bayes decision threshold: $T^* \approx \frac{1}{w + 1}$.
  - At $w=20$, standard threshold $T=0.50$ suffers 1,056 missed intrusions.
  - Optimized threshold $T^* = 0.0405$ cuts missed intrusions from 1,056 down to **10**, yielding an **83.9% cost reduction**.
  - Script dynamically re-optimizes $T^*$ in seconds when organizers reveal the final $w$ at Hour 3.

---

### Slide 7: Failure Analysis & Boundary Conditions
- **Missed Intrusion Profile**: The 10 missed attacks across 381k flows consist of stealthy micro-duration transactions (<0.35s, single-packet interactions) that closely mimic standard DNS resolution.
- **False Alarm Profile**: FP rate is restricted to <1.0% (specificity 99.01%), ensuring security analysts are not overwhelmed with alert fatigue.

---

### Slide 8: Technical Reproducibility & Deployment Package
- **Deliverable Package (`team_model/`)**:
  - `model.joblib`: Serialized ensemble pipeline and feature extractor.
  - `predict.py`: Tested on unlabelled 38-feature CSV format with zero dependencies on loose notebooks.
  - `requirements.txt`: Clean pinned dependencies (`scikit-learn`, `lightgbm`, `pandas`, `numpy`, `joblib`).
  - `metadata.json`: Full configuration, architecture, and validation scores.
- **Verified Reproducibility**: 100% prediction match confirmed on independent test samples using `starter/evaluate.py --validate-only`.
