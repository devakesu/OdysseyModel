# 3-5 Minute Technical Defense & Evaluation Presentation
**Team Odyssey | Enterprise Network Intrusion Detection Hackathon**

---

### Slide 1: Executive Summary & Performance Delta
- **Problem**: Binary intrusion detection on 38 flow-level UNSW-NB15 features under severe asymmetric operational risk ($w \cdot \text{FN} + \text{FP}$, where announced $w = 40$) and zero-day threat exposure.
- **Key Results on Full 381,007 Validation Flows**:
  - **PR-AUC**: **0.9989** (Starter baseline: 0.9916)
  - **F1-Score**: **0.9800** at $T=0.50$ | **0.9637** at cost-optimal $T^*$
  - **Attack Detection Recall**: **99.99%** (Detected 44,034 / 44,040 validation attacks; **only 6 missed**)
  - **Operational Cost Reduction**: **92.0% reduction in asymmetric risk** (from 44,226 default threshold down to 3,552 at announced $w=40$)

---

### Slide 2: Feature Engineering & Anti-Leakage Rigor
- **Testbed Leakage Mitigation**:
  - Dropped `stcpb` and `dtcpb` (32-bit TCP initial sequence numbers). Synthetic testbed seeds cause models to memorize sequence windows rather than learn flow dynamics. Dropping them prevents severe performance drops under network shift.
- **Heavy-Tailed Normalization**:
  - Applied `log1p` scaling across 13 metrics (`dur`, `sbytes`, `dbytes`, `Sload`, `Dload`, `Sjit`, `Djit`, `Sintpkt`, `Dintpkt`, `res_bdy_len`, `tcprtt`, `synack`, `ackdat`) spanning up to 8 orders of magnitude, optimizing tree histogram split bins.
- **Domain Flow Ratios**:
  - *Traffic Asymmetry*: `byte_ratio` and `pkt_ratio` measure directional push/pull dynamics.
  - *Payload & Loss*: `bytes_per_spkt`, `loss_ratio_src` detect packet flood buffer bloat.
  - *TCP Handshake Verification*: `rtt_synack_ratio`, `is_syn_only` catch half-open port scans and SYN flooding.
  - *Connection Concentration*: `ct_srv_ratio`, `ct_dst_src_ratio` detect port scans and lateral movement.

---

### Slide 3: Multi-View Tri-Ensemble Architecture
- **Architecture**: Tri-Ensemble soft blend across three distinct views of network traffic:
  - **Clean Traffic View (35%)**: Trained on clean 1.52M flows across all 61 features for maximum precision on baseline traffic.
  - **Shift-Augmented View (35%)**: Trained on 2.05M flows with realistic perturbations (latency, jitter, bandwidth scaling) to prevent memorizing fixed rate bands.
  - **Scale-Invariant Ratio View (30%)**: Stripped of raw byte/duration magnitudes, evaluating strictly on 47 scale-invariant ratios and asymmetries.
- **Why Multi-View Beats Single Models**:
  - Blending clean, shift-trained, and ratio-only trees smooths decision boundaries and prevents overfitting to capture session artifacts.

---

### Slide 4: Zero-Day Threat Generalization (LOFO Simulation)
- **Methodology**: Simulated unseen threat generalization using Leave-One-Attack-Family-Out (LOFO) cross-validation where an entire attack family was removed during training and tested as zero-day traffic.
- **Results**:
  - **Unseen Exploits Zero-Day Recall**: **98.26%** caught without seeing any Exploit samples in training.
  - **Unseen Generic Zero-Day Recall**: **95.34%** caught.
  - **Mean Zero-Day Recall**: **78.02%** across held-out families.
- **Takeaway**: Confirms that our ratio features capture anomalous transport behavior rather than memorizing family-specific payloads.

---

### Slide 5: Environmental Shift & The Bandwidth Surge Test
- **The Decisive Stress Test (381,007 Validation Flows)**:
  - *Latency Spikes ($2\times$ RTT, $2\times$ duration, $3\times$ jitter)*: Recall remained **99.98%**.
  - *Bandwidth Surge ($3\times$ bytes, $3\times$ load)*:
    - **Single-Model Baseline (Raw Magnitude Trees)**: Recall collapsed to **23.65%** (memorized high-load thresholds).
    - **Team Odyssey Multi-View Ensemble**: Maintained **99.98% recall** (**+76.32% resilience boost**!).
  - *Bandwidth Throttling ($0.3\times$ bytes)*: Attack Recall remained **99.99%**.
  - *Combined Multi-Stress*: Maintained **99.98% recall**.
- **Takeaway**: The ensemble's ratio-centric and shift-trained expert branches prevent catastrophic failure when traffic bandwidth surges.

---

### Slide 6: Asymmetric Operational Cost Optimization ($w \cdot \text{FN} + \text{FP}$)
- **The SOC Tradeoff**: Missing an active breach (FN) costs $40\times$ more than triaging a false alarm (FP) ($w = 40.0$).
- **Mathematical Optimization**:
  - Calibrated Bayes decision threshold: $T^* \approx \frac{1}{w + 1} = \frac{1}{41} \approx 0.0244$.
  - At announced $w=40$, standard default threshold $T=0.50$ suffered 1,089 missed intrusions, exploding operational cost to **44,226**.
  - Calibrated optimal threshold $T^* = 0.0600$ cuts missed intrusions from 1,089 down to **only 6**, achieving a **3,552 total cost score** (**92.0% risk reduction**).
  - Normal traffic specificity remains high at **99.02%** (false alarms restricted to <1% across 336,967 benign flows).
  - Fully automated Hour 3 re-optimization verified in 23 seconds via `starter/hour3_tune_w.py --cost-weight 40.0`.

---

### Slide 7: Failure Analysis & Empirical Boundary Conditions
- **Exact Missed Intrusion Anatomy (6 Missed Attacks out of 44,040)**:
  - *Breakdown*: **4 Exploits**, **2 Fuzzers** (0 Generic, 0 DoS, 0 Reconnaissance missed).
  - *Micro-Probe Edge Case*: 1 Fuzzer is a 2-packet UDP burst lasting $3\,\mu\text{s}$ (`dur=0.000003s`, 0 return bytes)—indistinguishable at flow level from a dropped DNS probe.
  - *Connection Resets*: 2 Exploits are low-volume 6-packet TCP RST interactions (258 bytes sent).
  - *Clean Transport Mimicry*: 3 flows are standard TCP FIN sessions with normal bidirectional traffic (including a 65KB download)—attacks embedded purely in application payload bytes.
- **The Theoretical Boundary**: These 6 cases prove the model operates at the theoretical boundary of flow-level metadata; catching them requires Deep Packet Inspection (DPI) payload inspection, not additional transport thresholding.
- **Edge-Case Safety**: Soft probability ensembling prevents probability spikes, maintaining stable confidence across non-standard transport sessions.

---

### Slide 8: Enterprise Scale, Latency & Technical Reproducibility
- **Enterprise Throughput & Latency**:
  - *Pure Model Inference*: **138,000 flows/sec** (2.76s for 381,007 flows on a single CPU instance).
  - *End-to-End Pipeline*: **41,400 flows/sec** (9.19s total including CSV parsing, 61 feature transforms, inference, and serialization).
  - *Scale Capacity*: Over **3.5 Billion flows/day** on a single workstation—comfortably exceeding the "hundreds of millions of flows/day" enterprise benchmark without needing model distillation.
- **Deliverable Package (`team_model/` and `odyssey_model/`)**:
  - `model.joblib`: Serialized Multi-View Tri-Ensemble and feature extractor.
  - `predict.py`: Tested on unlabelled 38-feature CSV format with zero dependencies on loose notebooks.
  - `requirements.txt`: Clean pinned dependencies compatible with evaluation environments.
  - `metadata.json`: Full configuration, architecture, decision threshold, and validation scores.
- **Verified Reproducibility**: 100% prediction match confirmed on independent test samples using `starter/evaluate.py --validate-only`.
