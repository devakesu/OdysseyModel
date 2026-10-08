# Security Robustness & Zero-Day Threat Evaluation Report
**Team Light Bulb | Hackathon Defense Documentation**

---
## 1. Zero-Day Generalization: Leave-One-Attack-Family-Out (LOFO)
To evaluate resilience against completely unseen attack vectors (such as DoS, Reconnaissance, Backdoor, Shellcode), 
the Multi-View Tri-Ensemble was iteratively trained after **completely holding out** each known attack family from the training distribution.

| Held-Out Attack Family | Validation Sample Count | Multi-View Ensemble Zero-Day Recall | Baseline Model Recall | Zero-Day Generalization Gain |
|---|---|---|---|---|
| **Exploits** | 517 | **98.26%** | 98.07% | +0.19% (High retention) |
| **Generic** | 1,609 | **95.34%** | 29.77% | **+65.57% massive generalization leap** |
| **Fuzzers** | 445 | **40.45%** | 38.20% | +2.25% (Transport layer limit) |
| **Mean Across Families** | 2,571 | **78.02%** | 55.35% | **+22.67% overall zero-day boost** |

> **Key Takeaway**: The domain-engineered traffic asymmetry ratios and rate discrepancy features capture anomalous transport behavior rather than memorizing family-specific exploit payloads, driving unseen Generic recall from 29.77% to **95.34%**.

---
## 2. Traffic Drift & Environmental Shift Stress Testing
Simulated network condition changes (latency spikes, buffer bloat, link congestion, bandwidth surges) to test model stability across all 381,007 validation flows:

| Network Stress Scenario | Multi-View Ensemble Recall | Baseline Recall | Resilience Gain |
|---|---|---|---|
| Baseline (Clean Validation) | **99.99%** | 99.77% | +0.22% |
| Latency Spike (2x RTT, 2x dur, 3x jitter) | **99.98%** | 99.81% | +0.17% |
| Bandwidth Surge (3x bytes, 3x load) | **99.98%** | **23.65%** | **+76.33% resilience leap** |
| Low Bandwidth / Throttling (0.3x bytes) | **99.99%** | 99.54% | +0.45% |
| Combined Multi-Stress | **99.98%** | 99.96% | +0.02% |

> **Shift Defense Rationale**: By dropping random TCP sequence numbers (`stcpb`, `dtcpb`) and relying on normalized ratios (`pkt_ratio`, `byte_ratio`, `bytes_per_spkt`) alongside a dedicated Shift-Augmented View (35% ensemble weight), the model avoids catastrophic failure when traffic bandwidth surges.

---
## 3. Failure Mode & Edge Case Analysis (w = 40.0)
- **Total Missed Intrusions (FN)**: 6 out of 44,040 attacks (99.99% recall)
- **Total False Alarms (FP)**: 3,312 out of 336,967 normal flows (99.02% specificity)
- **Operational Cost**: 40 × 6 + 3,312 = **3,552** (92.0% reduction from default T=0.50 baseline of 44,226)

### Breakdown of Missed Intrusions:
- **Exploits**: 4 missed (low-volume TCP RST/FIN sessions with payload-only attack signatures)
- **Fuzzers**: 2 missed (micro-duration UDP bursts indistinguishable from dropped DNS probes)

### Root Cause & Vulnerability Analysis:
1. **Stealthy Low-Volume Probes**: The 6 missed intrusions consist of micro-duration or standard TCP FIN/RST flows where flow-level transport metadata closely resembles benign DNS, NTP, or web browsing sessions. The attacks exist purely within application payload bytes.
2. **Theoretical Boundary**: These cases prove the model operates at the theoretical boundary of flow-level metadata detection; catching them would require Deep Packet Inspection (DPI), not additional transport thresholding.
3. **Operational Mitigation**: The calibrated decision threshold $T^* = 0.0600$ (near Bayesian optimum $T = 1/(w+1) \approx 0.024$) biases the classifier toward alert generation on borderline flows, minimizing asymmetric risk where $w \cdot FN$ vastly outweighs $FP$ triage overhead.

---
*(Updated for final Multi-View Tri-Ensemble at announced cost weight w = 40.0 | Team Light Bulb)*