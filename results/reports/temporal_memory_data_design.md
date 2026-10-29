# Real Temporal and Memory Data Collection Design Specification

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Document:** Research Specification for Future Temporal & Memory Telemetry Pipeline  
**Date:** September 2026  
**Status:** Design and Schema Phase Completed  

---

## 1. Research Gap
The baseline machine learning models (Random Forest, XGBoost) and reinforcement learning agent (PPO) were developed using a static/dynamic feature-summary dataset (`cleaned_dataset.csv`). While achieving 100% macro F1 on complete feature vectors, this dataset lacks physical execution timestamps, event sequence ordering, and dynamic memory page telemetry. Consequently, true physical early-stage detection (latency in milliseconds, detection prior to initial encryption) cannot currently be measured.

---

## 2. Current Dataset
- **Nature:** Post-mortem dynamic-analysis aggregate feature counts and entropy summaries.
- **Rows:** 1,750 samples across 3 ransomware families (G, E, L).
- **Evaluation:** Single-shot static evaluation or synthetic progressive feature revelation (5 steps: 20% to 100%).

---

## 3. Proposed Future Dataset
A high-resolution, continuous streaming event dataset capturing discrete sliding observation windows ($W_k$) from isolated hypervisor execution runs, recording event rates, memory deltas, and microsecond timestamps.

---

## 4. Temporal Schema
The proposed dataset schema comprises 27 structured fields spanning session identifiers, process tree metadata, microsecond timestamps, and behavioral rate counters (API calls/ms, file creations/s, registry modifications/s).

---

## 5. Memory Schema
Volatile memory telemetry captures working set byte size, private committed bytes, unbacked executable memory regions, heap allocation entropy, and handle/thread counts.

---

## 6. Feature Engineering
Raw circular event streams are transformed into temporal window observations via first-order deltas ($\Delta \text{entropy}$, $\Delta \text{memory}$, $\Delta \text{handles}$) and moving average rolling statistics.

---

## 7. Labeling
Dual-level labeling architecture: sample-level ground truth (benign vs. ransomware family) and window-level temporal phase labels (pre-encryption suspicious, active encryption, post-damage).

---

## 8. Ground Truth
Hardware-synchronized milestone logging tracking process creation, code injection, volume shadow tampering, and pre-positioned canary file access ($t_{\text{onset}}$).

---

## 9. Leakage Prevention
Strict prevention protocols addressing 7 temporal leakage vectors, including sample-disjoint partitioning, causal rolling window calculation, and golden hypervisor snapshot reversion.

---

## 10. PPO Integration
Smooth architectural migration from `RansomwareDetectionEnv` to `StreamingRansomwareDetectionEnv`, preserving the Discrete(3) action space (`CONTINUE`, `SUSPICIOUS`, `ALERT`) and Gymnasium API contracts.

---

## 11. Evaluation Metrics
Enables future physical metrics: Time-to-Detection (TTD in ms), Early Detection Rate (EDR), and Detection Before First Encryption (DBE).

---

## 12. Safety Boundary
All empirical data collection is strictly restricted to air-gapped, isolated hypervisor testbeds with automated snapshot rollbacks. Zero malware execution is performed in this design phase.

---

## 13. Implementation Roadmap
A 10-stage sequential development roadmap taking the project from schema specification to independent multi-sandbox zero-day validation.

---

## 14. Limitations
This specification designs future collection infrastructure; it does not collect real malware data or establish real-time latency on the current dataset. PPO results remain a synthetic proof-of-concept until real temporal telemetry is collected.