# Early-Stage Feature-Restricted Baseline Analysis

**Project Title:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Experiment:** Early-Stage Feature-Restricted Baseline Proxy  
**Date:** September 2026  
**Document:** Research Analysis & Experimental Findings Report  

---

## 1. Executive Summary & Research Framing

> [!IMPORTANT]
> **Fundamental Research Framing:**  
> **This experiment is an early-stage feature-restricted proxy because the available dataset does not contain explicit temporal execution windows.**  
> It does not measure elapsed physical execution time (e.g. seconds or milliseconds), nor does it claim genuine temporal early detection. Instead, it systematically tests whether malware family discrimination remains feasible when excluding post-mortem filesystem encryption artifacts and restricting the feature space to telemetry that is plausibly observable during early process setup.

---

## 2. Experimental Setup & Feature Sets

All experiments were executed on the strictly isolated 80% training ($n=1400$) and 20% testing ($n=350$) partitions loaded from `data/processed/split_indices.json`, ensuring zero data leakage.

Three feature spaces were compared across Random Forest and XGBoost:

1. **Dataset A (Full Features — 46 features):** Complete post-cleaning dynamic analysis feature space, including late-stage mass filesystem actions (`file_created`, `file_read`, `directory_enumerated`) and post-run sandbox signatures (`log`, `action`, `positives`, `errors`).
2. **Dataset B (Early-Stage Candidate Proxy — 24 features):** Features representing initial process creation, static/memory header attributes, initial API invocations, DLL load events, and configuration registry queries.
3. **Dataset C (Early-Stage Non-Redundant — 16 features):** Derived from Dataset B by collapsing exact mathematical duplicates ($r = 1.0000$ across all rows) identified in the leakage analysis:
   - `proc_pid` retained (removed duplicate `file`)
   - `type` retained (removed duplicate `name`, `path`)
   - `program` retained (removed duplicate `info`)
   - `arguments` retained (removed duplicate `api`)
   - `proc` retained (removed duplicate `beh_command_line`, `process_path`)
   - `tree_command_line` retained (removed duplicate `tree_process_name`)

---

## 3. Empirical Results Comparison

The performance across all 6 experiments is summarized below (from `results/reports/early_stage_comparison.csv`):

| Model | Feature Set | Features | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Minority ($L$) Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Full | 46 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** (33/33) |
| **Random Forest** | Early-stage proxy | 24 | **0.9971** | **0.9978** | **0.9979** | **0.9971** | **1.0000** (33/33) |
| **Random Forest** | Early-stage non-redundant | 16 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** (33/33) |
| **XGBoost** | Full | 46 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** (33/33) |
| **XGBoost** | Early-stage proxy | 24 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** (33/33) |
| **XGBoost** | Early-stage non-redundant | 16 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** (33/33) |

---

## 4. Key Scientific Insights

### A. Minimal Performance Degradation Under Feature Restriction
- Eliminating late-stage filesystem operations (`file_created`, `file_read`, `directory_enumerated`) and post-run sandbox verdicts caused virtually no loss in discriminative power (Macro F1 remained between $0.9979$ and $1.0000$).
- This demonstrates that early-stage telemetry (process hierarchy, argument patterns, library loading, registry queries, and memory entropy) retains strong family-specific signatures.

### B. Impact of Removing Redundant Features
- In Random Forest, moving from the 24-feature candidate set (Dataset B) to the 16-feature non-redundant set (Dataset C) eliminated dilution of tree split importances across duplicate columns, restoring test accuracy from $0.9971$ back to $1.0000$.
- In XGBoost, performance was completely robust ($1.0000$) across all configurations.

### C. Minority Class ($L$) Resilience
- Class $L$ represents only 9.31% of the dataset ($n=33$ in test set).
- Across all 6 experimental runs, minority class recall remained **100% (33/33)**, with zero false negatives or false alarms for Family $L$.

### D. Top Predictive Early-Stage Features
When restricting to Dataset C (16 non-redundant features), feature importances concentrated heavily in:
1. **Dynamic Memory & Library Invocations:** `dll` (17.5% RF), `dll_loaded` (9.2% RF, 4.6% XGB), `entropy` (9.0% RF, 4.9% XGB).
2. **Process Spawning & Execution Arguments:** `tree_command_line` (15.3% XGB, 4.1% RF), `arguments` (13.1% RF, 12.3% XGB), `command_line` (4.7% RF, 5.5% XGB).
3. **API & System Query Activity:** `apistats` (14.6% RF, 7.0% XGB), `regkey_written` (12.0% XGB, 4.0% RF), `regkey_opened` (5.4% RF, 5.4% XGB), `type` (12.3% XGB, 3.3% RF).

---

## 5. Critical Research Limitations

1. **Absence of Ground-Truth Temporal Sequences:** The features, while restricted to early-stage categories, still represent summary values collected over the sandbox execution run.
2. **Multi-Family Discrimination vs. Benign Anomaly Detection:** The task evaluates separating known ransomware strains (`G`, `E`, `L`), not stopping unknown malware from an uninfected baseline.
3. **Necessity of Reinforcement Learning:** To transform these static findings into genuine early-stage intervention, a sequential agent (PPO) must interact with time-sliced state transitions in an active monitoring loop.

