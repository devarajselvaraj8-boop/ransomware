# Research Validation Summary

**Project Title:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Role:** Lead ML Engineer  
**Date:** August 2026  
**Status:** Baseline Pipeline Validation & Scientific Integrity Report  

---

## 1. Dataset Overview and Dimensions

The current research dataset is derived from sandboxed dynamic execution traces (`Annex A - Dataset with 50 chosen features (1).csv`) and cleaned into a standardized feature matrix (`data/processed/cleaned_dataset.csv` and `data/processed/ml_ready_dataset.csv`).

- **Original Raw Dimensions:** 2,000 observations $\times$ 51 columns (50 features + 1 target: `family`).
- **Post-Cleaning Dimensions:** 1,750 observations $\times$ 47 columns (46 feature attributes + 1 target).
- **Missing / Infinite Values:** Exactly 0 missing values and 0 infinite values across all rows and numeric columns.

---

## 2. Target Variable and Class Distribution

The target variable is `family`, representing three distinct ransomware families:
- **Class `G`:** 820 samples (46.86% of dataset)
- **Class `E`:** 767 samples (43.83% of dataset)
- **Class `L`:** 163 samples (9.31% of dataset) — *Minority Class*

The dataset exhibits moderate class imbalance, particularly for Family `L`, requiring evaluation metrics that strictly account for class representation (Macro F1, Balanced Accuracy, and Per-Class Recall).

---

## 3. Duplicate Handling

- In the original dataset of 2,000 rows, exactly **250 exact duplicate rows** were identified and purged during initial data sanitization.
- The active cleaned dataset (`cleaned_dataset.csv`) contains **0 duplicate rows**, verified by automated hash-based row deduplication.
- Duplicate removal ensures that the exact same behavioral vector does not simultaneously appear in both training and test partitions, preventing artificial memorization.

---

## 4. Constant Feature Removal

Four zero-variance (constant) features were identified in the raw 50-feature matrix and removed:
1. `ext_urls` (variance = 0)
2. `sign_stacktrace` (variance = 0)
3. `imported_dll_count` (variance = 0)
4. `mitm` (variance = 0)

Zero-variance features provide zero mutual information and needlessly increase dimensionality; their removal reduces feature space to 46 clean attributes.

---

## 5. Train / Test Methodology & Data Isolation

To guarantee scientific reproducibility and prevent data snooping:
- **Partition Ratio:** 80% Training ($n = 1400$), 20% Testing ($n = 350$).
- **Random Seed:** `random_state = 42`.
- **Stratification:** Stratified sampling on `family` ensures equal class proportions across partitions:
  - **Training:** G = 656 (46.86%), E = 614 (43.86%), L = 130 (9.29%).
  - **Testing:** G = 164 (46.86%), E = 153 (43.71%), L = 33 (9.43%).
- **Isolation Invariant:** Preprocessing criteria, encoders, and scaling (where applicable) are strictly fit on training data alone. The test partition remains completely untouched until model evaluation. Split indices are saved to `data/processed/split_indices.json`.

---

## 6. Data Leakage & Feature Semantics Investigation

A critical research finding in this pipeline is the phenomenon of **100% test accuracy** across baseline models. In empirical cybersecurity research, such perfect scores must not be accepted uncritically.

Our deep inspection documented in `results/reports/leakage_analysis.txt` identified the following root causes:

### A. Perfectly Identical Column Redundancies
15 pairs of feature columns exhibit an exact 1:1 mathematical identity ($r = 1.0000$ across all 1,750 samples):
- `pe_res_name` $\equiv$ `filetype`
- `arguments` $\equiv$ `api` $\equiv$ `category` (all three identical)
- `pe_sec_name` $\equiv$ `entropy`
- `proc_pid` $\equiv$ `file`
- `proc` $\equiv$ `beh_command_line` $\equiv$ `process_path` (all three identical)
- `tree_command_line` $\equiv$ `tree_process_name`
- `program` $\equiv$ `info`
- `description` $\equiv$ `sign_name`

### B. Post-Execution vs. Dynamic Behavioral Distinction
Features such as `file_created`, `file_read`, `regkey_opened`, `apistats`, and `log` represent **cumulative end-of-run summaries** captured by the Cuckoo/CAPE sandbox after the sample has completed its full lifecycle:
- High values of `file_created` or `file_read` occur because the ransomware has *already executed* its encryption phase.
- Consequently, these features represent a post-mortem signature rather than early-stage behavioral signals.

### C. Feature Retention Decision
All 46 features were preserved in the baseline models to establish a consistent, benchmarkable comparison against earlier literature and notebook experiments. However, no feature should be falsely interpreted as causal without temporal validation.

---

## 7. Baseline Random Forest Performance

Trained under configuration: `RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)`.

- **Accuracy:** 1.0000
- **Balanced Accuracy:** 1.0000
- **Precision (Macro / Weighted):** 1.0000 / 1.0000
- **Recall (Macro / Weighted):** 1.0000 / 1.0000
- **F1-Score (Macro / Weighted):** 1.0000 / 1.0000
- **Per-Class Metrics:**
  - Class `E`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (n=153)
  - Class `G`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (n=164)
  - Class `L`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (n=33)
- **Misclassifications:** 0 (False Positives = 0, False Negatives = 0)

---

## 8. Baseline XGBoost Performance

Trained under configuration: `XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric="mlogloss")`.

- **Accuracy:** 1.0000
- **Balanced Accuracy:** 1.0000
- **Precision (Macro / Weighted):** 1.0000 / 1.0000
- **Recall (Macro / Weighted):** 1.0000 / 1.0000
- **F1-Score (Macro / Weighted):** 1.0000 / 1.0000
- **Per-Class Metrics:**
  - Class `E`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (n=153)
  - Class `G`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (n=164)
  - Class `L`: Precision = 1.0000, Recall = 1.0000, F1 = 1.0000 (n=33)

---

## 9. Top Predictive Features

Based on Gini importance from Random Forest and split gain from XGBoost:
1. `filetype` / `pe_res_name` (~0.0988)
2. `apistats` (~0.0760)
3. `dll` (~0.0725)
4. `file_created` (~0.0540)
5. `file_read` (~0.0518)
6. `arguments` / `category` / `api` (~0.0468)
7. `dll_loaded` (~0.0398)
8. `pe_sec_name` / `entropy` (~0.0298)
9. `directory_enumerated` (~0.0246)
10. `regkey_read` (~0.0219)

**Observation:** While static metadata (`filetype`, `pe_res_name`) and post-execution counts (`apistats`, `file_created`) dominate importance, they do not prove that detection is occurring before destructive payloads detonate.

---

## 10. Research Limitations & Crucial Scientific Warning

> [!WARNING]
> **Perfect performance requires further validation and should not automatically be interpreted as real-world ransomware detection capability.**

### Detailed Scientific Caveats:
1. **Absence of Benign Class:** The dataset contains only ransomware families (`G`, `E`, `L`), classifying *which* family a known sample belongs to (family attribution) rather than distinguishing *ransomware from benign processes* (detection).
2. **Static vs. Dynamic Conflation:** Features like `filetype` and `entropy` are PE header attributes known before execution, while `file_created` is an end-of-run aggregate.
3. **Synthetic / Laboratory Artifacts:** Sandbox environments induce specific execution behaviors that may not reflect evasive malware operating on real bare-metal host memory.
4. **Transition to Reinforcement Learning:** True early detection requires making sequential decisions during runtime using intermediate memory telemetry rather than evaluating finished logs.

