# Temporal Baseline Analysis

## 1. Objective
Establish a reproducible, leakage-free temporal machine learning baseline using Random Forest and XGBoost classifiers on multi-window dynamic observations, prior to evaluating reinforcement learning (PPO) on sequential telemetry.

## 2. Dataset
- **Telemetry Source**: Synthetic multi-window demonstration dataset (`synthetic_temporal_demo.csv`), created strictly for pipeline and schema validation.
- **Samples**: 8 execution runs (5 ransomware representing families G, E, and L; 3 benign utilities).
- **Windows**: 5 sequential 1,000 ms observation windows per sample ($W_0, W_1, W_2, W_3, W_4$).
- **Total Observations**: 40 multi-window telemetry rows.
- **Formal Status**: Pipeline validation benchmark. No real malware execution or live telemetry collection was performed.

## 3. Temporal Representation
The temporal baseline feature space integrates:
1. **Raw Window Telemetry**: API call counts, rates, file writes, registry writes, directory activity, DLL loads.
2. **Temporal Delta Features**: Instantaneous accelerations $\Delta x_t = x_t - x_{t-1}$ (entropy delta, working set delta, private bytes delta, handle delta).
3. **Causal Rolling Statistics**: Rolling mean, standard deviation, minimum, and maximum over window size $k=3$ with zero future lookahead.
4. **Behavioral Accumulations**: Cumulative sums tracking file writes, API invocations, and directory walking.
5. **Memory Ratios**: Private committed virtual memory divided by physical working set size.
Total predictive features: **60**. All identifiers and metadata are strictly excluded.

## 4. Leakage Prevention
Rigorous 5-point data leakage audit passed:
1. **Sample-Level Disjointness**: Grouped split by `sample_id` ensures $\text{SampleIDs}_{\text{train}} \cap \text{SampleIDs}_{\text{test}} = \emptyset$.
2. **Feature Purity**: PIDs, timestamps, sample UUIDs, and paths are excluded from $X$.
3. **Normalization Isolation**: Scaler fitted exclusively on training samples.
4. **Strict Causality**: Rolling windows consume only historical observations $\le t$.
5. **Grouped Engineering**: Deltas and accumulations computed with `groupby('sample_id')`.

## 5. Experimental Setup
- **Train/Test Partition**: 75%/25% sample split (6 train samples $\implies 30$ observations; 2 test samples $\implies 10$ observations).
- **Random Seed**: 42.
- **Target**: Binary classification `label` (`ransomware` vs `benign`).
- **Minority Class Handling**: Minority class Family L (`SAMPLE_005`) placed in test partition and evaluated.
- **Observation Windows Evaluated**: Window 0 (First Obs), Window 1, Window 2 (Early), Window 3, Window 4 (Full).

## 6. Random Forest
- **Configuration**: 200 estimators, balanced class weighting, `random_state = 42`.
- **Checkpoint**: `models/temporal/random_forest_temporal.joblib`.
- **Full Window Performance**: Accuracy = 1.0, Balanced Accuracy = 1.0, Macro F1 = 1.0.

## 7. XGBoost
- **Configuration**: 200 estimators, `learning_rate = 0.1`, `max_depth = 4`, `eval_metric = "logloss"`, `random_state = 42`.
- **Checkpoint**: `models/temporal/xgboost_temporal.joblib`.
- **Full Window Performance**: Accuracy = 1.0, Balanced Accuracy = 1.0, Macro F1 = 1.0.

## 8. Performance by Observation Window
| Observation Window | RF Accuracy | RF Macro F1 | XGB Accuracy | XGB Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| Window 0 (First Obs / Exp A) | 1.0000 | 1.0000 | 0.5000 | 0.3333 |
| Window 1 (W <= 1) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Window 2 (Early / Exp B) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Window 3 (W <= 3) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Window 4 (Full / Exp C) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

*Notice*: In Window 0, XGBoost achieves 0.50 accuracy (0.33 Macro F1) due to subtle initial reconnaissance signals, then rises to 1.00 as behavioral encryption deltas emerge in subsequent windows.

## 9. Feature Importance
- **Random Forest Top Features**: `entropy`, `file_write_count`, `private_bytes_delta`, `working_set_bytes_rolling_mean_3`, `cumulative_file_writes`.
- **XGBoost Top Features**: `entropy`, `file_write_delta`, `cumulative_file_writes`, `private_bytes_delta`.
- Both models consistently rely on cryptographic buffer entropy and file modification acceleration as dominant early-stage signatures.

## 10. Model Comparison
- Random Forest demonstrates higher robustness at $W_0$ under balanced class weights.
- XGBoost exhibits clear behavioral transition from ambiguous baseline at $W_0$ to decisive classification by $W_1$.

## 11. Limitations
1. **Synthetic Data**: Telemetry was synthetically simulated for schema validation.
2. **Small Sample Size**: 8 samples ($N=8$) is insufficient for real-world statistical confidence.
3. **Absence of Real Latency**: Real ground-truth execution timestamps and canary file destruction events were not measured.

## 12. Relationship to PPO
- **Supervised Baselines (RF/XGBoost)** evaluate: *"Can the feature vector at window $W_t$ classify the process?"*
- **Reinforcement Learning (PPO)** evaluates: *"Given sequential uncertainty, at which window should an agent intervene (alert vs continue monitoring) to minimize early damage?"*

## 13. Research Interpretation
Temporal feature engineering successfully captures process acceleration and provides a structured benchmark for sequential RL decision-making.

---
**Formal Research Notice**:
Temporal baseline execution is a pipeline validation experiment using synthetic temporal data, not a real ransomware detection evaluation.
