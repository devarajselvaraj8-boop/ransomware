# Unified Research Evaluation Summary

## 1. Master Model Comparison

| Model | Data Type | Feature Count | Learning Type | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Minority Recall | False Alert Rate | Average Decision Window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static Random Forest | Static Sandbox (Real) | 46 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | NA | NA |
| Static XGBoost | Static Sandbox (Real) | 46 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | NA | NA |
| Early-Stage Random Forest | Feature-Restricted Static (Real) | 16 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | NA | NA |
| Early-Stage XGBoost | Feature-Restricted Static (Real) | 16 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | NA | NA |
| Temporal Random Forest | Multi-Window Temporal (Synthetic Demo) | 60 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | NA | NA |
| Temporal XGBoost | Multi-Window Temporal (Synthetic Demo) | 60 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | NA | NA |
| Temporal PPO | Sequential Observations (Synthetic Demo) | 60 | Reinforcement Learning (Policy) | NA | NA | NA | NA | 1.0 | 0.0 | 3.5 |


## 2. Methodological Distinctions

- **Supervised Static Baselines (Models 1-4)**: Classify static process snapshots. Evaluated on the real 1,750-sample dataset.
- **Temporal Baselines (Models 5-6)**: Classify multi-window observation sequences using causal deltas, rates, and rolling statistics.
- **Temporal PPO (Model 7)**: Solves the sequential optimal stopping problem: deciding *when* to alert vs *when* to continue monitoring.

## 3. Scientific Caveat

Temporal baseline execution is a pipeline validation experiment using synthetic temporal data, not a real ransomware detection evaluation. No real malware execution was performed.
