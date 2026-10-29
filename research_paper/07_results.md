# Results

### Master Model Comparison Table

| Model | Data Type | Feature Count | Learning Type | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Minority Recall | False Alert Rate | Average Decision Window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Static Random Forest | Static Sandbox (Real) | 46 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | nan | nan |
| Static XGBoost | Static Sandbox (Real) | 46 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | nan | nan |
| Early-Stage Random Forest | Feature-Restricted Static (Real) | 16 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | nan | nan |
| Early-Stage XGBoost | Feature-Restricted Static (Real) | 16 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | nan | nan |
| Temporal Random Forest | Multi-Window Temporal (Synthetic Demo) | 60 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | nan | nan |
| Temporal XGBoost | Multi-Window Temporal (Synthetic Demo) | 60 | Supervised (Classification) | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | nan | nan |
| Temporal PPO | Sequential Observations (Synthetic Demo) | 60 | Reinforcement Learning (Policy) | nan | nan | nan | nan | 1.0 | 0.0 | 3.5 |


### Key Empirical Observations
1. **Supervised Baseline Parity**: Random Forest and XGBoost achieve identical high performance on both full (46) and early-stage non-redundant (16) feature sets.
2. **Window Progression Transition**: In Window $W_0$, XGBoost registers 0.50 accuracy (0.33 Macro F1), reflecting initial behavioral ambiguity before rising to 1.0 in Window $W_1$.
3. **PPO Sequential Efficiency**: The PPO agent achieves a 100% correct alert rate on test samples with zero false alarms.
