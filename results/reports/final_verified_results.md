# Final Independently Verified Research Results

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Review Status:** Verified for Publication Documentation  

| Model | Feature Set | Features | Accuracy | Bal Acc | Precision | Recall | F1 | Macro F1 | Minority Recall | Result Source | Verification Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| Random Forest (Full) | Full Dynamic Sandbox | 46 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Holdout Test Set (350 samples) | PASS (Independently Verified) |
| XGBoost (Full) | Full Dynamic Sandbox | 46 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Holdout Test Set (350 samples) | PASS (Independently Verified) |
| Random Forest (Early-Stage) | Early-Stage Non-Redundant | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Holdout Test Set (350 samples) | PASS (Independently Verified) |
| XGBoost (Early-Stage) | Early-Stage Non-Redundant | 16 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Holdout Test Set (350 samples) | PASS (Independently Verified) |
| Temporal Random Forest | Causal Multi-Window (W <= 4) | 60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Temporal Demonstration Test Set | PASS (Pipeline Validation) |
| Temporal XGBoost | Causal Multi-Window (W <= 4) | 60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Temporal Demonstration Test Set | PASS (Pipeline Validation) |
| Temporal PPO Agent | Causal Telemetry + Window Index | 60 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Sequential MDP Evaluation Traces | PASS (Simulation Validation) |
