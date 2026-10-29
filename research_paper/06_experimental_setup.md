# Experimental Setup

### Supervised Baseline Models
- **Random Forest**: 200 estimators, balanced class weights, Gini criterion, `random_state = 42`.
- **XGBoost**: 200 estimators, learning rate = 0.1, max depth = 4-6, histogram tree method, `random_state = 42`.

### Reinforcement Learning Agent (PPO)
- **Library**: Stable-Baselines3 (v2.9.0).
- **Architecture**: MlpPolicy, 2 hidden layers $\times$ 64 units.
- **Action Space**: Discrete(3) — CONTINUE_MONITORING, SUSPICIOUS, RANSOMWARE_ALERT.
- **Reward Configuration**:
  - Alert True Positive: $+10.0 + 2.5 \times (4 - t)$
  - Alert False Positive: $-10.0$
  - Monitoring Step Cost: $-0.1$
  - Final Miss Penalty: $-15.0$
  - Final Benign Cleared: $+5.0$
