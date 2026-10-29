# Temporal Reinforcement Learning (PPO) Analysis

## 1. Executive Summary
This experiment integrates Proximal Policy Optimization (PPO) with sequential multi-window temporal observations ($W_0 \to W_4$). Unlike static or feature-revealing proxies, the agent observes true progressive telemetry vectors at each discrete observation window.

## 2. Sequential Decision Metrics

- **Correct Alert Rate (True Positive Alerts)**: 100.0%
- **False Alert Rate (False Alarms on Benign)**: 0.0%
- **Miss Rate (Undetected Ransomware)**: 0.0%
- **Average Decision Observation Window**: 3.50
- **Average Episode Reward**: 8.95

## 3. Test Trace Evaluation

| Sample ID | Family | True Class | PPO Final Decision | Decision Window | Episode Reward |
|:---|:---:|:---:|:---:|:---:|:---:|
| SAMPLE_005 | L | Ransomware | RANSOMWARE_ALERT | 3 | 13.40 |
| SAMPLE_008 | benign | Benign | BENIGN_CLEARED | 4 | 4.50 |

## 4. Scientific Notice
Temporal baseline execution is a pipeline validation experiment using synthetic temporal data, not a real ransomware detection evaluation. No real malware execution was performed.
