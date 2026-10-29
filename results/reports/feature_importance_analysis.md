# Feature Importance & Explainability Analysis

## 1. Executive Summary
This analysis investigates the predictive signals across three experimental representations:
1. **Static Cumulative Telemetry**: Dominated by post-execution file/resource metadata (`filetype`, `pe_res_name`, `apistats`).
2. **Early-Stage Feature-Restricted**: Dominated by behavioral API activity, program structures, and registry actions (`apistats`, `program`, `proc`).
3. **Temporal Multi-Window Telemetry**: Dominated by instantaneous activity rates, memory dynamic allocations, and rolling moments (`api_count`, `private_bytes_rolling_max_3`, `working_set_bytes_rolling_min_3`).

## 2. Predictive Importance vs. Causality
We explicitly emphasize that tree-based feature importance reflects **statistical predictive utility** within the given dataset distributions, rather than direct causal drivers of ransomware creation.

## 3. PPO Sequential Action Drivers
- At Window $W_0$, initial baseline activity is low. The policy balances monitoring cost against false alarms.
- High velocity in private virtual memory allocation combined with rapid API invocation triggers early stopping alerts before encryption escalation.

## 4. Scientific Notice
Temporal baseline execution is a pipeline validation experiment using synthetic temporal data, not a real ransomware detection evaluation. No real malware execution was performed.
