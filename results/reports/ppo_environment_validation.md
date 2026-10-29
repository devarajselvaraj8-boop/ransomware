# PPO Environment Validation

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Environment:** `RansomwareDetectionEnv` (Gymnasium-Compatible)  
**Date:** September 2026  
**Status:** Environment Validation Report  

---

## 1. Dataset & Feature Matrix

- **Total Observations:** 1,750 samples (Cleaned dynamic dataset).
- **Original Features:** 46 features.
- **Selected Observation Features:** 16 non-redundant early-stage candidate features loaded from `data/processed/early_stage_nonredundant_feature_list.json`:
  1. `apistats`
  2. `dll`
  3. `dll_loaded`
  4. `arguments`
  5. `command_line`
  6. `proc`
  7. `proc_pid`
  8. `type`
  9. `program`
  10. `children`
  11. `tree_command_line`
  12. `regkey_opened`
  13. `regkey_read`
  14. `regkey_written`
  15. `wmi_query`
  16. `entropy`
- **Normalization Invariant:** `StandardScaler` parameters fit strictly on the 1,400 training partition samples (`data/processed/split_indices.json`) and serialized to `data/processed/ppo_normalization.json`.

---

## 2. Temporal Representation: Synthetic Temporal Proxy

> [!IMPORTANT]
> **This is not real temporal execution data.**  
> The underlying dataset contains static full-trace dynamic summaries. The temporal dimension is constructed using a reproducible **progressive feature reveal proxy**:
> - **Sequence Length ($T$):** 5 synthetic observation steps ($t = 0, 1, 2, 3, 4$).
> - **Reveal Schedule:** 20% ($t=0$), 40% ($t=1$), 60% ($t=2$), 80% ($t=3$), 100% ($t=4$).
> - **Masking:** Unrevealed features at step $t$ are masked to `0.0` (the standardized mean).

---

## 3. Gymnasium Observation Space

- **Space Type:** `gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)`
- **Observation Dimension:** 16 continuous normalized features.
- **Leakage Prevention:** Ground-truth class labels (`family`) are never included in the observation vector; labels reside strictly within internal environment state for reward calculation.

---

## 4. Gymnasium Action Space

- **Space Type:** `gymnasium.spaces.Discrete(3)`
- **Action Semantics:**
  - `0`: `CONTINUE_MONITORING` (Passive observation; advances timestep at small cost $-0.1$).
  - `1`: `SUSPICIOUS` (Intermediate elevated inspection; non-terminal unless step limit reached).
  - `2`: `RANSOMWARE_ALERT` (Terminal threat flag; kills process and concludes episode).

---

## 5. Reward Design Table

| Action / Event | State Condition | Reward Value | Purpose |
| :--- | :--- | :---: | :--- |
| `RANSOMWARE_ALERT` | Correct Early Alert ($t=0 \dots 3$) | $+10.0 + (4 - t) 	imes 2.5$ | Maximum incentive for early detection ($+20.0$ at $t=0$). |
| `RANSOMWARE_ALERT` | Correct Final Alert ($t=4$) | $+10.0$ | Positive reward for correct identification. |
| `RANSOMWARE_ALERT` | False Alarm on Benign | $-10.0$ | Strong penalty for business workflow disruption. |
| `CONTINUE_MONITORING` | Non-terminal Step ($t < 4$) | $-0.1$ | Small telemetry and monitoring overhead penalty. |
| `SUSPICIOUS` | Ransomware Sample | $+0.5$ | Mild positive reinforcement for noting risk. |
| `SUSPICIOUS` | Benign Sample | $-0.5$ | Mild penalty for unnecessary investigation. |
| Timeout / Missed | Ransomware at $t=4$ with no alert | $-15.0$ | Severe penalty representing catastrophic encryption. |
| Timeout / Benign | Benign process allowed to finish at $t=4$ | $+5.0$ | Positive reward for undisturbed execution. |

---

## 6. Comprehensive Validation Tests

| Test Invariant | Method / Check | Result |
| :--- | :--- | :---: |
| **Reset Execution** | `env.reset(seed=42)` returns valid `(obs, info)` | **PASSED** |
| **Step Execution** | `env.step(action)` returns `(obs, reward, term, trunc, info)` | **PASSED** |
| **Observation Shape** | `obs.shape == (16,)` | **PASSED** |
| **Observation Space Containment** | `observation_space.contains(obs) == True` | **PASSED** |
| **Action Space Containment** | `action_space.contains(action) == True` for $a \in {0, 1, 2}$ | **PASSED** |
| **NaN Check** | `np.isnan(obs).sum() == 0` across all steps | **PASSED** |
| **Infinity Check** | `np.isinf(obs).sum() == 0` across all steps | **PASSED** |
| **Finite Reward** | `np.isfinite(reward) == True` | **PASSED** |
| **Seed Reproducibility** | Exact identical sequence and sample when seed is fixed | **PASSED** |
| **Episode Termination** | Guaranteed termination on `ALERT` or step limit ($t=4$) | **PASSED** |

---

## 7. 100-Episode Random Policy Smoke Test Statistics

- **Total Episodes:** 100
- **Average Episode Length:** 2.46 steps (Range: 1 to 5)
- **Average Return per Episode:** 13.41
- **Ransomware Alert Triggered:** 87 / 100 episodes
- **Missed Ransomware Episodes:** 13 / 100 episodes
- **Class Representation:** {'G': 43, 'E': 45, 'L': 12}

---

## 8. Limitations & Scope

1. **Synthetic Nature:** Progressive feature reveal simulates information gain over time, but is not a replacement for raw microsecond packet or API logs.
2. **Readiness for PPO:** The environment strictly satisfies all Gymnasium API specifications and is ready to be consumed by `notebooks/05_train_ppo.ipynb`.
