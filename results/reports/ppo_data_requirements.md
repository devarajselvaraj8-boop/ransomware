# PPO Reinforcement Learning Data Requirements & Environment Specification

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Document:** PPO Data Requirements & Simulation Specifications  
**Status:** Architectural Specification (Phase 2 RL Preparation)  

---

## 1. Overview & Defensive Mandate

The goal of the Reinforcement Learning phase is to train a **Proximal Policy Optimization (PPO)** agent capable of optimal dynamic monitoring and early-stage threat termination without requiring live malware execution.

The training environment (`RansomwareDefenseEnv`) will be constructed as a Gymnasium-compliant simulated environment parameterized by the statistical feature distributions of `cleaned_dataset.csv`.

---

## 2. PPO State Space Representation ($S_t$)

At each discrete time-step $t \in [0, T_{\max}]$, the agent receives an observation vector composed of three concatenated components:

$$s_t = \Big[ \mathbf{x}_t \,\|\, \Delta \mathbf{x}_t \,\|\, \mathbf{c}_t \Big]$$

### 1. Current Feature Vector ($\mathbf{x}_t \in [0, 1]^{16}$ or $[0, 1]^{46}$):
- Cumulative behavioral and memory feature values observed up to step $t$.
- Features are normalized to $[0, 1]$ via min-max bounds derived from the training set.

### 2. Recent Observation Dynamics ($\Delta \mathbf{x}_t$):
- First-order temporal difference vector capturing the rate of behavioral change over the last $k$ steps:
  $$\Delta \mathbf{x}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$$
- Reflects burst activity (e.g., sudden spikes in registry modifications or API call volume).

### 3. Agent Context & Previous Monitoring State ($\mathbf{c}_t \in \mathbb{R}^3$):
- **Normalized Step Index:** $t / T_{\max}$ (temporal urgency indicator).
- **Current Monitoring Level ($\mu_t$):** $\mu_t \in \{0.2, 0.6, 1.0\}$ indicating current observation fidelity.
- **Previous Action Executed ($a_{t-1}$):** One-hot or scalar representation of the prior decision.

---

## 3. Discrete Action Space ($A$)

The PPO agent selects from 4 discrete actions:

$$A = \{0, 1, 2, 3\}$$

| Action ID | Action Name | System Semantic | Cost / Operational Effect |
| :---: | :--- | :--- | :--- |
| **0** | `CONTINUE_MONITORING` | Process appears benign or quiescent. Maintain baseline passive monitoring. | Lowest resource overhead ($c_0 = 0.01$). |
| **1** | `INCREASE_MONITORING` | Mild anomaly suspected. Increase API tracing and memory polling rate. | Moderate resource cost ($c_1 = 0.05$). Yields finer telemetry. |
| **2** | `FLAG_SUSPICIOUS` | Significant suspicious activity. Pre-quarantine process network and take memory dump. | Higher cost ($c_2 = 0.15$). Freezes outbound sockets. |
| **3** | `TRIGGER_ANALYSIS` | Malicious ransomware behavior confirmed. Kill process and alert defensive EDR. | Terminal action. Episode concludes immediately. |

---

## 4. Asymmetric Reward Function Formulation ($R_t$)

The reward structure reflects the real-world operational economics of host cybersecurity, strongly prioritizing early threat elimination and low false alarm rates:

### Step-Wise Monitoring Penalty (Non-Terminal Steps $t < T_{\max}$):
$$R_{\text{step}}(s_t, a_t) = -c(a_t)$$
- Unnecessary elevated monitoring incurs a small continuous negative penalty to incentivize lean resource usage.

### Terminal Decision Rewards:
| Outcome | Condition | Reward Value | Rationale |
| :--- | :--- | :---: | :--- |
| **Early True Positive** | Ransomware terminated at $t \ll T_{\max}$ | **$+100 \times (1 - \frac{t}{T_{\max}}) + 20$** | Maximal reward for stopping ransomware before encryption begins. |
| **Late True Positive** | Ransomware terminated near $T_{\max}$ | **$+10.0$** | Positive, but penalized for allowing execution to proceed too far. |
| **True Negative** | Benign process allowed to finish safely at $T_{\max}$ | **$+30.0$** | Reward for zero disruption to normal business workflows. |
| **False Positive** | Benign process killed prematurely (`TRIGGER_ANALYSIS`) | **$-50.0$** | Substantial penalty for disrupting legitimate processes. |
| **False Negative (Missed)** | Ransomware runs to completion ($t = T_{\max}$) unstopped | **$-150.0$** | Severe penalty representing catastrophic disk encryption data loss. |

---

## 5. Safe Simulation Protocol (Zero Malware Execution)

- The environment samples synthetic process trajectories from the conditional probability distributions $P(\mathbf{x}_t \mid \text{family})$ of the dataset.
- Process initialization, API accumulation, and memory entropy elevation are modeled as piecewise stochastic jump processes.
- No live binaries or malicious payloads are executed.

