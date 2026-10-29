# Architectural Design: Proximal Policy Optimization (PPO) for Early-Stage Ransomware Detection

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Document:** PPO Reinforcement Learning Architectural Specification  
**Status:** Design Document (Phase 2 Preparation)  

---

## 1. Motivation for Reinforcement Learning in Defensive Detection

Traditional static and single-shot machine learning classifiers evaluate an entire feature vector all at once. In a real-time host environment, however, defensive detection is an **optimal stopping and monitoring problem under uncertainty**:

- An agent observes sequential signals emitted by a running process.
- Premature termination of a benign process disrupts critical user work (false positive).
- Delayed intervention allows ransomware to begin irreversibly encrypting disk sectors (false negative).
- Continuous deep memory introspection consumes host CPU, disk I/O, and battery resources.

By framing early-stage detection as a **Partially Observable Markov Decision Process (POMDP)**, a Reinforcement Learning agent trained via **Proximal Policy Optimization (PPO)** can learn an optimal policy that dynamically balances detection speed, diagnostic accuracy, and monitoring overhead.

---

## 2. Safe Simulated Environment Specification

Per research ethics and safety requirements:
- **No live malware or executable execution environment will be used.**
- The environment will be implemented as a safe, simulated Gymnasium/OpenAI Gym environment (`RansomwareDefenseEnv`) using synthetic trajectory sampling derived from the dynamic feature distributions of `cleaned_dataset.csv`.
- Each episode represents a process execution lifecycle lasting up to $T_{\max} = 100$ discrete time-steps.
- At step $t = 0$, the process initializes. Feature values incrementally reveal over time, simulating the gradual accumulation of API calls, memory allocations, and filesystem queries.

---

## 3. State Space ($S_t$)

At each time-step $t$, the state vector $s_t \in \mathbb{R}^d$ provided to the PPO agent consists of:

1. **Normalized Feature Vector ($x_t \in [0, 1]^{46}$):**
   - Current observable dynamic behavioral and memory features (e.g. cumulative API calls up to $t$, current memory entropy, DLL loaded count).
2. **Temporal Dynamics Vector ($\Delta x_t$):**
   - First-order rate of change of features over the preceding $k$ steps:
     $$\Delta x_t = x_t - x_{t-1}$$
   - Captures sudden behavioral spikes (e.g., sudden burst in file enumeration or encryption API calls).
3. **Internal Agent Context ($c_t \in \mathbb{R}^3$):**
   - Normalized elapsed time: $t / T_{\max}$.
   - Current monitoring fidelity level: $\mu_t \in \{0.2, 0.6, 1.0\}$.
   - Observation noise level / signal confidence estimate.

---

## 4. Discrete Action Space ($A$)

The PPO agent selects from a discrete action space of size 4:

$$A = \{0, 1, 2, 3\}$$

| Action ID | Action Name | Semantic Meaning | Environmental Effect |
| :--- | :--- | :--- | :--- |
| **0** | `CONTINUE_MONITORING` | Process appears normal / benign. Maintain current low-overhead sampling. | Episode advances to $t+1$; minimal computational cost. |
| **1** | `INCREASE_MONITORING` | Ambiguous or mild anomalies detected. Elevate memory inspection frequency. | Increases telemetry granularity; incurs a modest resource penalty. |
| **2** | `FLAG_SUSPICIOUS` | Elevated risk identified. Pre-isolate network and snapshot process memory. | Freezes network sockets; triggers lightweight memory forensics. |
| **3** | `TRIGGER_ANALYSIS` | Malicious ransomware behavior confirmed. Terminate process and alert defense engine. | Terminal action. Process killed; episode concludes. |

---

## 5. Reward Function Formulation ($R_t$)

The reward structure directly encodes defensive cybersecurity economics, strongly penalizing ransomware escapes and false alarms while incentivizing rapid, early detection.

At non-terminal steps ($t < T$, action $a_t \in \{\text{CONTINUE}, \text{INCREASE}\}$):

$$R_{\text{step}}(s_t, a_t) = -\lambda_{\text{cost}} \cdot c(a_t)$$

Where:
- $c(\text{CONTINUE\_MONITORING}) = 0.01$ (nominal background overhead)
- $c(\text{INCREASE\_MONITORING}) = 0.05$ (telemetry logging and memory polling cost)

At terminal steps (action $a_t = \text{TRIGGER\_ANALYSIS}$ or episode timeout $t = T_{\max}$):

| Outcome Condition | Event Description | Reward Calculation | Target Value |
| :--- | :--- | :--- | :--- |
| **True Positive (Early)** | Agent triggers analysis on ransomware at step $t \ll T_{\max}$ | $R_{\text{TP}} = +100 \times \left(1 - \frac{t}{T_{\max}}\right) + 20$ | **$+50.0 \dots +120.0$** |
| **True Positive (Late)** | Agent triggers analysis on ransomware near end of run | $R_{\text{TP}} = +10$ | **$+10.0$** |
| **False Positive** | Agent triggers analysis on a benign/normal process | $R_{\text{FP}} = -50.0$ | **$-50.0$** |
| **False Negative (Missed)** | Ransomware reaches $T_{\max}$ without being terminated | $R_{\text{FN}} = -150.0$ | **$-150.0$** |
| **True Negative** | Benign process runs to completion without interruption | $R_{\text{TN}} = +30.0$ | **$+30.0$** |

---

## 6. PPO Policy Architecture & Hyperparameters

### Actor-Critic Neural Network:
- **Shared Trunk:** 2-layer MLP (128 units, ReLU activation).
- **Actor Head:** Linear layer $\to$ Softmax outputting action probabilities $\pi_\theta(a \mid s_t)$ over the 4 discrete actions.
- **Critic Head:** Linear layer outputting estimated state-value function $V_\phi(s_t)$.

### PPO Hyperparameter Specifications:
- **Optimizer:** Adam ($\text{lr} = 3 \times 10^{-4}$, linear decay)
- **Discount Factor ($\gamma$):** 0.99
- **GAE Parameter ($\lambda$):** 0.95
- **Clip Coefficient ($\epsilon$):** 0.20
- **Entropy Regularization ($\beta$):** 0.01 (promotes early exploratory monitoring)
- **Rollout Length:** 2,048 environment steps per update
- **Mini-batch Size:** 64
- **Epochs per Update:** 10

---

## 7. Next Steps for Implementation

1. Validate baseline ML pipeline (Random Forest and XGBoost) and ensure reproducibility.
2. Build `src/environment.py` implementing the Gymnasium-compliant simulated environment.
3. Build `src/train_ppo.py` utilizing Stable-Baselines3 or PyTorch for PPO policy optimization.
4. Benchmark PPO early-intervention performance against static baseline decision thresholds.

