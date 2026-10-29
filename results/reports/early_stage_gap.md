# Research Gap: Early-Stage Ransomware Detection via Dynamic Memory Analysis

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Document:** Research Gap & Future Data Requirements  
**Status:** Theoretical and Architectural Gap Analysis (Updated Post Feature-Restricted Baseline)  

---

## 1. Defining "Early-Stage" in Defensive Cybersecurity

The overarching goal of this research initiative is:

> *"Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning"*

In real-world endpoint protection, ransomware exhibits a multi-phase kill-chain:
1. **Pre-Execution / Delivery:** Dropper execution, packing unpacking, anti-analysis checks.
2. **Environment Discovery & Staging:** Process enumeration, privilege escalation, volume shadow copy deletion (`vssadmin`), defense evasion.
3. **Key Generation / Exchange:** Cryptographic initialization, command-and-control (C2) communication, entropy elevation in memory.
4. **Mass Encryption & Destruction:** File traversal, reading plaintext, writing ciphertext, renaming/deleting originals.
5. **Ransom Demand:** Displaying ransom notes, terminating processes.

**Crucial Definition:**  
Detection is only **"early-stage"** if it occurs during **Stages 1 through 3**, *before* Stage 4 (mass encryption) causes irreversible loss of user data. Once Stage 4 has completed, detection is merely a post-mortem autopsy, regardless of model accuracy.

---

## 2. Insights & Limitations from the Feature-Restricted Baseline Proxy

In our second experiment (`notebooks/03_early_stage_baseline.ipynb`), we evaluated an **Early-Stage Feature-Restricted Proxy**:
- Late-stage mass filesystem features (`file_created`, `file_read`, `directory_enumerated`) and post-run sandbox artifacts (`log`, `action`, `errors`) were removed.
- Classifiers trained on the remaining 16 non-redundant early-stage candidate features (process creation, argument vectors, DLL invocations, registry queries, memory entropy) still achieved near-perfect accuracy ($0.9971 - 1.0000$) and $100\%$ minority class recall.

**Core Research Limitation:**  
While this proves that early-stage telemetry contains sufficient mutual information to discriminate ransomware families, **it remains a proxy experiment**. Because the underlying dataset lacks timestamps, the features are still whole-run scalar aggregates rather than time-series snapshots taken during the first few milliseconds of execution.

---

## 3. Telemetry Required for Genuine Early-Stage Detection

To bridge this fundamental research gap and advance beyond static feature-restricted proxies, defensive pipelines require the following granular, temporal telemetry:

```
+-------------------------------------------------------------------------------+
|                       GENUINE EARLY-STAGE TELEMETRY                           |
+-----------------------------------+-------------------------------------------+
| Vector                            | Telemetry & Data Attributes               |
+-----------------------------------+-------------------------------------------+
| 1. Timestamped Behavioral Events  | - Chronologically ordered event logs with |
|                                   |   microsecond timestamps (t_0, t_1, ...)  |
|                                   | - Inter-arrival times between operations  |
+-----------------------------------+-------------------------------------------+
| 2. Sliding Observation Windows    | - Micro-windows (e.g. t = 50ms, 250ms,    |
|                                   |   1s, 2s, 5s) evaluating behavioral rate  |
|                                   |   of acceleration prior to mass encryption|
+-----------------------------------+-------------------------------------------+
| 3. Sequential API Call Sequences  | - Explicit API execution graphs:          |
|                                   |   [OpenProcess -> VirtualAllocEx ->       |
|                                   |    WriteProcessMemory -> CreateRemoteThrd]|
|                                   | - Frequency and argument delta over time  |
+-----------------------------------+-------------------------------------------+
| 4. Dynamic Memory Snapshots       | - Periodic VMI / memory dumps             |
|                                   | - Dynamic heap/stack entropy deltas       |
|                                   | - In-memory RWX page allocation bursts    |
|                                   | - Cryptographic key schedules in memory   |
+-----------------------------------+-------------------------------------------+
| 5. File-Operation Sequences       | - Early canary file access detection      |
|                                   | - Pre-encryption file handle queries vs.  |
|                                   |   mass write-and-replace bursts           |
+-----------------------------------+-------------------------------------------+
```

---

## 4. Direct Motivation for the PPO Reinforcement Learning Environment

The impossibility of solving optimal real-time intervention through static supervised learning directly motivates our **Proximal Policy Optimization (PPO)** reinforcement learning framework:

1. **Sequential Decision-Making Under Uncertainty:** A real-time defensive agent cannot wait for an execution run to finish. It must make sequential decisions step-by-step ($t_0, t_1, \dots, t_k$) based on incomplete, streaming observations.
2. **Dynamic Cost-Benefit Balancing:** Continuous deep memory analysis incurs CPU/RAM overhead. The agent must dynamically decide whether to maintain low overhead, elevate monitoring fidelity, snapshot memory, or immediately terminate the process.
3. **Simulated Gym Environment:** Since live malware execution is unsafe and unfeasible on static datasets, the PPO agent will be trained in a simulated Markov Decision Process (MDP) environment (`RansomwareDefenseEnv`) that models sequential telemetry accumulation derived from empirical sandbox distributions.
