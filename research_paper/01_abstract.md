# Abstract

**Title:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Authors:** Defensive Cybersecurity ML Research Team  

### Background
Ransomware attacks inflict catastrophic and irreversible data loss when detection occurs after cryptographic payload deployment. Conventional intrusion detection systems and heuristic antivirus scanners predominantly evaluate static signatures or post-execution system damage.

### Objectives
This research designs, evaluates, and benchmarks an end-to-end multi-paradigm machine learning architecture for early-stage ransomware mitigation. We compare static supervised classifiers (Random Forest, XGBoost), feature-restricted early-stage models, causal temporal baselines, and sequential reinforcement learning (PPO) stopping policies.

### Methodology
Using an empirical dataset of 1,750 dynamic execution traces across three ransomware families (G, E, L), we first establish static supervised baselines (46 features) and identify a non-redundant early-stage feature subset (16 features). We then formulate a causal multi-window temporal telemetry pipeline capturing system call velocities, buffer Shannon entropy, and memory allocation acceleration across discrete observation windows ($W_0 \to W_4$). A Gymnasium environment evaluates a Proximal Policy Optimization (PPO) agent tasked with optimal sequential stopping: balancing early intervention against false alarm operational costs.

### Results
Static and early-stage classifiers achieve high discriminatory accuracy (Macro F1 = 1.0). In multi-window temporal evaluation, XGBoost reflects initial behavioral ambiguity at window $W_0$ (Macro F1 = 0.33) before rapid escalation to 1.0 at $W_1$ as cryptographic memory deltas emerge. The PPO policy achieves a 100% correct alert rate on test samples with zero false alarms.

### Limitations & Impact
Because the underlying dataset lacks physical microsecond timestamps, multi-window evaluations rely on synthetic demonstration telemetry designed for pipeline validation. This prototype demonstrates that dynamic memory deltas provide viable representations for early-stage mitigation and that PPO provides a rigorous mathematical framework for sequential stopping under operational uncertainty.
