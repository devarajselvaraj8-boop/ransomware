# Discussion

### The Danger of Post-Execution Artifacts
The fact that static classifiers achieve perfect 1.0 accuracy on the baseline dataset highlights a recurring risk in malware research: post-execution feature leakage. When models consume cumulative file write counts measured after hours of sandbox execution, they learn to recognize massive damage rather than early malicious indicators.

### The Value of Temporal Memory Deltas
By transitioning to temporal deltas (rate of handle creation, rapid shifts in Shannon entropy, private heap growth rates), defensive systems can detect ransomware during memory staging, before irreversible file encryption takes place.

### Supervised vs. Reinforcement Learning Roles
Supervised models answer: *"What is the probability this observation is malicious?"*  
Reinforcement learning answers: *"Given the cost of downtime and the cost of false alarms, should I intervene right now or wait for one more window?"*  
The PPO framework provides the operational decision logic necessary for automated endpoint defense.
