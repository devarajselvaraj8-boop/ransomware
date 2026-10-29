# Introduction

Modern ransomware threats have evolved from simplistic locker programs to sophisticated, multi-stage extortion engines. A fundamental asymmetry exists between attackers and defenders: while a threat actor requires only seconds to initiate automated symmetric encryption loops, human incident responders and static endpoint detection agents often take minutes or hours to confirm malicious intent.

By the time bulk file encryption triggers conventional disk activity thresholds (e.g. thousands of modified files), key files have already been irreversibly compromised. Early-stage detection—intercepting malicious execution during initial reconnaissance, privilege staging, or early key generation—is therefore paramount.

This research formulates an end-to-end defensive framework investigating whether dynamic memory analysis and reinforcement learning can achieve reliable early intervention. We structure the inquiry across three central research questions:
1. **RQ1**: Can static feature spaces be pruned to non-redundant early-stage behavioral signals without degrading classification accuracy?
2. **RQ2**: How do causal temporal deltas (rate of change in entropy, handle counts, and committed private bytes) alter detection accuracy across discrete observation windows?
3. **RQ3**: How does a sequential decision-making reinforcement learning policy (PPO) compare against conventional static supervised classifiers in balancing alert velocity against false alarms?
