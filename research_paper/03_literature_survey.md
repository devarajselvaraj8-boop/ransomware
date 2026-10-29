# Literature Survey

Dynamic analysis and machine learning have been widely explored for ransomware detection. Prior research establishes several important foundations:

1. **Static vs. Dynamic Telemetry**:
   Static analysis of PE headers and opcodes frequently suffers from obfuscation, packing, and polymorphic evasion. Dynamic monitoring in sandbox environments captures runtime behaviors including API call sequences, registry operations, and memory modifications [REFERENCE TO BE VERIFIED: Dynamic Analysis Foundations].

2. **Information Entropy as a Cryptographic Marker**:
   Shannon entropy of memory buffers and file write streams has emerged as a primary dynamic feature. Pure plaintext exhibits lower entropy (~4.0-5.5 bits), whereas AES/RSA encrypted ciphertext approaches maximum entropy (~7.8-8.0 bits) [REFERENCE TO BE VERIFIED: Entropy in Ransomware].

3. **Temporal Gaps in Static ML**:
   Most existing machine learning benchmarks compress hours of sandbox execution into single summary rows. Recent literature emphasizes the necessity of temporal execution traces to avoid retrospective bias and prevent post-execution data leakage [REFERENCE TO BE VERIFIED: Temporal Sandbox Leakage].

4. **Reinforcement Learning for Intrusion Detection**:
   While supervised learning treats detection as a one-shot classification problem, reinforcement learning models the defender's problem as a partially observable Markov decision process (POMDP), optimizing the trade-off between observation latency and intervention risk [REFERENCE TO BE VERIFIED: RL in Defensive Cybersecurity].
