# Dataset and Preprocessing

### Empirical Dataset Characteristics
- **Raw Acquisition**: 2,000 samples, 51 columns (50 features, 1 target `family`).
- **Target Distribution**: Ransomware Family G (820), Family E (767), Minority Family L (163).
- **Deduplication**: 250 exact duplicate rows identified and removed.
- **Zero-Variance Pruning**: 4 constant columns (`pcap`, `virustotal`, `crypto_keys`, `file_deleted`) removed.
- **Cleaned Dataset**: 1,750 samples $\times$ 46 features.

### Partitioning & Leakage Prevention
- Partitioned using 80% train / 20% test split with random seed 42.
- Stratified on target labels.
- Verified 0% data leakage across sample boundaries.
