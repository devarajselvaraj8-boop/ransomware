# Comparison: Current Feature-Summary vs. Future Temporal Dataset

| Dimension | Current Dataset | Future Real Dataset |
| --- | --- | --- |
| Data Nature | Static/Dynamic Feature Summary | Continuous Streaming Event & Memory Log |
| Temporal Metadata | None (No timestamps, no sequence index) | High-resolution microsecond timestamps (UTC) |
| Event Ordering | Unordered post-mortem aggregates | Strictly chronological event causality |
| Memory Telemetry | Static overall entropy column only | Dynamic working set, page entropy, handle counts |
| Observation Unit | Single execution summary row per sample | Time-bucketed observation windows (W_0, W_1, ...) |
| Ground Truth | Family label only (G, E, L) | Family label + exact timestamp of malicious onset |
| Measurable Latency | Synthetic observation steps only (t in [0, 4]) | True physical time-to-detection (ms/sec) |