# Limitations

This research candidly documents four primary limitations:

1. **Static Dataset Limitations**:
   The primary benchmark dataset lacks physical microsecond timestamps.
2. **Synthetic Temporal Demonstration**:
   Multi-window sequential telemetry ($W_0 \to W_4$) was synthetically generated to validate pipeline architecture and Gymnasium interfacing. It must not be cited as empirical real-world detection latency.
3. **Sample Size Constraints**:
   The temporal demonstration dataset evaluates 8 execution samples ($N=8$). Larger sample sizes are required for generalized statistical confidence.
4. **Absence of Real Latency Measurements**:
   Real-world detection latency requires microsecond measurement of the exact interval between process launch and the first canary file modification.
