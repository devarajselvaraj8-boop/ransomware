# Methodology

The research methodology encompasses five sequential phases:

```
Static Sandbox Telemetry (1,750 samples)
       ↓
Data Cleaning & Non-Redundant Feature Selection (16 features)
       ↓
Causal Multi-Window Temporal Telemetry Pipeline (W0 → W4)
       ↓
Supervised Temporal Baselines (Random Forest & XGBoost)
       ↓
Gymnasium Sequential RL Environment & PPO Training
```

### Causal Feature Formulation
For any metric $x$ at discrete window $t$:
- **Temporal Delta**: $\Delta x_t = x_t - x_{t-1}$ (with $\Delta x_0 = 0.0$).
- **Operational Rate**: $\Delta x_t / \Delta t$.
- **Causal Rolling Moments**: Computed strictly over historical windows $\le t$ with zero future lookahead:
$$\mu_{t, k} = \frac{1}{\min(t+1, k)} \sum_{j=\max(0, t-k+1)}^t x_j$$
