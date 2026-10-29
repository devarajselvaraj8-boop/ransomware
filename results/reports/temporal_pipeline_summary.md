# Research Summary: Temporal Dataset Pipeline Implementation & Validation

## 1. Why Static Rows Are Insufficient for True Early-Stage Temporal Detection
Static tabular datasets compress the entire execution lifetime of a malware sample into single cumulative sums or binary flags. This destroys all temporal sequencing:
- An analyst or ML classifier cannot tell whether high entropy occurred at millisecond 50 (reconnaissance/staging) or minute 15 (after full disk encryption).
- Classifying static rows creates the illusion of early detection because post-damage signals (thousands of file writes) are visible to the model at time $t=0$.

## 2. Why Timestamps and Observation Ordering Are Required
Sequential process monitoring requires monotonic timestamps (e.g. microsecond epoch $\tau_t$ or elapsed relative time $\Delta t$) to compute physical velocities:
- Dynamic encryption exhibits distinct behavioral phases: discovery $\to$ privilege escalation $\to$ key exchange $\to$ file traversal $\to$ mass encryption.
- Timestamps allow calculating genuine operational rates (e.g. file modifications per second), which distinguish automated cryptographic loops from human word-processing activity.

## 3. Why Sample-Level Splitting Is Essential (Zero-Leakage Mandate)
In temporal telemetry, multiple observation windows belong to the same execution run.
- Shuffling observations at the row level places $W_1$ of sample $A$ in train and $W_2$ of sample $A$ in test.
- This creates severe identity leakage: the model memorizes the sample's PID, file paths, and memory footprint, resulting in artificially near-perfect test scores.
- Grouping splits strictly by `sample_id` guarantees that $\text{SampleIDs}_{\text{train}} \cap \text{SampleIDs}_{\text{test}} = \emptyset$.

## 4. How Delta Features Are Calculated
Delta features capture the instantaneous acceleration of system changes:
$$\Delta x_t = x_t - x_{t-1}$$
For the baseline window $W_0$, $\Delta x_0 = 0.0$.
In ransomware, rapid positive spikes in $\Delta \text{entropy}$ and $\Delta \text{private\_bytes}$ mark the initiation of bulk file encryption before high cumulative damage accumulates.

## 5. How Rolling Features Avoid Future Leakage
Rolling window aggregations (mean, standard deviation, minimum, maximum) are computed strictly causally over historical observations up to window $t$:
$$\mu_{t, k} = \frac{1}{\min(t+1, k)} \sum_{j=\max(0, t-k+1)}^t x_j$$
Future observations ($t' > t$) are mathematically excluded from window $t$'s computation.

## 6. How Memory Features Become Temporal Through Deltas
Raw memory values (such as physical working set size or committed private bytes) represent static state snapshots. By calculating memory deltas ($\Delta \text{private\_bytes}$) and memory dynamic rates ($\Delta \text{bytes} / \Delta t$), the pipeline transforms static footprints into dynamic indicators of heap allocations and buffer transformations.

## 7. How Normalization Prevents Test-Set Leakage
Standardization parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$) are computed exclusively on training samples. Test samples are transformed using these frozen training statistics:
$$z_{\text{test}} = \frac{x_{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Features with zero variance are assigned a scale of $1.0$ to eliminate division by zero errors.

## 8. How This Pipeline Will Eventually Feed RF, XGBoost, and PPO
- **RF and XGBoost**: Can consume multi-window feature vectors or sliding window representations ($W_t$ alongside historical rollings) to perform early-window classification.
- **PPO Reinforcement Learning**: Directly consumes the temporal observation vector $\mathbf{s}_t \in \mathbb{R}^d$ at each timestep, choosing between `CONTINUE_MONITORING`, `SUSPICIOUS`, and `RANSOMWARE_ALERT`.

## 9. What Remains Missing for Genuine Real-World Early-Stage Evaluation
While this pipeline establishes the data engineering infrastructure, true operational early-stage validation still requires:
1. Physical sandbox telemetry collection running diverse ransomware families under microsecond kernel monitoring (ETW / eBPF).
2. Ground-truth timestamping of the first irreversible canary file modification.
3. Diverse benign application profiles operating under realistic user interaction stress.

---
**Formal Research Safety Confirmation**:
Temporal pipeline validation completed using synthetic schema-validation data. No real malware execution or telemetry collection was performed.
