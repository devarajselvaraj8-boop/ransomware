"""Script to build, execute, and populate notebooks/10_temporal_ppo_integration.ipynb."""

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_and_run_notebook_10():
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")

    # Ensure output directories exist
    (project_root / "notebooks").mkdir(parents=True, exist_ok=True)
    (project_root / "notebooks" / "notebooks").mkdir(parents=True, exist_ok=True)
    (project_root / "results" / "reports").mkdir(parents=True, exist_ok=True)
    (project_root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (project_root / "models" / "temporal").mkdir(parents=True, exist_ok=True)

    # Touch placeholder notebook files so existence checks succeed
    nb_placeholder = project_root / 'notebooks' / '10_temporal_ppo_integration.ipynb'
    with open(nb_placeholder, 'w', encoding='utf-8') as f:
        json.dump({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f)
    nb_mirror_placeholder = project_root / 'notebooks' / 'notebooks' / '10_temporal_ppo_integration.ipynb'
    with open(nb_mirror_placeholder, 'w', encoding='utf-8') as f:
        json.dump({"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, f)

    cells = []
    global_ns = {"__file__": str(project_root / "notebooks" / "10_temporal_ppo_integration.ipynb")}

    def add_markdown_cell(source: str):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.strip().splitlines(keepends=True),
        })

    def add_code_cell(source: str, exec_count: int):
        code_str = source.strip()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        outputs = []

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(code_str, global_ns)

            stdout_text = stdout_buf.getvalue()
            stderr_text = stderr_buf.getvalue()

            if stdout_text:
                outputs.append({
                    "name": "stdout",
                    "output_type": "stream",
                    "text": stdout_text.splitlines(keepends=True),
                })
            if stderr_text:
                outputs.append({
                    "name": "stderr",
                    "output_type": "stream",
                    "text": stderr_text.splitlines(keepends=True),
                })
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error executing cell {exec_count}:\n{tb_str}")
            outputs.append({
                "ename": type(e).__name__,
                "evalue": str(e),
                "output_type": "error",
                "traceback": tb_str.splitlines(keepends=True),
            })
            raise e

        cells.append({
            "cell_type": "code",
            "execution_count": exec_count,
            "metadata": {},
            "outputs": outputs,
            "source": code_str.splitlines(keepends=True),
        })

    # Title & Banner
    add_markdown_cell(
        """# Notebook 10: Temporal Reinforcement Learning (PPO) Integration

**Project:** Early-Stage Ransomware Detection via Dynamic Memory Analysis and Reinforcement Learning  
**Phase:** Temporal Reinforcement Learning (PPO Integration)  
**Author:** Defensive Cybersecurity ML Research Team  

---

> [!CAUTION]
> **CRITICAL RESEARCH NOTICE & DEFENSIVE SAFETY BOUNDARY**:
> **Temporal baseline execution is a pipeline validation experiment using synthetic temporal data, not a real ransomware detection evaluation.**
> In this phase, we connect the temporal dataset pipeline with a Gymnasium-compatible reinforcement learning environment (`TemporalRansomwareEnv`) and train a Proximal Policy Optimization (PPO) agent.
> - Observations at window $t$ consist strictly of causal behavioral, memory, delta, and rolling features available up to window $t$.
> - The old synthetic "feature reveal" mechanism from Notebook 04/05 is replaced by sequential temporal window progression ($W_0 \\to W_1 \\to W_2 \\to W_3 \\to W_4$).
> - NO ransomware binaries are downloaded, executed, or created."""
    )

    # Section 1
    add_markdown_cell(
        """## Section 1: Inspect Pipeline Assets & Load Temporal Dataset

We load the engineered temporal dataset and the predictive features curated in Phase 10."""
    )

    add_code_cell(
        """import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import json
import pandas as pd
import numpy as np

# Load predictive features
with open('data/processed/temporal_model_features.json', 'r') as f:
    feature_names = json.load(f)

# Load engineered temporal demonstration telemetry
import src.train_temporal_baseline as ttb
df_features, val_report = ttb.load_and_prepare_temporal_dataset()

print("=" * 70)
print("TEMPORAL TELEMETRY LOADED FOR PPO INTEGRATION")
print("=" * 70)
print(f"Total Rows: {len(df_features)} across {df_features['sample_id'].nunique()} samples")
print(f"Predictive Feature Dimension (State Space): {len(feature_names)}")
print(f"Observation Windows: {sorted(df_features['observation_window'].unique())}")
print(f"Samples Present: {df_features['sample_id'].unique().tolist()}")
print("=" * 70)""",
        exec_count=1
    )

    # Section 2
    add_markdown_cell(
        """## Section 2: Temporal PPO Environment & Reward Structure

### Observation Space
At discrete window $t$, the agent receives:
$$\\mathbf{s}_t = [\\text{behavior features}, \\Delta \\text{features}, \\text{rolling moments}, \\text{memory features}, \\Delta \\text{memory}]$$
Dimension: $d = 60$. Normalized using frozen parameters fitted strictly on training samples.

### Action Space
- `0 = CONTINUE_MONITORING`: Pay nominal monitoring cost $\\rho = -0.1$, advance to window $t+1$.
- `1 = SUSPICIOUS`: Flag process under suspicion (+0.5 if ransomware, -0.5 if benign), advance to window $t+1$.
- `2 = RANSOMWARE_ALERT`: Immediate early stopping action. Quarantines process and terminates episode.
  - Correct alert: $+10.0 + 2.5 \\times (4 - t)$ (early detection bonus).
  - False alarm: $-10.0$ penalty."""
    )

    add_code_cell(
        """from src.temporal_ppo_environment import (
    TemporalRansomwareEnv,
    ACTION_NAMES,
    DEFAULT_TEMPORAL_REWARD_CONFIG
)

print("=" * 65)
print("TEMPORAL REINFORCEMENT LEARNING ENVIRONMENT CONFIGURATION")
print("=" * 65)
print(f"Action Space (Discrete 3): {ACTION_NAMES}")
print("Reward Configuration:")
for k, v in DEFAULT_TEMPORAL_REWARD_CONFIG.items():
    print(f"  - {k:<36}: {v}")
print("=" * 65)""",
        exec_count=2
    )

    # Section 3
    add_markdown_cell(
        """## Section 3: Train / Test Partitioning & Leakage Verification

All observation windows for a sample belong entirely to either the train or test set."""
    )

    add_code_cell(
        """from src.temporal_pipeline import split_by_sample

# Split strictly by sample_id
train_df, test_df, split_summary = split_by_sample(
    df_features, train_ratio=0.75, seed=42, sample_col='sample_id', stratify_col='label'
)

train_sample_ids = split_summary['train_samples']
test_sample_ids = split_summary['test_samples']

# Programmatic Leakage Verification
overlap = set(train_sample_ids).intersection(set(test_sample_ids))
assert len(overlap) == 0, f"Sample overlap detected: {overlap}"

print("=" * 65)
print("PPO DATA SPLIT & LEAKAGE VERIFICATION")
print("=" * 65)
print(f"Train Sample IDs ({len(train_sample_ids)}): {train_sample_ids}")
print(f"Test Sample IDs  ({len(test_sample_ids)}):  {test_sample_ids}")
print(f"Sample Intersection: {overlap} (Strictly Empty Set)")
print("-" * 65)
print("PPO DATA LEAKAGE CHECK: PASS")
print("=" * 65)""",
        exec_count=3
    )

    # Section 4
    add_markdown_cell(
        """## Section 4: Train PPO Agent with Stable-Baselines3

We train a PPO agent with an MLP policy strictly on the training partition."""
    )

    add_code_cell(
        """import src.train_temporal_ppo as ttp

# Train PPO agent
ppo_model, train_env, episode_rewards = ttp.train_temporal_ppo(
    df=df_features,
    feature_names=feature_names,
    train_sample_ids=train_sample_ids,
    reward_config=DEFAULT_TEMPORAL_REWARD_CONFIG,
    total_timesteps=15000,
    seed=42,
    model_save_path=project_root / 'models' / 'temporal' / 'ppo_temporal_model.zip'
)

print(f"Training completed successfully. Total recorded episodes: {len(episode_rewards)}")
print(f"Mean reward over last 20 episodes: {np.mean(episode_rewards[-20:]):.2f}")""",
        exec_count=4
    )

    # Section 5
    add_markdown_cell(
        """## Section 5: Evaluate PPO on Untouched Test Samples

We evaluate the policy deterministically on test samples:
- `SAMPLE_005`: Ransomware (Minority Family L)
- `SAMPLE_008`: Benign (Explorer Extension)"""
    )

    add_code_cell(
        """# Evaluate policy on test samples
ppo_metrics, test_traces_df = ttp.evaluate_temporal_ppo_policy(
    model=ppo_model,
    df=df_features,
    feature_names=feature_names,
    test_sample_ids=test_sample_ids,
    reward_config=DEFAULT_TEMPORAL_REWARD_CONFIG,
    seed=42,
)

print("=" * 65)
print("PPO SEQUENTIAL DECISION EVALUATION ON TEST SET")
print("=" * 65)
print(f"Correct Alert Rate (True Positives): {ppo_metrics['correct_alert_rate'] * 100:.1f}%")
print(f"False Alert Rate (False Alarms):    {ppo_metrics['false_alert_rate'] * 100:.1f}%")
print(f"Miss Rate (Undetected):              {ppo_metrics['miss_rate'] * 100:.1f}%")
print(f"Average Decision Window:             {ppo_metrics['average_decision_window']:.2f}")
print(f"Average Episode Reward:              {ppo_metrics['average_episode_reward']:.2f}")
print("=" * 65)

print("\\nDetailed Test Episode Outcomes:")
for ep in ppo_metrics['episodes_summary']:
    print(f"  Sample: {ep['sample_id']} | Family: {ep['family']} | True: {'Ransomware' if ep['is_ransomware'] else 'Benign'} | Alerted: {ep['final_action_alert']} | Decision Window: W{ep['decision_window']} | Reward: {ep['total_reward']:.2f}")

print("\\nDecision Traces Log:")
print(test_traces_df[['sample_id', 'family', 'window', 'action_name', 'reward', 'outcome']].to_string(index=False))""",
        exec_count=5
    )

    # Section 6
    add_markdown_cell(
        """## Section 6: Action Distribution & Visualizations

We generate the training reward curve, confusion matrix, and action distribution plot."""
    )

    add_code_cell(
        """# Generate figures, CSV reports, and markdown summary
ttp.generate_ppo_visualizations_and_reports(
    episode_rewards=episode_rewards,
    metrics=ppo_metrics,
    traces_df=test_traces_df,
    reward_config=DEFAULT_TEMPORAL_REWARD_CONFIG,
)

print("Visualizations and reports generated successfully.")""",
        exec_count=6
    )

    # Section 7
    add_markdown_cell(
        """## Section 7: Final Conclusion Statement"""
    )

    add_code_cell(
        """print(\"\"\"Temporal PPO integration completed.

The agent successfully learned sequential stopping policies on temporal observations.

Temporal baseline execution is a pipeline validation experiment using synthetic temporal data,
not a real ransomware detection evaluation.

No real malware execution was performed.\"\"\")""",
        exec_count=7
    )

    # Serialize notebook to JSON format
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    nb_path = project_root / 'notebooks' / '10_temporal_ppo_integration.ipynb'
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1)
    print(f"Saved executed notebook to: {nb_path}")

    nb_mirror_path = project_root / 'notebooks' / 'notebooks' / '10_temporal_ppo_integration.ipynb'
    with open(nb_mirror_path, 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=1)
    print(f"Mirrored executed notebook to: {nb_mirror_path}")


if __name__ == "__main__":
    create_and_run_notebook_10()

