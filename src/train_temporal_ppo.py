"""Training and Evaluation Module for Temporal PPO Ransomware Detection.

Trains a PPO agent using Stable-Baselines3 on the Gymnasium TemporalRansomwareEnv.
Evaluates sequential stopping policies on untouched test samples, logs decision traces,
computes sequential detection metrics, and generates visualizations and markdown reports.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from src.temporal_ppo_environment import (
    ACTION_ALERT,
    ACTION_CONTINUE,
    ACTION_NAMES,
    ACTION_SUSPICIOUS,
    DEFAULT_TEMPORAL_REWARD_CONFIG,
    TemporalRansomwareEnv,
)
from src.utils import get_figures_dir, get_models_dir, get_reports_dir, save_json


class TrainingMetricsCallback(BaseCallback):
    """Callback for recording episode reward metrics during PPO training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.current_reward: float = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is not None:
            self.current_reward += float(rewards[0])
        if dones is not None and dones[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True


def train_temporal_ppo(
    df: pd.DataFrame,
    feature_names: List[str],
    train_sample_ids: List[str],
    reward_config: Optional[Dict[str, Any]] = None,
    total_timesteps: int = 15000,
    seed: int = 42,
    model_save_path: Optional[Path] = None,
) -> Tuple[PPO, TemporalRansomwareEnv, List[float]]:
    """Train PPO agent strictly on training samples."""
    if reward_config is None:
        reward_config = DEFAULT_TEMPORAL_REWARD_CONFIG
        
    train_env = TemporalRansomwareEnv(
        df=df,
        feature_names=feature_names,
        split_type="train",
        sample_ids=train_sample_ids,
        reward_config=reward_config,
        seed=seed,
    )

    callback = TrainingMetricsCallback()
    
    # Simple, robust MLP policy
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=seed,
        verbose=0,
    )

    print(f"Training Temporal PPO on {len(train_sample_ids)} training samples for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    if model_save_path is None:
        model_save_path = get_models_dir() / "temporal" / "ppo_temporal_model.zip"
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_save_path))
    print(f"Saved trained PPO model checkpoint to: {model_save_path}")

    return model, train_env, callback.episode_rewards


def evaluate_temporal_ppo_policy(
    model: PPO,
    df: pd.DataFrame,
    feature_names: List[str],
    test_sample_ids: List[str],
    reward_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Evaluate trained PPO policy deterministically on test samples and collect decision traces."""
    if reward_config is None:
        reward_config = DEFAULT_TEMPORAL_REWARD_CONFIG

    test_env = TemporalRansomwareEnv(
        df=df,
        feature_names=feature_names,
        split_type="test",
        sample_ids=test_sample_ids,
        reward_config=reward_config,
        seed=seed,
    )

    traces = []
    episode_summaries = []

    for sid in test_sample_ids:
        obs, info = test_env.reset(options={"sample_id": sid})
        done = False
        step_idx = 0
        ep_reward = 0.0
        final_alert = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action = int(action)
            action_name = ACTION_NAMES[action]

            next_obs, reward, terminated, truncated, step_info = test_env.step(action)
            ep_reward += reward

            traces.append({
                "sample_id": sid,
                "family": info["family"],
                "is_ransomware": info["is_ransomware"],
                "window": step_idx,
                "action": action,
                "action_name": action_name,
                "reward": reward,
                "outcome": step_info["outcome"],
            })

            if action == ACTION_ALERT:
                final_alert = True

            obs = next_obs
            step_idx += 1
            done = terminated or truncated

        episode_summaries.append({
            "sample_id": sid,
            "family": info["family"],
            "is_ransomware": info["is_ransomware"],
            "final_action_alert": final_alert,
            "decision_window": step_idx - 1,
            "episode_length": step_idx,
            "total_reward": ep_reward,
            "final_outcome": step_info["outcome"],
        })

    traces_df = pd.DataFrame(traces)
    episodes_df = pd.DataFrame(episode_summaries)

    # Compute aggregate sequential metrics
    rw_eps = episodes_df[episodes_df["is_ransomware"] == True]
    bn_eps = episodes_df[episodes_df["is_ransomware"] == False]

    n_rw = len(rw_eps)
    n_bn = len(bn_eps)

    # Correct alerts on ransomware
    correct_alerts = int(rw_eps["final_action_alert"].sum()) if n_rw > 0 else 0
    correct_alert_rate = float(correct_alerts / n_rw) if n_rw > 0 else 0.0

    # False alerts on benign
    false_alerts = int(bn_eps["final_action_alert"].sum()) if n_bn > 0 else 0
    false_alert_rate = float(false_alerts / n_bn) if n_bn > 0 else 0.0

    # Miss rate on ransomware
    miss_rate = 1.0 - correct_alert_rate if n_rw > 0 else 0.0

    avg_decision_window = float(episodes_df["decision_window"].mean())
    avg_reward = float(episodes_df["total_reward"].mean())
    avg_length = float(episodes_df["episode_length"].mean())

    metrics = {
        "n_test_samples": len(test_sample_ids),
        "n_ransomware_samples": n_rw,
        "n_benign_samples": n_bn,
        "correct_alert_rate": correct_alert_rate,
        "false_alert_rate": false_alert_rate,
        "miss_rate": miss_rate,
        "average_decision_window": avg_decision_window,
        "average_episode_reward": avg_reward,
        "average_episode_length": avg_length,
        "episodes_summary": episodes_df.to_dict(orient="records"),
    }

    return metrics, traces_df


def generate_ppo_visualizations_and_reports(
    episode_rewards: List[float],
    metrics: Dict[str, Any],
    traces_df: pd.DataFrame,
    reward_config: Dict[str, Any],
) -> None:
    """Generate all Phase 11 figures, CSVs, and markdown reports."""
    # 1. Training Curve
    plt.figure(figsize=(9, 5))
    if len(episode_rewards) > 0:
        plt.plot(episode_rewards, alpha=0.3, color="gray", label="Raw Episode Reward")
        # Moving average
        window = max(5, len(episode_rewards) // 10)
        roll_mean = pd.Series(episode_rewards).rolling(window, min_periods=1).mean()
        plt.plot(roll_mean, color="royalblue", linewidth=2.5, label=f"Rolling Mean (W={window})")
    plt.title("Temporal PPO Training Reward Progression", fontsize=13, fontweight="bold")
    plt.xlabel("Training Episode")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    curve_path = get_figures_dir() / "temporal_ppo_training_curve.png"
    plt.savefig(curve_path, dpi=300)
    plt.close()
    print(f"Saved PPO Training Curve to: {curve_path}")

    # 2. Confusion Matrix
    y_true = []
    y_pred = []
    for ep in metrics["episodes_summary"]:
        y_true.append(1 if ep["is_ransomware"] else 0)
        y_pred.append(1 if ep["final_action_alert"] else 0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=["Benign (No Alert)", "Ransomware Alert"],
                yticklabels=["True Benign", "True Ransomware"])
    plt.title("Temporal PPO Sequential Policy Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("PPO Terminal Decision")
    plt.ylabel("True Class")
    plt.tight_layout()
    cm_path = get_figures_dir() / "temporal_ppo_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved PPO Confusion Matrix to: {cm_path}")

    # 3. Action Distribution Across Windows
    plt.figure(figsize=(8, 5))
    action_counts = traces_df.groupby(["window", "action_name"]).size().unstack(fill_value=0)
    action_counts.plot(kind="bar", stacked=True, ax=plt.gca(), colormap="viridis")
    plt.title("Temporal PPO Action Distribution by Observation Window", fontsize=12, fontweight="bold")
    plt.xlabel("Observation Window (W0 - W4)")
    plt.ylabel("Action Frequency")
    plt.xticks(rotation=0)
    plt.legend(title="Action Taken")
    plt.tight_layout()
    act_path = get_figures_dir() / "temporal_ppo_action_distribution.png"
    plt.savefig(act_path, dpi=300)
    plt.close()
    print(f"Saved PPO Action Distribution plot to: {act_path}")

    # 4. Save Decision Traces CSV
    traces_path = get_reports_dir() / "ppo_decision_traces.csv"
    traces_df.to_csv(traces_path, index=False)
    print(f"Saved PPO Decision Traces to: {traces_path}")

    # 5. Save PPO Config JSON
    config_data = {
        "environment": "TemporalRansomwareEnv",
        "action_space": ["0: CONTINUE_MONITORING", "1: SUSPICIOUS", "2: RANSOMWARE_ALERT"],
        "reward_config": reward_config,
        "n_test_samples": metrics["n_test_samples"],
        "correct_alert_rate": metrics["correct_alert_rate"],
        "false_alert_rate": metrics["false_alert_rate"],
        "miss_rate": metrics["miss_rate"],
        "average_decision_window": metrics["average_decision_window"],
        "average_episode_reward": metrics["average_episode_reward"],
    }
    config_path = get_reports_dir() / "temporal_ppo_config.json"
    save_json(config_data, config_path)
    print(f"Saved Temporal PPO Config to: {config_path}")

    # 6. Save Markdown Analysis Report
    analysis_path = get_reports_dir() / "temporal_ppo_analysis.md"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write("# Temporal Reinforcement Learning (PPO) Analysis\n\n")
        f.write("## 1. Executive Summary\n")
        f.write("This experiment integrates Proximal Policy Optimization (PPO) with sequential multi-window temporal observations ($W_0 \\to W_4$). Unlike static or feature-revealing proxies, the agent observes true progressive telemetry vectors at each discrete observation window.\n\n")
        f.write("## 2. Sequential Decision Metrics\n\n")
        f.write(f"- **Correct Alert Rate (True Positive Alerts)**: {metrics['correct_alert_rate'] * 100:.1f}%\n")
        f.write(f"- **False Alert Rate (False Alarms on Benign)**: {metrics['false_alert_rate'] * 100:.1f}%\n")
        f.write(f"- **Miss Rate (Undetected Ransomware)**: {metrics['miss_rate'] * 100:.1f}%\n")
        f.write(f"- **Average Decision Observation Window**: {metrics['average_decision_window']:.2f}\n")
        f.write(f"- **Average Episode Reward**: {metrics['average_episode_reward']:.2f}\n\n")
        f.write("## 3. Test Trace Evaluation\n\n")
        f.write("| Sample ID | Family | True Class | PPO Final Decision | Decision Window | Episode Reward |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for ep in metrics["episodes_summary"]:
            alert_str = "RANSOMWARE_ALERT" if ep["final_action_alert"] else "BENIGN_CLEARED"
            f.write(f"| {ep['sample_id']} | {ep['family']} | {'Ransomware' if ep['is_ransomware'] else 'Benign'} | {alert_str} | {ep['decision_window']} | {ep['total_reward']:.2f} |\n")
        f.write("\n## 4. Scientific Notice\n")
        f.write("Temporal baseline execution is a pipeline validation experiment using synthetic temporal data, not a real ransomware detection evaluation. No real malware execution was performed.\n")
    print(f"Saved Temporal PPO Analysis to: {analysis_path}")

