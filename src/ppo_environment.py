"""Gymnasium-compatible Synthetic Temporal Proxy Environment for PPO Ransomware Detection.

Provides RansomwareDetectionEnv and temporal sequence generation utilities.
Note: The temporal progression in this environment is a synthetic proxy created
by progressive feature reveal, as the underlying dataset lacks execution timestamps.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils import get_data_dir, load_json, save_json

# Action Space Constants
ACTION_CONTINUE = 0
ACTION_SUSPICIOUS = 1
ACTION_ALERT = 2

ACTION_NAMES = {
    ACTION_CONTINUE: "CONTINUE_MONITORING",
    ACTION_SUSPICIOUS: "SUSPICIOUS",
    ACTION_ALERT: "RANSOMWARE_ALERT",
}

# Centralized Default Configuration
DEFAULT_TEMPORAL_CONFIG = {
    "sequence_length": 5,
    "reveal_schedule": [0.2, 0.4, 0.6, 0.8, 1.0],
    "mask_value": 0.0,
}

DEFAULT_REWARD_CONFIG = {
    "alert_true_positive": 10.0,
    "early_detection_bonus_factor": 2.5,
    "alert_false_positive": -10.0,
    "monitoring_cost": -0.1,
    "suspicious_ransomware_reward": 0.5,
    "suspicious_benign_penalty": -0.5,
    "final_missed_ransomware_penalty": -15.0,
    "final_benign_true_negative_reward": 5.0,
}


def fit_and_save_normalization(
    data_path: Optional[Path] = None,
    split_indices_path: Optional[Path] = None,
    feature_list_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Fit StandardScaler strictly on training samples and serialize to disk."""
    if data_path is None:
        data_path = get_data_dir() / "processed" / "ml_ready_dataset.csv"
    if split_indices_path is None:
        split_indices_path = get_data_dir() / "processed" / "split_indices.json"
    if feature_list_path is None:
        feature_list_path = get_data_dir() / "processed" / "early_stage_nonredundant_feature_list.json"
    if output_path is None:
        output_path = get_data_dir() / "processed" / "ppo_normalization.json"

    df = pd.read_csv(data_path)
    split_info = load_json(split_indices_path)
    feature_names = load_json(feature_list_path)

    train_idx = split_info["train_indices"]
    train_df = df.iloc[train_idx]

    scaler = StandardScaler()
    scaler.fit(train_df[feature_names])

    norm_config = {
        "method": "StandardScaler",
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "var": scaler.var_.tolist(),
        "n_samples_fit": int(scaler.n_samples_seen_),
        "random_seed": split_info.get("random_state", 42),
        "source_dataset": str(data_path.name),
        "train_samples_count": len(train_idx),
    }

    save_json(norm_config, output_path)
    print(f"Fit and saved normalization parameters to: {output_path}")
    return norm_config


def generate_temporal_sequence(
    feature_vector: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Transform a 1D normalized feature vector into a synthetic progressive observation sequence.

    Args:
        feature_vector: 1D array of normalized feature values of shape (n_features,).
        config: Temporal configuration dictionary specifying sequence_length and reveal_schedule.

    Returns:
        2D array of shape (sequence_length, n_features) representing progressive observations.
    """
    if config is None:
        config = DEFAULT_TEMPORAL_CONFIG

    seq_len = config["sequence_length"]
    schedule = config["reveal_schedule"]
    mask_val = config.get("mask_value", 0.0)

    n_features = len(feature_vector)
    sequence = np.full((seq_len, n_features), mask_val, dtype=np.float32)

    for step_idx in range(seq_len):
        fraction = schedule[step_idx]
        n_revealed = int(np.ceil(fraction * n_features))
        n_revealed = min(n_revealed, n_features)

        # Progressive reveal: first n_revealed features take actual values
        sequence[step_idx, :n_revealed] = feature_vector[:n_revealed]

    return sequence


class RansomwareDetectionEnv(gym.Env):
    """Gymnasium environment for synthetic temporal early-stage ransomware detection.

    Simulates sequential decision-making over progressive observation steps.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        split_type: str = "train",
        split_indices_path: Optional[Path] = None,
        norm_config_path: Optional[Path] = None,
        temporal_config: Optional[Dict[str, Any]] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()

        # Paths and resources
        if df is None:
            data_path = get_data_dir() / "processed" / "ml_ready_dataset.csv"
            df = pd.read_csv(data_path)
        self.df = df

        if feature_names is None:
            fl_path = get_data_dir() / "processed" / "early_stage_nonredundant_feature_list.json"
            feature_names = load_json(fl_path)
        self.feature_names = feature_names
        self.n_features = len(feature_names)

        if split_indices_path is None:
            split_indices_path = get_data_dir() / "processed" / "split_indices.json"
        self.split_info = load_json(split_indices_path)

        if norm_config_path is None:
            norm_config_path = get_data_dir() / "processed" / "ppo_normalization.json"
            if not norm_config_path.exists():
                fit_and_save_normalization(output_path=norm_config_path)
        self.norm_config = load_json(norm_config_path)

        self.mean = np.array(self.norm_config["mean"], dtype=np.float32)
        self.scale = np.array(self.norm_config["scale"], dtype=np.float32)

        # Partition data by split_type
        if split_type == "train":
            self.sample_indices = self.split_info["train_indices"]
        elif split_type == "test":
            self.sample_indices = self.split_info["test_indices"]
        elif split_type == "all":
            self.sample_indices = list(range(len(self.df)))
        else:
            raise ValueError(f"Unknown split_type: {split_type}. Must be 'train', 'test', or 'all'.")

        self.split_type = split_type
        self.temporal_config = temporal_config or DEFAULT_TEMPORAL_CONFIG
        self.reward_config = reward_config or DEFAULT_REWARD_CONFIG
        self.sequence_length = self.temporal_config["sequence_length"]

        # Gymnasium Spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Internal State Variables
        self.np_random = np.random.default_rng(seed)
        self.current_sample_idx: Optional[int] = None
        self.current_label: Optional[str] = None
        self.is_ransomware: bool = True
        self.current_sequence: Optional[np.ndarray] = None
        self.current_step: int = 0
        self.episode_history: List[Dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and generate a new synthetic episode."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        # Select sample
        if options and "sample_idx" in options:
            self.current_sample_idx = options["sample_idx"]
        else:
            self.current_sample_idx = int(self.np_random.choice(self.sample_indices))

        sample_row = self.df.iloc[self.current_sample_idx]
        self.current_label = sample_row["family"]
        # In current dataset, all classes (G, E, L) are ransomware families
        # Supports synthetic benign flagging if passed in options
        self.is_ransomware = options.get("is_ransomware", True) if options else True

        # Extract & normalize feature vector
        raw_vec = sample_row[self.feature_names].to_numpy(dtype=np.float32)
        norm_vec = (raw_vec - self.mean) / self.scale

        # Generate synthetic progressive temporal observation sequence
        self.current_sequence = generate_temporal_sequence(norm_vec, self.temporal_config)
        self.current_step = 0
        self.episode_history = []

        obs = self.current_sequence[self.current_step].copy()
        info = {
            "sample_idx": self.current_sample_idx,
            "step": self.current_step,
            "true_label": self.current_label,
            "is_ransomware": self.is_ransomware,
            "split_type": self.split_type,
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action, compute reward, and advance or terminate episode."""
        if self.current_sequence is None:
            raise RuntimeError("Environment must be reset before calling step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}! Must be in {list(ACTION_NAMES.keys())}")

        step_idx = self.current_step
        max_step = self.sequence_length - 1
        reward = 0.0
        terminated = False
        truncated = False

        r_cfg = self.reward_config

        # ---------------------------------------------------------
        # Action Evaluation & Reward Economics
        # ---------------------------------------------------------
        if action == ACTION_ALERT:
            terminated = True
            if self.is_ransomware:
                # True Positive Alert with Early Detection Bonus
                early_bonus = (max_step - step_idx) * r_cfg["early_detection_bonus_factor"]
                reward = r_cfg["alert_true_positive"] + early_bonus
                terminal_reason = "CORRECT_EARLY_ALERT" if step_idx < max_step else "CORRECT_ALERT"
            else:
                # False Positive Alert on Benign Process
                reward = r_cfg["alert_false_positive"]
                terminal_reason = "FALSE_ALERT"

        elif action == ACTION_SUSPICIOUS:
            # Intermediate Inspection: does not terminate episode
            if self.is_ransomware:
                reward = r_cfg["suspicious_ransomware_reward"]
            else:
                reward = r_cfg["suspicious_benign_penalty"]

            # Advance step
            if step_idx >= max_step:
                terminated = True
                if self.is_ransomware:
                    reward += r_cfg["final_missed_ransomware_penalty"]
                    terminal_reason = "MISSED_RANSOMWARE"
                else:
                    reward += r_cfg["final_benign_true_negative_reward"]
                    terminal_reason = "BENIGN_COMPLETED"
            else:
                self.current_step += 1
                terminal_reason = "CONTINUED"

        elif action == ACTION_CONTINUE:
            if step_idx >= max_step:
                terminated = True
                if self.is_ransomware:
                    reward = r_cfg["final_missed_ransomware_penalty"]
                    terminal_reason = "MISSED_RANSOMWARE"
                else:
                    reward = r_cfg["final_benign_true_negative_reward"]
                    terminal_reason = "BENIGN_COMPLETED"
            else:
                reward = r_cfg["monitoring_cost"]
                self.current_step += 1
                terminal_reason = "CONTINUED"

        # Current or next observation
        next_obs_step = min(self.current_step, max_step)
        next_obs = self.current_sequence[next_obs_step].copy()

        step_record = {
            "step": step_idx,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": round(reward, 4),
            "terminated": terminated,
            "reason": terminal_reason,
        }
        self.episode_history.append(step_record)

        info = {
            "sample_idx": self.current_sample_idx,
            "step": self.current_step,
            "true_label": self.current_label,
            "is_ransomware": self.is_ransomware,
            "terminal_reason": terminal_reason,
            "episode_length": len(self.episode_history),
        }

        return next_obs, float(reward), terminated, truncated, info

    def render(self, mode: str = "human") -> Optional[str]:
        """Render current episode trajectory summary."""
        lines = [
            f"Episode Sample Index: {self.current_sample_idx} (Family: '{self.current_label}', Step: {self.current_step}/{self.sequence_length-1})",
            "-" * 65,
            f"{'Step':<6}{'Action Taken':<22}{'Reward':<10}{'Terminated':<12}{'Outcome Reason':<20}",
            "-" * 65,
        ]
        for rec in self.episode_history:
            lines.append(
                f"{rec['step']:<6}{rec['action_name']:<22}{rec['reward']:<10.2f}{str(rec['terminated']):<12}{rec['reason']:<20}"
            )
        lines.append("-" * 65)
        text = "\n".join(lines)
        if mode == "human":
            print(text)
            return None
        return text


if __name__ == "__main__":
    print("Testing ppo_environment module...")
    env = RansomwareDetectionEnv(seed=42)
    obs, info = env.reset(seed=42)
    print(f"Reset observation shape: {obs.shape}")
    print(f"Reset info: {info}")
    obs, rew, term, trunc, info = env.step(ACTION_CONTINUE)
    print(f"Step 1: Action=CONTINUE, reward={rew}, terminated={term}")
    obs, rew, term, trunc, info = env.step(ACTION_ALERT)
    print(f"Step 2: Action=ALERT, reward={rew}, terminated={term}")
    env.render()
    print("ppo_environment module sanity test passed successfully!")

