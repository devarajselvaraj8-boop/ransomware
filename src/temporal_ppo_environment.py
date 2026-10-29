"""Gymnasium Environment for Temporal Reinforcement Learning Ransomware Detection.

This module provides a sequential decision-making environment (TemporalRansomwareEnv)
operating on true multi-window temporal observations (W0 -> W1 -> W2 -> W3 -> W4).

At each timestep t, the agent observes the system state at window t (behavioral rates,
memory deltas, rolling statistics) strictly without future lookahead.

Action Space:
0 = CONTINUE_MONITORING: Defer decision and observe the next temporal window
1 = SUSPICIOUS: Flag process for elevated scrutiny, proceed to next window
2 = RANSOMWARE_ALERT: Terminate episode immediately and raise quarantine alarm

Reward Function:
Rewards timely true positive detection, penalizes late detection and false alarms,
and accounts for cumulative monitoring costs.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.utils import get_data_dir, load_json, save_json

# Action space definitions
ACTION_CONTINUE = 0
ACTION_SUSPICIOUS = 1
ACTION_ALERT = 2

ACTION_NAMES = {
    ACTION_CONTINUE: "CONTINUE_MONITORING",
    ACTION_SUSPICIOUS: "SUSPICIOUS",
    ACTION_ALERT: "RANSOMWARE_ALERT",
}

DEFAULT_TEMPORAL_REWARD_CONFIG = {
    "alert_true_positive": 10.0,
    "early_detection_bonus_factor": 2.5,
    "alert_false_positive": -10.0,
    "monitoring_cost": -0.1,
    "suspicious_ransomware_reward": 0.5,
    "suspicious_benign_penalty": -0.5,
    "final_missed_ransomware_penalty": -15.0,
    "final_benign_true_negative_reward": 5.0,
}


class TemporalRansomwareEnv(gym.Env):
    """Gymnasium environment simulating sequential process monitoring over temporal observation windows."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_names: List[str],
        split_type: str = "train",
        sample_ids: Optional[List[str]] = None,
        norm_config_path: Optional[Path] = None,
        reward_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = 42,
    ):
        super().__init__()

        self.df = df.copy()
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.split_type = split_type
        self.reward_config = reward_config or DEFAULT_TEMPORAL_REWARD_CONFIG

        # Load normalization parameters (fitted strictly on training set)
        if norm_config_path is None:
            norm_config_path = get_data_dir() / "processed" / "temporal_normalization.json"
        
        if norm_config_path.exists():
            self.norm_config = load_json(norm_config_path)
            self.means = np.array([self.norm_config["mean"].get(f, 0.0) for f in self.feature_names], dtype=np.float32)
            self.scales = np.array([self.norm_config["scale"].get(f, 1.0) for f in self.feature_names], dtype=np.float32)
            # Prevent zero division
            self.scales = np.where(self.scales <= 1e-7, 1.0, self.scales)
        else:
            self.means = np.zeros(self.n_features, dtype=np.float32)
            self.scales = np.ones(self.n_features, dtype=np.float32)

        # Precompute per-sample ordered observation matrices
        self.samples_data: Dict[str, Dict[str, Any]] = {}
        all_unique_samples = self.df["sample_id"].unique()
        
        if sample_ids is not None:
            active_samples = [s for s in sample_ids if s in all_unique_samples]
        else:
            active_samples = list(all_unique_samples)

        self.sample_ids = sorted(active_samples)
        if len(self.sample_ids) == 0:
            raise ValueError(f"No valid samples found for split_type '{split_type}'!")

        for sid in self.sample_ids:
            s_rows = self.df[self.df["sample_id"] == sid].sort_values("observation_window")
            raw_matrix = s_rows[self.feature_names].to_numpy(dtype=np.float32)
            # Apply training-derived normalization
            norm_matrix = (raw_matrix - self.means) / self.scales
            
            lbl = s_rows["label"].iloc[0]
            fam = s_rows["family"].iloc[0]
            is_rw = (lbl.lower() == "ransomware")
            
            self.samples_data[sid] = {
                "observations": norm_matrix,
                "n_windows": len(norm_matrix),
                "label": lbl,
                "family": fam,
                "is_ransomware": is_rw,
            }

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_features,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Internal episode variables
        self.np_random = np.random.default_rng(seed)
        self.current_sample_id: Optional[str] = None
        self.current_step: int = 0
        self.max_steps: int = 0
        self.current_observations: Optional[np.ndarray] = None
        self.is_ransomware: bool = False
        self.current_label: str = ""
        self.current_family: str = ""
        self.episode_history: List[Dict[str, Any]] = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to start a new monitoring episode for a sample."""
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        if options and "sample_id" in options:
            self.current_sample_id = options["sample_id"]
        else:
            self.current_sample_id = str(self.np_random.choice(self.sample_ids))

        s_info = self.samples_data[self.current_sample_id]
        self.current_observations = s_info["observations"]
        self.max_steps = s_info["n_windows"]
        self.is_ransomware = s_info["is_ransomware"]
        self.current_label = s_info["label"]
        self.current_family = s_info["family"]

        self.current_step = 0
        self.episode_history = []

        obs = self.current_observations[self.current_step].copy()
        info = {
            "sample_id": self.current_sample_id,
            "step": self.current_step,
            "true_label": self.current_label,
            "family": self.current_family,
            "is_ransomware": self.is_ransomware,
            "split_type": self.split_type,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one sequential action in the temporal monitoring window."""
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}! Allowed: 0, 1, 2.")

        reward = 0.0
        terminated = False
        truncated = False
        action_name = ACTION_NAMES[action]
        step_idx = self.current_step
        max_idx = self.max_steps - 1

        if action == ACTION_ALERT:
            # Immediate stopping action: Raise alert and terminate
            terminated = True
            if self.is_ransomware:
                # True positive alert with early detection bonus
                bonus = self.reward_config["early_detection_bonus_factor"] * (max_idx - step_idx)
                reward = self.reward_config["alert_true_positive"] + bonus
                outcome = "ALERT_TRUE_POSITIVE"
            else:
                # False alarm penalty on benign process
                reward = self.reward_config["alert_false_positive"]
                outcome = "ALERT_FALSE_POSITIVE"

        elif action == ACTION_SUSPICIOUS:
            # Soft flag action: Continue monitoring under suspicion
            if self.is_ransomware:
                reward = self.reward_config["suspicious_ransomware_reward"]
            else:
                reward = self.reward_config["suspicious_benign_penalty"]

            self.current_step += 1
            if self.current_step >= self.max_steps:
                terminated = True
                if self.is_ransomware:
                    reward += self.reward_config["final_missed_ransomware_penalty"]
                    outcome = "MISSED_RANSOMWARE"
                else:
                    reward += self.reward_config["final_benign_true_negative_reward"]
                    outcome = "BENIGN_CLEARED"
            else:
                outcome = "CONTINUE_SUSPICIOUS"

        elif action == ACTION_CONTINUE:
            # Continue monitoring: Small operational cost to discourage procrastination
            reward = self.reward_config["monitoring_cost"]
            self.current_step += 1
            if self.current_step >= self.max_steps:
                terminated = True
                if self.is_ransomware:
                    reward += self.reward_config["final_missed_ransomware_penalty"]
                    outcome = "MISSED_RANSOMWARE"
                else:
                    reward += self.reward_config["final_benign_true_negative_reward"]
                    outcome = "BENIGN_CLEARED"
            else:
                outcome = "CONTINUE_MONITORING"

        # Record decision step history
        self.episode_history.append({
            "step": step_idx,
            "action": action,
            "action_name": action_name,
            "reward": reward,
            "outcome": outcome,
        })

        if terminated:
            # When terminated, observation remains last observed state
            next_obs = self.current_observations[min(step_idx, max_idx)].copy()
        else:
            next_obs = self.current_observations[self.current_step].copy()

        info = {
            "sample_id": self.current_sample_id,
            "step": self.current_step,
            "action_taken": action,
            "action_name": action_name,
            "outcome": outcome,
            "is_ransomware": self.is_ransomware,
            "family": self.current_family,
            "episode_length": len(self.episode_history),
        }

        return next_obs, reward, terminated, truncated, info

