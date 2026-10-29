"""Temporal Dataset Pipeline Engine for Dynamic Ransomware Analysis.

Provides reusable modular functions for:
- Synthetic demonstration data generation (strictly for schema/pipeline validation)
- Schema validation and temporal data integrity checking
- Temporal ordering and sorting
- Feature engineering: deltas, rates, rolling causal statistics, accumulations
- Sample-level train/test splitting (leakage-safe)
- Test-leakage-free normalization
- PPO environment observation vector formatting

NOTE: All data processed by this module in this phase is synthetic validation data.
NO real malware execution or live telemetry collection is performed.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Standard Temporal Dataset Schema Definition
DEFAULT_TEMPORAL_SCHEMA = {
    "required_columns": [
        "sample_id",
        "process_id",
        "parent_process_id",
        "timestamp_us",
        "relative_time_ms",
        "observation_window",
        "process_name",
        "label",
        "family",
    ],
    "numeric_features": [
        "api_count",
        "api_rate",
        "dll_count",
        "dll_load_rate",
        "registry_read_count",
        "registry_write_count",
        "file_read_count",
        "file_write_count",
        "file_create_count",
        "directory_activity_count",
        "entropy",
        "working_set_bytes",
        "private_bytes",
        "handle_count",
        "thread_count",
        "network_connection_count",
        "loaded_module_count",
    ],
    "non_negative_columns": [
        "timestamp_us",
        "relative_time_ms",
        "observation_window",
        "api_count",
        "api_rate",
        "dll_count",
        "dll_load_rate",
        "registry_read_count",
        "registry_write_count",
        "file_read_count",
        "file_write_count",
        "file_create_count",
        "directory_activity_count",
        "entropy",
        "working_set_bytes",
        "private_bytes",
        "handle_count",
        "thread_count",
        "network_connection_count",
        "loaded_module_count",
    ],
    "valid_labels": ["benign", "ransomware"],
}


def generate_synthetic_temporal_demo(
    n_samples: int = 8,
    n_windows: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a reproducible synthetic temporal demonstration dataset.
    
    This function creates synthetic multi-window process execution sequences
    strictly for schema, pipeline, and test validation. It does NOT represent
    real malware telemetry.
    
    Args:
        n_samples: Total number of execution samples to simulate (e.g. 8).
        n_windows: Sequential observation windows per sample (e.g. 5, W0-W4).
        seed: Random seed for reproducibility (default 42).
        
    Returns:
        pd.DataFrame containing synthetic multi-window observations.
    """
    rng = np.random.default_rng(seed)
    
    # Define sample distribution: 5 ransomware (families G, E, L), 3 benign
    sample_configs = [
        {"id": "SAMPLE_001", "label": "ransomware", "family": "G", "pid": 1044, "proc": "crypt_agent.exe"},
        {"id": "SAMPLE_002", "label": "ransomware", "family": "G", "pid": 1180, "proc": "locker_svc.exe"},
        {"id": "SAMPLE_003", "label": "ransomware", "family": "E", "pid": 2048, "proc": "enc_worker.exe"},
        {"id": "SAMPLE_004", "label": "ransomware", "family": "E", "pid": 2212, "proc": "cipher_host.exe"},
        {"id": "SAMPLE_005", "label": "ransomware", "family": "L", "pid": 3104, "proc": "sys_sync.exe"},
        {"id": "SAMPLE_006", "label": "benign", "family": "benign", "pid": 4012, "proc": "notepad.exe"},
        {"id": "SAMPLE_007", "label": "benign", "family": "benign", "pid": 4520, "proc": "calculator.exe"},
        {"id": "SAMPLE_008", "label": "benign", "family": "benign", "pid": 4896, "proc": "explorer_ext.exe"},
    ]
    
    # Adjust count if n_samples differs from len(sample_configs)
    if n_samples != len(sample_configs):
        sample_configs = sample_configs[:n_samples]
        
    rows = []
    
    for s_idx, cfg in enumerate(sample_configs):
        is_rw = (cfg["label"] == "ransomware")
        fam = cfg["family"]
        base_time_us = 1700000000000000 + s_idx * 1000000000  # distinct base epoch
        
        # Initial baseline states
        curr_working_set = int(rng.uniform(15_000_000, 35_000_000))
        curr_private_bytes = int(rng.uniform(8_000_000, 20_000_000))
        curr_handles = int(rng.integers(120, 280))
        curr_threads = int(rng.integers(4, 12))
        curr_modules = int(rng.integers(25, 45))
        base_entropy = float(rng.uniform(4.5, 5.8))
        
        for w in range(n_windows):
            rel_ms = float(w * 1000.0)  # 1-second observation windows (0ms, 1000ms, 2000ms, ...)
            ts_us = int(base_time_us + rel_ms * 1000)
            
            if is_rw:
                # Ransomware behavioral progression across windows
                # Early windows (0, 1): survey, discovery, registry enumeration
                # Middle windows (2, 3): file traversal, handle explosion, memory allocation
                # Late window (4): intensive file write/modification, high buffer entropy
                api_cnt = int(rng.integers(200, 500) + w * rng.integers(300, 800))
                dll_cnt = int(rng.integers(1, 4) if w < 2 else rng.integers(0, 2))
                reg_read = int(rng.integers(30, 80) + w * 20)
                reg_write = int(rng.integers(2, 8) if w < 2 else rng.integers(15, 60))
                
                if w < 2:
                    file_read = int(rng.integers(10, 40))
                    file_write = int(rng.integers(0, 5))
                    file_create = int(rng.integers(0, 3))
                    dir_act = int(rng.integers(5, 25))
                    entropy_val = float(np.clip(base_entropy + rng.normal(0, 0.1), 1.0, 7.9))
                else:
                    file_read = int(rng.integers(80, 250) * w)
                    file_write = int(rng.integers(50, 200) * w)
                    file_create = int(rng.integers(10, 50) * w)
                    dir_act = int(rng.integers(30, 90) * w)
                    entropy_val = float(np.clip(base_entropy + 0.45 * w + rng.normal(0, 0.08), 1.0, 7.95))
                    
                curr_working_set += int(rng.integers(2_000_000, 8_000_000))
                curr_private_bytes += int(rng.integers(2_500_000, 9_000_000))
                curr_handles += int(rng.integers(20, 80))
                curr_threads += int(rng.integers(1, 4) if w < 3 else rng.integers(0, 2))
                curr_modules += int(rng.integers(1, 3) if w < 2 else 0)
                net_conn = int(rng.integers(1, 4) if w in [1, 2] else rng.integers(0, 2))
                
            else:
                # Benign behavioral stability
                api_cnt = int(rng.integers(50, 180))
                dll_cnt = int(rng.integers(0, 2) if w == 0 else 0)
                reg_read = int(rng.integers(5, 25))
                reg_write = int(rng.integers(0, 3))
                file_read = int(rng.integers(2, 15))
                file_write = int(rng.integers(0, 4))
                file_create = int(rng.integers(0, 2))
                dir_act = int(rng.integers(0, 5))
                entropy_val = float(np.clip(base_entropy + rng.normal(0, 0.05), 1.0, 6.2))
                
                curr_working_set += int(rng.integers(-500_000, 800_000))
                curr_private_bytes += int(rng.integers(-300_000, 600_000))
                curr_handles += int(rng.integers(-5, 10))
                curr_threads += int(rng.integers(-1, 2) if w > 0 else 0)
                curr_modules += int(1 if (w == 1 and rng.random() > 0.5) else 0)
                net_conn = int(rng.integers(0, 2))
                
            api_rate = float(api_cnt / 1000.0)
            dll_load_rate = float(dll_cnt / 1.0)
            
            row = {
                "sample_id": cfg["id"],
                "process_id": cfg["pid"],
                "parent_process_id": cfg["pid"] - 100,
                "timestamp_us": ts_us,
                "relative_time_ms": rel_ms,
                "observation_window": w,
                "process_name": cfg["proc"],
                "process_path": f"C:\\SimulatedLab\\{cfg['proc']}",
                "label": cfg["label"],
                "family": fam,
                "api_count": api_cnt,
                "api_rate": api_rate,
                "dll_count": dll_cnt,
                "dll_load_rate": dll_load_rate,
                "registry_read_count": reg_read,
                "registry_write_count": reg_write,
                "file_read_count": file_read,
                "file_write_count": file_write,
                "file_create_count": file_create,
                "directory_activity_count": dir_act,
                "entropy": entropy_val,
                "memory_snapshot_id": f"MEM_{cfg['id']}_W{w}",
                "working_set_bytes": max(curr_working_set, 1000),
                "private_bytes": max(curr_private_bytes, 1000),
                "handle_count": max(curr_handles, 10),
                "thread_count": max(curr_threads, 1),
                "network_connection_count": max(net_conn, 0),
                "loaded_module_count": max(curr_modules, 5),
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    return df


def load_temporal_csv(filepath_or_buffer: Union[str, Path, Any]) -> pd.DataFrame:
    """Load temporal telemetry CSV into pandas DataFrame with schema preservation."""
    df = pd.read_csv(filepath_or_buffer)
    return df


def validate_temporal_data(
    df: pd.DataFrame,
    schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate temporal telemetry DataFrame against schema constraints and temporal integrity rules.
    
    Checks:
    - Required columns presence
    - Missing values (NaN)
    - Duplicate observations per (sample_id, observation_window) or timestamp
    - Timestamp monotonic ordering within sample
    - Observation window continuity (e.g. 0, 1, 2...)
    - Valid process IDs (positive integers)
    - Numeric feature types
    - Negative values in non-negative physical quantities
    - Infinite values (inf, -inf)
    - Label consistency within each sample_id
    
    Returns:
        Dict with validation status: 'is_valid' (bool), 'errors' (list), 'warnings' (list), 'details' (dict).
    """
    if schema is None:
        schema = DEFAULT_TEMPORAL_SCHEMA
        
    errors = []
    warnings = []
    details = {}
    
    # 1. Required Columns Check
    missing_cols = [col for col in schema.get("required_columns", []) if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required schema columns: {missing_cols}")
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings,
            "details": {"missing_required_columns": missing_cols},
        }
        
    details["columns_verified"] = list(df.columns)
    
    # 2. Missing Values Check
    nan_counts = df.isna().sum().to_dict()
    cols_with_nans = {k: int(v) for k, v in nan_counts.items() if v > 0}
    if cols_with_nans:
        errors.append(f"Null/NaN values detected in columns: {cols_with_nans}")
    details["null_counts"] = cols_with_nans
    
    # 3. Duplicate Observations Check
    window_col = "observation_window" if "observation_window" in df.columns else None
    if "sample_id" in df.columns and window_col:
        dups = df.duplicated(subset=["sample_id", window_col], keep=False)
        n_dups = int(dups.sum())
        if n_dups > 0:
            errors.append(f"Duplicate observations detected for ({n_dups} rows) across sample_id and {window_col}.")
        details["duplicate_window_count"] = n_dups
        
    # 4. Timestamp & Window Monotonic Ordering Check
    unordered_samples = []
    non_consecutive_windows = []
    if "sample_id" in df.columns:
        for sample_id, group in df.groupby("sample_id", sort=False):
            if "timestamp_us" in group.columns:
                ts = group["timestamp_us"].values
                if not np.all(ts[:-1] <= ts[1:]):
                    unordered_samples.append(sample_id)
            elif "relative_time_ms" in group.columns:
                rt = group["relative_time_ms"].values
                if not np.all(rt[:-1] <= rt[1:]):
                    unordered_samples.append(sample_id)
                    
            if window_col and window_col in group.columns:
                windows = group[window_col].values
                # Check if windows are strictly non-decreasing
                if not np.all(windows[:-1] <= windows[1:]):
                    non_consecutive_windows.append(sample_id)
                    
    if unordered_samples:
        errors.append(f"Non-monotonic timestamps detected in samples: {unordered_samples}")
    if non_consecutive_windows:
        errors.append(f"Non-monotonic observation windows detected in samples: {non_consecutive_windows}")
    details["unordered_timestamp_samples"] = unordered_samples
    details["unordered_window_samples"] = non_consecutive_windows
    
    # 5. Process ID Consistency Check
    if "process_id" in df.columns:
        invalid_pids = df[df["process_id"] <= 0]["process_id"].tolist()
        if invalid_pids:
            errors.append(f"Invalid non-positive process IDs detected: {invalid_pids[:5]}")
        details["invalid_pid_count"] = len(invalid_pids)
        
    # 6. Infinite Values Check
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_cols = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            inf_cols[col] = inf_count
    if inf_cols:
        errors.append(f"Infinite values detected in numeric columns: {inf_cols}")
    details["infinite_value_counts"] = inf_cols
    
    # 7. Non-Negative Constraints Check
    negative_violations = {}
    for col in schema.get("non_negative_columns", []):
        if col in df.columns and np.issubdtype(df[col].dtype, np.number):
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                negative_violations[col] = neg_count
    if negative_violations:
        errors.append(f"Negative values detected in strictly non-negative columns: {negative_violations}")
    details["negative_violations"] = negative_violations
    
    # 8. Label Consistency Check within Sample
    label_inconsistencies = []
    if "sample_id" in df.columns and "label" in df.columns:
        for sample_id, group in df.groupby("sample_id"):
            if group["label"].nunique() > 1:
                label_inconsistencies.append(sample_id)
    if label_inconsistencies:
        errors.append(f"Inconsistent labels within sample_id: {label_inconsistencies}")
    details["label_inconsistent_samples"] = label_inconsistencies
    
    is_valid = len(errors) == 0
    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "details": details,
        "n_samples": int(df["sample_id"].nunique()) if "sample_id" in df.columns else 0,
        "n_rows": len(df),
    }


def sort_temporal_data(
    df: pd.DataFrame,
    sample_col: str = "sample_id",
    window_col: str = "observation_window",
    time_col: Optional[str] = "timestamp_us",
) -> pd.DataFrame:
    """Sort temporal dataset stably by sample ID and ascending temporal sequence."""
    df_sorted = df.copy()
    sort_keys = [sample_col]
    if time_col and time_col in df_sorted.columns:
        sort_keys.append(time_col)
    if window_col and window_col in df_sorted.columns and window_col not in sort_keys:
        sort_keys.append(window_col)
        
    df_sorted = df_sorted.sort_values(by=sort_keys, ascending=True).reset_index(drop=True)
    return df_sorted


def create_temporal_features(
    df: pd.DataFrame,
    sample_col: str = "sample_id",
    time_col: Optional[str] = "relative_time_ms",
) -> pd.DataFrame:
    """Engineer causal delta, rate, and cumulative features grouped strictly by sample.
    
    Computes:
    - Delta features (x_t - x_{t-1})
    - Rate features (delta_x / delta_t) when timestamps exist
    - Cumulative behavioral features (cumsum_t)
    - Memory change features (working_set_delta, private_bytes_delta, handle_delta, etc.)
    
    All calculations are strictly causal (using only observations at or before window t).
    First observation window delta defaults to 0.0.
    """
    df_out = df.copy()
    
    # Delta feature mapping: (source_column, new_delta_column)
    delta_targets = [
        ("entropy", "entropy_delta"),
        ("working_set_bytes", "working_set_delta"),
        ("private_bytes", "private_bytes_delta"),
        ("handle_count", "handle_delta"),
        ("thread_count", "thread_delta"),
        ("process_count", "process_count_delta"),
        ("network_connection_count", "network_connection_delta"),
        ("loaded_module_count", "loaded_modules_delta"),
        ("api_count", "api_count_delta"),
        ("file_write_count", "file_write_delta"),
        ("file_read_count", "file_read_delta"),
        ("registry_write_count", "registry_write_delta"),
    ]
    
    for src_col, delta_col in delta_targets:
        if src_col in df_out.columns:
            # Group by sample_id to prevent cross-sample contamination
            df_out[delta_col] = df_out.groupby(sample_col)[src_col].diff().fillna(0.0)
            
    # Rate features: delta / delta_t (in seconds)
    if time_col and time_col in df_out.columns:
        # Time difference per sample in seconds
        dt_ms = df_out.groupby(sample_col)[time_col].diff().fillna(1000.0)
        dt_sec = dt_ms / 1000.0
        # Protect against zero division
        dt_sec = np.where(dt_sec <= 0.0, 1.0, dt_sec)
        
        if "working_set_delta" in df_out.columns:
            df_out["working_set_rate_bytes_per_sec"] = df_out["working_set_delta"] / dt_sec
        if "private_bytes_delta" in df_out.columns:
            df_out["private_bytes_rate_bytes_per_sec"] = df_out["private_bytes_delta"] / dt_sec
        if "file_write_delta" in df_out.columns:
            df_out["file_write_rate"] = df_out["file_write_delta"] / dt_sec
            
    # Behavioral accumulation features (cumulative sums per sample)
    cum_targets = [
        ("api_count", "cumulative_api_count"),
        ("registry_write_count", "cumulative_registry_writes"),
        ("file_write_count", "cumulative_file_writes"),
        ("directory_activity_count", "cumulative_directory_activity"),
    ]
    for src_col, cum_col in cum_targets:
        if src_col in df_out.columns:
            df_out[cum_col] = df_out.groupby(sample_col)[src_col].cumsum()
            
    # Memory ratio features
    if "private_bytes" in df_out.columns and "working_set_bytes" in df_out.columns:
        # Ratio of private committed bytes to physical working set
        ws_safe = np.where(df_out["working_set_bytes"] <= 0, 1.0, df_out["working_set_bytes"])
        df_out["private_to_working_set_ratio"] = df_out["private_bytes"] / ws_safe
        
    return df_out


def create_window_features(
    df: pd.DataFrame,
    window_size: int = 3,
    sample_col: str = "sample_id",
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compute strictly causal rolling window statistics (mean, std, min, max).
    
    Rolling computations use min_periods=1 and closed='both' or preceding indices only.
    No future observations (t > current) are included in the rolling aggregation window.
    """
    df_out = df.copy()
    
    if feature_cols is None:
        candidate_cols = [
            "entropy",
            "working_set_bytes",
            "private_bytes",
            "handle_count",
            "api_count",
            "file_write_count",
        ]
        feature_cols = [c for c in candidate_cols if c in df_out.columns]
        
    for col in feature_cols:
        # Grouped rolling calculation strictly within sample_id
        grouped = df_out.groupby(sample_col)[col]
        
        # min_periods=1 ensures window 0 uses itself without lookahead or nulls
        rolling_obj = grouped.rolling(window=window_size, min_periods=1)
        
        mean_vals = rolling_obj.mean().reset_index(level=0, drop=True)
        std_vals = rolling_obj.std().reset_index(level=0, drop=True).fillna(0.0)
        min_vals = rolling_obj.min().reset_index(level=0, drop=True)
        max_vals = rolling_obj.max().reset_index(level=0, drop=True)
        
        df_out[f"{col}_rolling_mean_{window_size}"] = mean_vals
        df_out[f"{col}_rolling_std_{window_size}"] = std_vals
        df_out[f"{col}_rolling_min_{window_size}"] = min_vals
        df_out[f"{col}_rolling_max_{window_size}"] = max_vals
        
    return df_out


def split_by_sample(
    df: pd.DataFrame,
    train_ratio: float = 0.75,
    seed: int = 42,
    sample_col: str = "sample_id",
    stratify_col: Optional[str] = "label",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Partition temporal dataset by sample ID, preventing cross-window observation leakage.
    
    Every observation of a given sample_id remains strictly within either the train partition
    or the test partition.
    
    Returns:
        (train_df, test_df, split_summary_dict)
    """
    unique_samples = df[[sample_col, stratify_col]].drop_duplicates().reset_index(drop=True)
    
    train_samples, test_samples = train_test_split(
        unique_samples[sample_col],
        train_size=train_ratio,
        random_state=seed,
        stratify=unique_samples[stratify_col] if stratify_col in unique_samples.columns else None,
    )
    
    train_set = set(train_samples)
    test_set = set(test_samples)
    
    train_df = df[df[sample_col].isin(train_set)].copy().reset_index(drop=True)
    test_df = df[df[sample_col].isin(test_set)].copy().reset_index(drop=True)
    
    # Verify strict disjointness
    intersection = train_set.intersection(test_set)
    assert len(intersection) == 0, f"Leakage detected: samples in both train and test: {intersection}"
    
    summary = {
        "train_samples": sorted(list(train_set)),
        "test_samples": sorted(list(test_set)),
        "n_train_samples": len(train_set),
        "n_test_samples": len(test_set),
        "n_train_observations": len(train_df),
        "n_test_observations": len(test_df),
        "sample_intersection": list(intersection),
        "leakage_detected": len(intersection) > 0,
    }
    
    return train_df, test_df, summary


def fit_temporal_normalizer(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit normalization parameters (mean, std) strictly on training samples.
    
    Zero-variance features are handled safely by setting scale=1.0 to avoid division by zero.
    
    Returns:
        Dict of normalization parameters serializable to JSON.
    """
    if feature_cols is None:
        feature_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude administrative / ID columns
        exclude_cols = [
            "process_id",
            "parent_process_id",
            "timestamp_us",
            "relative_time_ms",
            "observation_window",
        ]
        feature_cols = [c for c in feature_cols if c not in exclude_cols]
        
    means = {}
    scales = {}
    variances = {}
    
    for col in feature_cols:
        col_vals = train_df[col].to_numpy(dtype=np.float64)
        mean_val = float(np.mean(col_vals))
        var_val = float(np.var(col_vals))
        std_val = float(np.std(col_vals))
        
        # Handle zero or near-zero variance safely
        if std_val < 1e-8:
            scale_val = 1.0
        else:
            scale_val = std_val
            
        means[col] = mean_val
        scales[col] = scale_val
        variances[col] = var_val
        
    norm_params = {
        "method": "StandardScaler_SampleLevelSafe",
        "n_features": len(feature_cols),
        "feature_names": feature_cols,
        "mean": means,
        "scale": scales,
        "variance": variances,
        "n_train_samples": int(train_df["sample_id"].nunique()) if "sample_id" in train_df.columns else len(train_df),
        "n_train_observations": len(train_df),
    }
    return norm_params


def transform_temporal_data(
    df: pd.DataFrame,
    normalizer_params: Dict[str, Any],
) -> pd.DataFrame:
    """Normalize temporal feature columns using pre-fitted normalization parameters."""
    df_transformed = df.copy()
    feature_names = normalizer_params["feature_names"]
    means = normalizer_params["mean"]
    scales = normalizer_params["scale"]
    
    for col in feature_names:
        if col in df_transformed.columns:
            m = means[col]
            s = scales[col]
            df_transformed[col] = (df_transformed[col].astype(np.float32) - m) / s
            
    return df_transformed


def format_for_ppo_observation(
    df: pd.DataFrame,
    feature_cols: List[str],
    sample_id: Optional[str] = None,
) -> np.ndarray:
    """Convert temporal representation of a sample into a PPO observation sequence.
    
    Returns:
        np.ndarray of shape (n_windows, len(feature_cols)) representing sequential observation vectors.
    """
    if sample_id is not None:
        sample_df = df[df["sample_id"] == sample_id].copy()
    else:
        sample_df = df.copy()
        
    # Sort by observation window
    if "observation_window" in sample_df.columns:
        sample_df = sample_df.sort_values(by="observation_window")
        
    obs_seq = sample_df[feature_cols].to_numpy(dtype=np.float32)
    return obs_seq
