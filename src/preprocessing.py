"""Data validation, preprocessing, and reproducible partitioning module.

Validates the cleaned ransomware dataset against strict experimental criteria,
formats numeric feature spaces, and partitions data into stratified 80/20 train/test
splits with zero data leakage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import get_data_dir, save_json

EXPECTED_ROWS = 1750
EXPECTED_COLS = 47
EXPECTED_FEATURES_COUNT = 46
EXPECTED_TARGET = "family"
EXPECTED_CLASSES = {"G": 820, "E": 767, "L": 163}


def validate_dataset(filepath: Optional[Path] = None) -> Dict[str, any]:
    """Validate dataset structure, integrity, distributions, and invariants.

    Raises RuntimeError if structural expectations are violated.
    Returns a comprehensive validation report dictionary.
    """
    if filepath is None:
        filepath = get_data_dir() / "processed" / "cleaned_dataset.csv"

    if not filepath.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    report: Dict[str, any] = {}

    # 1. Shape validation
    actual_shape = df.shape
    report["shape"] = {"rows": actual_shape[0], "columns": actual_shape[1]}
    if actual_shape != (EXPECTED_ROWS, EXPECTED_COLS):
        raise RuntimeError(
            f"Shape validation failed! Expected ({EXPECTED_ROWS}, {EXPECTED_COLS}), "
            f"but found {actual_shape}. Halting execution per research safety guidelines."
        )

    # 2. Target validation
    if EXPECTED_TARGET not in df.columns:
        raise RuntimeError(
            f"Target column '{EXPECTED_TARGET}' missing from dataset! Available columns: {list(df.columns)}"
        )

    # 3. Missing and Infinite values
    null_counts = int(df.isnull().sum().sum())
    report["missing_values"] = null_counts
    if null_counts > 0:
        raise RuntimeError(f"Unexpected missing values detected ({null_counts} nulls).")

    numeric_cols = [c for c in df.columns if c != EXPECTED_TARGET]
    inf_counts = int(np.isinf(df[numeric_cols].to_numpy()).sum())
    report["infinite_values"] = inf_counts
    if inf_counts > 0:
        raise RuntimeError(f"Unexpected infinite values detected ({inf_counts} infs).")

    # 4. Duplicate rows
    dup_counts = int(df.duplicated().sum())
    report["duplicates"] = dup_counts
    if dup_counts != 0:
        raise RuntimeError(f"Expected 0 duplicates in cleaned dataset, but found {dup_counts}.")

    # 5. Constant features
    constant_features = [col for col in numeric_cols if df[col].nunique() <= 1]
    report["constant_features"] = constant_features
    if len(constant_features) > 0:
        raise RuntimeError(f"Found unexpected constant features in cleaned dataset: {constant_features}")

    # 6. Class distribution
    class_dist = df[EXPECTED_TARGET].value_counts().to_dict()
    report["class_distribution"] = class_dist
    if class_dist != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Class distribution mismatch! Expected {EXPECTED_CLASSES}, but found {class_dist}."
        )

    # 7. Unique values and feature ranges
    unique_vals = {col: int(df[col].nunique()) for col in df.columns}
    ranges = {col: {"min": float(df[col].min()), "max": float(df[col].max())} for col in numeric_cols}
    report["unique_values"] = unique_vals
    report["feature_ranges"] = ranges

    # 8. Detect perfectly identical feature columns (redundancies)
    identical_pairs = []
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            c1 = numeric_cols[i]
            c2 = numeric_cols[j]
            if (df[c1] == df[c2]).all():
                identical_pairs.append((c1, c2))
    report["identical_feature_pairs"] = identical_pairs

    # 9. Potentially suspicious identifier/metadata features
    suspicious_features = []
    for col in numeric_cols:
        # High cardinality or metadata indicators
        if unique_vals[col] > 300 or col in ["log", "positives", "proc_pid"]:
            suspicious_features.append(col)
    report["suspicious_features"] = suspicious_features

    # Print clean validation report
    print("=" * 60)
    print("DATASET VALIDATION REPORT")
    print("=" * 60)
    print(f"File Source          : {filepath.name}")
    print(f"Dataset Shape        : {actual_shape[0]} rows x {actual_shape[1]} columns")
    print(f"Target Column        : '{EXPECTED_TARGET}'")
    print(f"Feature Columns      : {len(numeric_cols)}")
    print(f"Missing Values       : {null_counts}")
    print(f"Infinite Values      : {inf_counts}")
    print(f"Duplicate Rows       : {dup_counts}")
    print(f"Constant Features    : {len(constant_features)}")
    print(f"Class Distribution   : {class_dist}")
    for cls_name, cnt in class_dist.items():
        pct = (cnt / actual_shape[0]) * 100
        print(f"  - Class '{cls_name}': {cnt:4d} ({pct:5.2f}%)")
    print(f"Identical Col Pairs  : {len(identical_pairs)} pairs identified")
    for p1, p2 in identical_pairs:
        print(f"  * {p1} == {p2}")
    print("=" * 60)
    print("Validation passed successfully! All integrity invariants verified.\n")

    return report


def prepare_ml_dataset(
    input_path: Optional[Path] = None,
    output_csv_path: Optional[Path] = None,
    output_feature_list_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Load cleaned data, enforce strict numeric types, save ml_ready dataset and feature list."""
    if input_path is None:
        input_path = get_data_dir() / "processed" / "cleaned_dataset.csv"
    if output_csv_path is None:
        output_csv_path = get_data_dir() / "processed" / "ml_ready_dataset.csv"
    if output_feature_list_path is None:
        output_feature_list_path = get_data_dir() / "processed" / "feature_list.json"

    # Validate first
    validate_dataset(input_path)

    df = pd.read_csv(input_path)
    y = df[EXPECTED_TARGET].copy()
    X = df.drop(columns=[EXPECTED_TARGET]).copy()

    # Ensure all feature columns are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="raise")

    feature_list = list(X.columns)

    # Save ml_ready_dataset.csv
    ml_ready_df = pd.concat([y, X], axis=1)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    ml_ready_df.to_csv(output_csv_path, index=False)

    # Save feature_list.json
    save_json(feature_list, output_feature_list_path)

    print(f"Saved ML-ready dataset to: {output_csv_path} (Shape: {ml_ready_df.shape})")
    print(f"Saved feature list ({len(feature_list)} features) to: {output_feature_list_path}")

    return X, y, feature_list


def get_train_test_split(
    test_size: float = 0.20,
    random_state: int = 42,
    save_indices: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Produce reproducible, stratified 80/20 train/test split.

    Guarantees that test data is strictly isolated. Saves split indices to disk.
    """
    X, y, feature_list = prepare_ml_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    if save_indices:
        indices_path = get_data_dir() / "processed" / "split_indices.json"
        split_indices = {
            "random_state": random_state,
            "test_size": test_size,
            "train_indices": [int(idx) for idx in X_train.index],
            "test_indices": [int(idx) for idx in X_test.index],
            "train_size": len(X_train),
            "test_size_count": len(X_test),
        }
        save_json(split_indices, indices_path)
        print(f"Saved split indices to: {indices_path}")

    print("-" * 50)
    print("TRAIN / TEST SPLIT SUMMARY")
    print("-" * 50)
    print(f"Total samples: {len(X)}")
    print(f"Training samples: {len(X_train)} ({len(X_train)/len(X):.1%})")
    print(f"Testing samples : {len(X_test)} ({len(X_test)/len(X):.1%})")
    print("\nTraining Class Distribution:")
    for cls_name, count in y_train.value_counts().items():
        print(f"  Class {cls_name}: {count:4d} ({count/len(y_train):.2%})")
    print("\nTesting Class Distribution:")
    for cls_name, count in y_test.value_counts().items():
        print(f"  Class {cls_name}: {count:4d} ({count/len(y_test):.2%})")
    print("-" * 50 + "\n")

    return X_train, X_test, y_train, y_test, feature_list


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset validation and preprocessing")
    parser.add_argument("--validate", action="store_true", help="Run dataset validation")
    parser.add_argument("--prepare", action="store_true", help="Prepare ML dataset and splits")
    args = parser.parse_args()

    if args.validate:
        validate_dataset()
    elif args.prepare:
        get_train_test_split()
    else:
        # Default: run full preparation pipeline
        validate_dataset()
        get_train_test_split()

