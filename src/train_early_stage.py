"""Early-Stage Feature-Restricted Baseline Training & Evaluation Module.

Executes controlled proxy experiments comparing Full, Early-Stage Candidate,
and Early-Stage Non-Redundant feature sets across Random Forest and XGBoost
using the strictly isolated stratified train/test split.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.utils import (
    get_data_dir,
    get_figures_dir,
    get_reports_dir,
    load_json,
    save_json,
)

# Candidate Early-Stage Features (Plausibly observable prior to mass file encryption)
EARLY_STAGE_CANDIDATE_FEATURES = [
    "apistats",
    "api",
    "dll",
    "dll_loaded",
    "arguments",
    "command_line",
    "beh_command_line",
    "process_path",
    "proc",
    "proc_pid",
    "file",
    "type",
    "name",
    "path",
    "program",
    "info",
    "children",
    "tree_command_line",
    "tree_process_name",
    "regkey_opened",
    "regkey_read",
    "regkey_written",
    "wmi_query",
    "entropy",
]

# Non-Redundant Early-Stage Features (Collapsing identical duplicate columns)
# proc_pid == file (removed file)
# type == name == path (removed name, path)
# program == info (removed info)
# arguments == api (removed api)
# proc == beh_command_line == process_path (removed beh_command_line, process_path)
# tree_command_line == tree_process_name (removed tree_process_name)
EARLY_STAGE_NONREDUNDANT_FEATURES = [
    "apistats",
    "dll",
    "dll_loaded",
    "arguments",
    "command_line",
    "proc",
    "proc_pid",
    "type",
    "program",
    "children",
    "tree_command_line",
    "regkey_opened",
    "regkey_read",
    "regkey_written",
    "wmi_query",
    "entropy",
]


def load_datasets_and_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, List[str]]]:
    """Load ML-ready data, retrieve exact train/test split indices, and define feature sets."""
    data_path = get_data_dir() / "processed" / "ml_ready_dataset.csv"
    indices_path = get_data_dir() / "processed" / "split_indices.json"

    if not data_path.exists():
        raise FileNotFoundError(f"ML-ready dataset not found at {data_path}")
    if not indices_path.exists():
        raise FileNotFoundError(f"Split indices not found at {indices_path}")

    df = pd.read_csv(data_path)
    split_info = load_json(indices_path)

    train_idx = split_info["train_indices"]
    test_idx = split_info["test_indices"]

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    y_train = train_df["family"].copy()
    y_test = test_df["family"].copy()

    full_features = [c for c in df.columns if c != "family"]

    # Verify candidate features exist in full features
    for f in EARLY_STAGE_CANDIDATE_FEATURES:
        if f not in full_features:
            raise ValueError(f"Feature '{f}' not found in dataset columns!")

    for f in EARLY_STAGE_NONREDUNDANT_FEATURES:
        if f not in full_features:
            raise ValueError(f"Feature '{f}' not found in dataset columns!")

    feature_sets = {
        "Full": full_features,
        "Early-stage proxy": EARLY_STAGE_CANDIDATE_FEATURES,
        "Early-stage non-redundant": EARLY_STAGE_NONREDUNDANT_FEATURES,
    }

    # Save feature lists
    save_json(full_features, get_data_dir() / "processed" / "full_feature_list.json")
    save_json(EARLY_STAGE_CANDIDATE_FEATURES, get_data_dir() / "processed" / "early_stage_feature_list.json")
    save_json(EARLY_STAGE_NONREDUNDANT_FEATURES, get_data_dir() / "processed" / "early_stage_nonredundant_feature_list.json")

    print("Loaded datasets and partitions successfully:")
    print(f"  Training samples: {len(train_df)} | Testing samples: {len(test_df)}")
    print(f"  Full feature set (Dataset A): {len(full_features)} features")
    print(f"  Early-stage proxy set (Dataset B): {len(EARLY_STAGE_CANDIDATE_FEATURES)} features")
    print(f"  Early-stage non-redundant set (Dataset C): {len(EARLY_STAGE_NONREDUNDANT_FEATURES)} features")

    return train_df, test_df, y_train, y_test, feature_sets


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, labels: List[str]) -> Dict[str, float]:
    """Compute all required metrics including per-class recall for minority class L."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_w = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_w = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    l_idx = labels.index("L")
    l_recall = float(rec_per_class[l_idx])

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cr = classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision_weighted": round(prec_w, 4),
        "recall_weighted": round(rec_w, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "L_recall": round(l_recall, 4),
        "confusion_matrix": cm,
        "classification_report": cr,
    }


def run_early_stage_experiments() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute the 6 baseline experiments (RF & XGBoost across Datasets A, B, C)."""
    train_df, test_df, y_train, y_test, feature_sets = load_datasets_and_splits()
    labels = sorted(list(y_train.unique()))

    comparison_records = []
    rf_feature_importances = {}
    xgb_feature_importances = {}

    print("\n" + "=" * 70)
    print("RUNNING 6 EARLY-STAGE RESTRICTED BASELINE EXPERIMENTS")
    print("=" * 70)

    # -------------------------------------------------------------
    # 1. Random Forest Experiments
    # -------------------------------------------------------------
    for fset_name, fcols in feature_sets.items():
        print(f"\n[Random Forest] Training on: {fset_name} ({len(fcols)} features)...")
        X_tr = train_df[fcols]
        X_te = test_df[fcols]

        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )
        rf.fit(X_tr, y_train)
        y_pred = rf.predict(X_te)

        res = evaluate_predictions(y_test, y_pred, labels)
        print(f"  Accuracy: {res['accuracy']:.4f} | Balanced Acc: {res['balanced_accuracy']:.4f} | Macro F1: {res['f1_macro']:.4f} | L-Recall: {res['L_recall']:.4f}")

        comparison_records.append({
            "model": "Random Forest",
            "feature_set": fset_name,
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "precision_weighted": res["precision_weighted"],
            "recall_weighted": res["recall_weighted"],
            "f1_weighted": res["f1_weighted"],
            "f1_macro": res["f1_macro"],
            "L_recall": res["L_recall"],
        })

        if fset_name == "Early-stage non-redundant":
            fi_df = pd.DataFrame({
                "feature": fcols,
                "importance": rf.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            rf_feature_importances["nonredundant"] = fi_df

    # -------------------------------------------------------------
    # 2. XGBoost Experiments
    # -------------------------------------------------------------
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    for fset_name, fcols in feature_sets.items():
        print(f"\n[XGBoost] Training on: {fset_name} ({len(fcols)} features)...")
        X_tr = train_df[fcols]
        X_te = test_df[fcols]

        xgb = XGBClassifier(
            n_estimators=200,
            random_state=42,
            eval_metric="mlogloss",
            learning_rate=0.1,
            max_depth=6,
            n_jobs=-1,
            tree_method="hist",
        )
        xgb.fit(X_tr, y_train_enc)
        y_pred_enc = xgb.predict(X_te)
        y_pred = le.inverse_transform(y_pred_enc)

        res = evaluate_predictions(y_test, y_pred, labels)
        print(f"  Accuracy: {res['accuracy']:.4f} | Balanced Acc: {res['balanced_accuracy']:.4f} | Macro F1: {res['f1_macro']:.4f} | L-Recall: {res['L_recall']:.4f}")

        comparison_records.append({
            "model": "XGBoost",
            "feature_set": fset_name,
            "accuracy": res["accuracy"],
            "balanced_accuracy": res["balanced_accuracy"],
            "precision_weighted": res["precision_weighted"],
            "recall_weighted": res["recall_weighted"],
            "f1_weighted": res["f1_weighted"],
            "f1_macro": res["f1_macro"],
            "L_recall": res["L_recall"],
        })

        if fset_name == "Early-stage non-redundant":
            fi_df = pd.DataFrame({
                "feature": fcols,
                "importance": xgb.feature_importances_,
            }).sort_values("importance", ascending=False).reset_index(drop=True)
            xgb_feature_importances["nonredundant"] = fi_df

    # -------------------------------------------------------------
    # Save Comparison CSV
    # -------------------------------------------------------------
    comp_df = pd.DataFrame(comparison_records)
    csv_path = get_reports_dir() / "early_stage_comparison.csv"
    comp_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison metrics to: {csv_path}")

    # -------------------------------------------------------------
    # Save Feature Importance CSVs
    # -------------------------------------------------------------
    rf_fi_df = rf_feature_importances["nonredundant"]
    xgb_fi_df = xgb_feature_importances["nonredundant"]

    rf_fi_csv = get_reports_dir() / "early_stage_rf_feature_importance.csv"
    xgb_fi_csv = get_reports_dir() / "early_stage_xgb_feature_importance.csv"

    rf_fi_df.to_csv(rf_fi_csv, index=False)
    xgb_fi_df.to_csv(xgb_fi_csv, index=False)
    print(f"Saved RF feature importance to: {rf_fi_csv}")
    print(f"Saved XGBoost feature importance to: {xgb_fi_csv}")

    # -------------------------------------------------------------
    # Generate Visualizations
    # -------------------------------------------------------------
    plot_early_stage_comparison(comp_df)
    plot_early_stage_feature_importances(rf_fi_df, xgb_fi_df)

    return comp_df, rf_fi_df, xgb_fi_df


def plot_early_stage_comparison(comp_df: pd.DataFrame) -> None:
    """Generate comparative visualization across the 3 core research metrics."""
    metrics_to_plot = ["f1_macro", "balanced_accuracy", "L_recall"]
    metric_titles = {
        "f1_macro": "Macro F1-Score",
        "balanced_accuracy": "Balanced Accuracy",
        "L_recall": "Minority Class (L) Recall",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        sns.barplot(
            data=comp_df,
            x="feature_set",
            y=metric,
            hue="model",
            palette=["#1f77b4", "#2ca02c"],
            ax=ax,
            edgecolor="black",
        )
        ax.set_title(metric_titles[metric], fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Feature Set", fontsize=10, labelpad=6)
        ax.set_ylabel("Score (0.0 to 1.0)" if i == 0 else "", fontsize=10)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=15)

        # Annotate bars with exact score
        for p in ax.patches:
            h = p.get_height()
            if not np.isnan(h) and h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    xytext=(0, 2),
                    textcoords="offset points",
                )

        if i == 0:
            ax.legend(title="Model", loc="lower right")
        else:
            ax.get_legend().remove()

    plt.suptitle("Early-Stage Feature-Restricted Baseline Comparison", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = get_figures_dir() / "early_stage_metrics_comparison.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved metric comparison figure to: {out_path}")


def plot_early_stage_feature_importances(rf_fi: pd.DataFrame, xgb_fi: pd.DataFrame) -> None:
    """Generate publication-ready horizontal bar charts of early-stage feature importances."""
    # Random Forest Plot
    plt.figure(figsize=(9, 6))
    top_rf = rf_fi.sort_values("importance", ascending=True)
    plt.barh(top_rf["feature"], top_rf["importance"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.title("Random Forest - Early-Stage Feature Importances (Non-Redundant)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Gini Importance Score", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    rf_out = get_figures_dir() / "early_stage_rf_feature_importance.png"
    plt.savefig(rf_out, dpi=300)
    plt.close()
    print(f"Saved RF feature importance plot to: {rf_out}")

    # XGBoost Plot
    plt.figure(figsize=(9, 6))
    top_xgb = xgb_fi.sort_values("importance", ascending=True)
    plt.barh(top_xgb["feature"], top_xgb["importance"], color="#2ca02c", edgecolor="black", alpha=0.85)
    plt.title("XGBoost - Early-Stage Feature Importances (Non-Redundant)", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Split Gain Importance Score", fontsize=10)
    plt.ylabel("Feature Name", fontsize=10)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    xgb_out = get_figures_dir() / "early_stage_xgb_feature_importance.png"
    plt.savefig(xgb_out, dpi=300)
    plt.close()
    print(f"Saved XGBoost feature importance plot to: {xgb_out}")


if __name__ == "__main__":
    run_early_stage_experiments()

