"""Utility functions for paths, metric computations, and scientific plotting.

Provides reproducible helper functions for the early-stage ransomware detection pipeline.
All filesystem paths are managed dynamically using pathlib.Path.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return path to data directory."""
    return get_project_root() / "data"


def get_models_dir() -> Path:
    """Return path to models directory."""
    models_dir = get_project_root() / "models" / "baseline"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_results_dir() -> Path:
    """Return path to results directory."""
    return get_project_root() / "results"


def get_figures_dir() -> Path:
    """Return path to results/figures directory."""
    figures_dir = get_results_dir() / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def get_reports_dir() -> Path:
    """Return path to results/reports directory."""
    reports_dir = get_results_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def get_baseline_results_dir() -> Path:
    """Return path to results/baseline directory."""
    baseline_dir = get_results_dir() / "baseline"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    return baseline_dir


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series, list],
    y_pred: Union[np.ndarray, pd.Series, list],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive multiclass classification metrics.

    Calculates overall and per-class metrics:
    - Accuracy and Balanced Accuracy
    - Precision (Weighted and Macro)
    - Recall (Weighted and Macro)
    - F1 Score (Weighted and Macro)
    - Per-class Precision, Recall, F1, and Support
    - Confusion Matrix (formatted as list for JSON serialization)
    - Formatted Classification Report string
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    if labels is None:
        labels = sorted(list(set(y_true_arr) | set(y_pred_arr)))

    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

    prec_weighted = float(precision_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    rec_weighted = float(recall_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    f1_weighted = float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

    prec_per_class = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    rec_per_class = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    f1_per_class = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    report_str = classification_report(y_true_arr, y_pred_arr, labels=labels, digits=4, zero_division=0)

    per_class_dict = {}
    for i, label in enumerate(labels):
        support_val = int((y_true_arr == label).sum())
        per_class_dict[str(label)] = {
            "precision": float(prec_per_class[i]),
            "recall": float(rec_per_class[i]),
            "f1_score": float(f1_per_class[i]),
            "support": support_val,
        }

    metrics = {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(prec_weighted, 4),
        "recall": round(rec_weighted, 4),
        "f1_score": round(f1_weighted, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "classes": [str(lbl) for lbl in labels],
        "classification_report": report_str,
    }

    return metrics


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Generate and save a publication-quality confusion matrix heatmap."""
    cm_arr = np.array(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_arr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=10, labelpad=8)
    plt.ylabel("True Class", fontsize=10, labelpad=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str = "Top 20 Feature Importances",
    top_n: int = 20,
) -> None:
    """Plot and save horizontal bar chart of top N features."""
    top_df = importance_df.head(top_n).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 7))
    plt.barh(top_df["feature"], top_df["importance"], color="#1f77b4", edgecolor="black", alpha=0.85)
    plt.title(title, fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance Score", fontsize=10, labelpad=8)
    plt.ylabel("Feature Name", fontsize=10, labelpad=8)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path,
    title: str = "Baseline Model Performance Comparison",
) -> None:
    """Plot grouped bar chart comparing Random Forest and XGBoost."""
    metric_cols = ["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "precision", "recall"]
    plot_data = pd.melt(
        comparison_df,
        id_vars=["model"],
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score",
    )
    # Clean metric names for display
    plot_data["Metric"] = plot_data["Metric"].str.replace("_", " ").str.title()

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=plot_data,
        x="Metric",
        y="Score",
        hue="model",
        palette=["#2b5c8f", "#d95f02"],
        edgecolor="black",
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Evaluation Metric", fontsize=11, labelpad=8)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=11, labelpad=8)
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(title="Model", loc="lower right")

    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_json(data: Any, filepath: Path) -> None:
    """Save dictionary or serializable structure to JSON with indentation."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_json(filepath: Path) -> Any:
    """Load JSON file from disk."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

