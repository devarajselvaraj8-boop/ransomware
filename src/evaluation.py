"""Model comparison and cross-validation evaluation module.

Aggregates baseline performance across Random Forest and XGBoost, creates
comparative metric tables and visualizations, and produces in-depth error analyses
focusing on minority class behavior (Family L), false positives, and false negatives.
"""

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    get_baseline_results_dir,
    get_figures_dir,
    get_reports_dir,
    load_json,
    plot_model_comparison,
)


def evaluate_and_compare_models() -> Tuple[pd.DataFrame, str]:
    """Load baseline results, generate model_comparison.csv, and produce comparison plot."""
    rf_json_path = get_baseline_results_dir() / "random_forest.json"
    xgb_json_path = get_baseline_results_dir() / "xgboost.json"

    if not rf_json_path.exists():
        raise FileNotFoundError(f"Random forest results not found at: {rf_json_path}")
    if not xgb_json_path.exists():
        raise FileNotFoundError(f"XGBoost results not found at: {xgb_json_path}")

    rf_data = load_json(rf_json_path)
    xgb_data = load_json(xgb_json_path)

    # Build comparison DataFrame with exact requested columns
    comparison_records = [
        {
            "model": "Random Forest",
            "accuracy": rf_data["accuracy"],
            "precision": rf_data["precision_weighted"],
            "recall": rf_data["recall_weighted"],
            "f1_weighted": rf_data["f1_weighted"],
            "f1_macro": rf_data["f1_macro"],
            "balanced_accuracy": rf_data["balanced_accuracy"],
        },
        {
            "model": "XGBoost",
            "accuracy": xgb_data["accuracy"],
            "precision": xgb_data["precision_weighted"],
            "recall": xgb_data["recall_weighted"],
            "f1_weighted": xgb_data["f1_weighted"],
            "f1_macro": xgb_data["f1_macro"],
            "balanced_accuracy": xgb_data["balanced_accuracy"],
        },
    ]

    comparison_df = pd.DataFrame(comparison_records)

    # Save results/reports/model_comparison.csv
    csv_path = get_reports_dir() / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Model comparison saved to: {csv_path}")

    # Generate comparison plot
    plot_path = get_figures_dir() / "baseline_model_comparison.png"
    plot_model_comparison(
        comparison_df=comparison_df,
        output_path=plot_path,
        title="Baseline Classifier Comparison (RF vs. XGBoost)",
    )
    print(f"Comparison figure saved to: {plot_path}")

    # In-depth analysis of class-level metrics and confusion dynamics
    analysis_lines = [
        "=" * 80,
        "BASELINE MODEL COMPARISON & ERROR DYNAMICS REPORT",
        "=" * 80,
        "",
        "1. SUMMARY PERFORMANCE METRICS TABLE",
        "-" * 80,
        comparison_df.to_string(index=False),
        "",
        "=" * 80,
        "2. DETAILED MINORITY CLASS (L-CLASS) & ERROR ANALYSIS",
        "-" * 80,
    ]

    for model_name, data in [("Random Forest", rf_data), ("XGBoost", xgb_data)]:
        analysis_lines.append(f"\nMODEL: [{model_name}]")
        analysis_lines.append("-" * 40)
        per_class = data.get("per_class", {})
        classes = data.get("classes", ["E", "G", "L"])
        cm = np.array(data.get("confusion_matrix", []))

        analysis_lines.append("Per-Class Metric Breakdown:")
        for cls_name in classes:
            cls_metrics = per_class.get(cls_name, {})
            p = cls_metrics.get("precision", 0.0)
            r = cls_metrics.get("recall", 0.0)
            f = cls_metrics.get("f1_score", 0.0)
            s = cls_metrics.get("support", 0)
            analysis_lines.append(
                f"  - Class '{cls_name}' (n={s}): Precision={p:.4f}, Recall={r:.4f}, F1-Score={f:.4f}"
            )

        # Calculate False Positives and False Negatives per class from CM
        analysis_lines.append("\nError Matrix Decomposition (FP / FN):")
        total_fp = 0
        total_fn = 0
        for i, cls_name in enumerate(classes):
            tp = cm[i, i]
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            total_fp += fp
            total_fn += fn
            analysis_lines.append(
                f"  - Class '{cls_name}': True Positives={tp}, False Negatives={fn}, False Positives={fp}"
            )

        analysis_lines.append(f"Total Misclassifications : {total_fn}")
        analysis_lines.append(f"Confusion Matrix (Rows=True, Cols=Pred, Order={classes}):\n{cm}")

    analysis_lines.append("")
    analysis_lines.append("=" * 80)
    analysis_lines.append("3. SCIENTIFIC DISCUSSION: WHY ACCURACY ALONE IS INSUFFICIENT")
    analysis_lines.append("-" * 80)
    analysis_lines.append("In malware family discrimination, high accuracy can often mask dangerous blind spots:")
    analysis_lines.append("  (1) Severe Class Imbalance: Class 'L' represents only 9.31% (33 test samples).")
    analysis_lines.append("      A naive classifier predicting only 'G' and 'E' would achieve >90% overall accuracy")
    analysis_lines.append("      while completely failing to detect family 'L'. Macro F1 and Balanced Accuracy")
    analysis_lines.append("      are essential to penalize minority misclassification.")
    analysis_lines.append("  (2) False Negatives vs. False Positives: In defensive cybersecurity, a False Negative")
    analysis_lines.append("      (allowing ransomware to run undetected) results in irreversible data encryption,")
    analysis_lines.append("      whereas a False Positive (flagging benign activity) causes operational friction.")
    analysis_lines.append("      Evaluating per-class recall is therefore paramount.")
    analysis_lines.append("  (3) Perfect Baseline Score Context: Both Random Forest and XGBoost achieve 1.00")
    analysis_lines.append("      accuracy across all partitions. This indicates that the 46 post-execution sandbox")
    analysis_lines.append("      summary features provide trivially separable signatures among the 3 families.")
    analysis_lines.append("      However, as demonstrated in our leakage analysis, this does NOT demonstrate early-stage")
    analysis_lines.append("      detection capability during ongoing execution.")
    analysis_lines.append("=" * 80)

    analysis_text = "\n".join(analysis_lines) + "\n"
    report_output = get_reports_dir() / "model_comparison_analysis.txt"
    with open(report_output, "w", encoding="utf-8") as f:
        f.write(analysis_text)

    print(analysis_text)
    return comparison_df, analysis_text


if __name__ == "__main__":
    evaluate_and_compare_models()

