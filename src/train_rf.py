"""Random Forest baseline training and evaluation module.

Trains a balanced Random Forest classifier strictly on the 80% training set,
evaluates on the untouched 20% test set, extracts feature importances,
and outputs standardized metrics and figures.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import get_train_test_split
from src.utils import (
    calculate_metrics,
    get_baseline_results_dir,
    get_figures_dir,
    get_models_dir,
    get_reports_dir,
    plot_confusion_matrix,
    plot_feature_importance,
    save_json,
)


def train_random_forest() -> Tuple[RandomForestClassifier, dict, pd.DataFrame]:
    """Train and evaluate Random Forest baseline model under strict train/test isolation."""
    print("=" * 60)
    print("RANDOM FOREST BASELINE EXPERIMENT")
    print("=" * 60)

    # 1. Obtain strictly isolated train/test sets
    X_train, X_test, y_train, y_test, feature_list = get_train_test_split()

    # 2. Instantiate reproducible baseline Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # 3. Train ONLY on training partition
    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    rf_model.fit(X_train, y_train)

    # 4. Evaluate ONLY on untouched test partition
    print(f"Evaluating RandomForestClassifier on {len(X_test)} test samples...")
    y_pred = rf_model.predict(X_test)
    classes = sorted(list(np.unique(y_train)))

    # Compute comprehensive metrics
    metrics = calculate_metrics(y_test, y_pred, labels=classes)
    metrics["model"] = "Random Forest"
    metrics["n_estimators"] = 200
    metrics["random_state"] = 42
    metrics["class_weight"] = "balanced"
    metrics["train_samples"] = int(len(X_train))
    metrics["test_samples"] = int(len(X_test))

    # 5. Extract and rank feature importances
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_list,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # 6. Save Model Artifact
    model_path = get_models_dir() / "random_forest.joblib"
    joblib.dump(rf_model, model_path)
    print(f"Model saved to: {model_path}")

    # 7. Save Metrics JSON
    json_path = get_baseline_results_dir() / "random_forest.json"
    save_json(metrics, json_path)
    print(f"Metrics saved to: {json_path}")

    # 8. Save Classification Report Text
    report_path = get_reports_dir() / "random_forest_report.txt"
    report_content = (
        f"============================================================\n"
        f"RANDOM FOREST BASELINE CLASSIFICATION REPORT\n"
        f"============================================================\n"
        f"Model: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')\n"
        f"Train size: {len(X_train)} | Test size: {len(X_test)}\n\n"
        f"Overall Metrics:\n"
        f"  - Accuracy          : {metrics['accuracy']:.4f}\n"
        f"  - Balanced Accuracy : {metrics['balanced_accuracy']:.4f}\n"
        f"  - Precision (Macro) : {metrics['precision_macro']:.4f}\n"
        f"  - Precision (Wtd)   : {metrics['precision_weighted']:.4f}\n"
        f"  - Recall (Macro)    : {metrics['recall_macro']:.4f}\n"
        f"  - Recall (Wtd)      : {metrics['recall_weighted']:.4f}\n"
        f"  - F1-Score (Macro)  : {metrics['f1_macro']:.4f}\n"
        f"  - F1-Score (Wtd)    : {metrics['f1_weighted']:.4f}\n\n"
        f"Detailed Classification Report:\n"
        f"{metrics['classification_report']}\n"
        f"Confusion Matrix (labels={classes}):\n"
        f"{np.array(metrics['confusion_matrix'])}\n"
        f"============================================================\n"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Report saved to: {report_path}")

    # 9. Save Confusion Matrix Figure
    cm_fig_path = get_figures_dir() / "random_forest_confusion_matrix.png"
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=classes,
        output_path=cm_fig_path,
        title="Random Forest Baseline - Confusion Matrix",
    )
    print(f"Confusion matrix figure saved to: {cm_fig_path}")

    # 10. Save Feature Importance CSV and Figure
    fi_csv_path = get_reports_dir() / "random_forest_feature_importance.csv"
    importance_df.to_csv(fi_csv_path, index=False)
    print(f"Feature importance CSV saved to: {fi_csv_path}")

    fi_fig_path = get_figures_dir() / "rf_feature_importance.png"
    plot_feature_importance(
        importance_df=importance_df,
        output_path=fi_fig_path,
        title="Random Forest - Top 20 Feature Importances",
        top_n=20,
    )
    print(f"Feature importance plot saved to: {fi_fig_path}")

    print("\nRandom Forest Results Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f} | Weighted F1: {metrics['f1_weighted']:.4f}")
    print("=" * 60 + "\n")

    return rf_model, metrics, importance_df


if __name__ == "__main__":
    train_random_forest()

