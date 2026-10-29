
"""XGBoost baseline training and evaluation module.

Trains an XGBoost multiclass classifier strictly on the 80% training partition,
evaluates on the untouched 20% test set using identical reproducible splits,
extracts feature importances, and records standardized baseline metrics.
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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



def train_xgboost() -> Tuple[XGBClassifier, dict, pd.DataFrame]:
    """Train and evaluate XGBoost baseline model under identical train/test isolation."""
    print("=" * 60)
    print("XGBOOST BASELINE EXPERIMENT")
    print("=" * 60)

    # 1. Obtain identical, strictly isolated train/test sets
    X_train, X_test, y_train, y_test, feature_list = get_train_test_split()

    # 2. Encode target labels for XGBoost multiclass objective (0, 1, 2)
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)
    classes = list(label_encoder.classes_)

    # 3. Instantiate reproducible baseline XGBoost classifier
    xgb_model = XGBClassifier(
        n_estimators=200,
        random_state=42,
        eval_metric="mlogloss",
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        tree_method="hist",
    )

    # 4. Train ONLY on training partition
    print(f"Training XGBClassifier on {len(X_train)} samples...")
    xgb_model.fit(X_train, y_train_enc)

    # 5. Evaluate ONLY on untouched test partition
    print(f"Evaluating XGBClassifier on {len(X_test)} test samples...")
    y_pred_enc = xgb_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_enc)

    # Compute comprehensive metrics
    metrics = calculate_metrics(y_test, y_pred, labels=classes)
    metrics["model"] = "XGBoost"
    metrics["n_estimators"] = 200
    metrics["random_state"] = 42
    metrics["learning_rate"] = 0.1
    metrics["max_depth"] = 6
    metrics["train_samples"] = int(len(X_train))
    metrics["test_samples"] = int(len(X_test))

    # 6. Extract and rank feature importances
    importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": feature_list,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    # 7. Save Model and Encoder Artifacts
    model_path = get_models_dir() / "xgboost.joblib"
    joblib.dump({"model": xgb_model, "label_encoder": label_encoder}, model_path)
    print(f"Model and encoder saved to: {model_path}")

    # 8. Save Metrics JSON
    json_path = get_baseline_results_dir() / "xgboost.json"
    save_json(metrics, json_path)
    print(f"Metrics saved to: {json_path}")

    # 9. Save Classification Report Text
    report_path = get_reports_dir() / "xgboost_report.txt"
    report_content = (
        f"============================================================\n"
        f"XGBOOST BASELINE CLASSIFICATION REPORT\n"
        f"============================================================\n"
        f"Model: XGBClassifier(n_estimators=200, max_depth=6, lr=0.1, random_state=42)\n"
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

    # 10. Save Confusion Matrix Figure
    cm_fig_path = get_figures_dir() / "xgboost_confusion_matrix.png"
    plot_confusion_matrix(
        cm=metrics["confusion_matrix"],
        class_names=classes,
        output_path=cm_fig_path,
        title="XGBoost Baseline - Confusion Matrix",
    )
    print(f"Confusion matrix figure saved to: {cm_fig_path}")

    # 11. Save Feature Importance CSV and Figure
    fi_csv_path = get_reports_dir() / "xgboost_feature_importance.csv"
    importance_df.to_csv(fi_csv_path, index=False)
    print(f"Feature importance CSV saved to: {fi_csv_path}")

    fi_fig_path = get_figures_dir() / "xgb_feature_importance.png"
    plot_feature_importance(
        importance_df=importance_df,
        output_path=fi_fig_path,
        title="XGBoost - Top 20 Feature Importances",
        top_n=20,
    )
    print(f"Feature importance plot saved to: {fi_fig_path}")

    print("\nXGBoost Results Summary:")
    print(f"Accuracy: {metrics['accuracy']:.4f} | Balanced Acc: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f} | Weighted F1: {metrics['f1_weighted']:.4f}")
    print("=" * 60 + "\n")

    return xgb_model, metrics, importance_df


if __name__ == "__main__":
    train_xgboost()

