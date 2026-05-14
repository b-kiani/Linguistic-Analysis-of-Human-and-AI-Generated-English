from pathlib import Path
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import joblib


ROOT = Path(__file__).resolve().parents[1]

IN_PATH = ROOT / "data" / "processed" / "linguistic_features_balanced.csv"

OUT_DIR = ROOT / "outputs" / "ml_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


DROP_COLUMNS = [
    "dataset",
    "domain",
    "label",
    "source",
    "text"
]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 80)
    print(f"Training: {name}")
    print("=" * 80)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("\nMetrics:")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k}: {v:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Human", "AI"]
    ))

    print("\nConfusion matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save classification report
    report = classification_report(
        y_test,
        y_pred,
        target_names=["Human", "AI"],
        output_dict=True
    )

    with open(OUT_DIR / f"{name}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # Save confusion matrix
    cm_df = pd.DataFrame(
        cm,
        index=["Actual_Human", "Actual_AI"],
        columns=["Predicted_Human", "Predicted_AI"]
    )
    cm_df.to_csv(OUT_DIR / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")

    # Save model
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")

    return metrics


def plot_metrics(results_df):
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    for metric in metric_cols:
        plt.figure(figsize=(8, 5))
        plt.bar(results_df["model"], results_df[metric])
        plt.ylim(0, 1)
        plt.title(f"Model comparison: {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"model_comparison_{metric}.png", dpi=300)
        plt.close()


def main():
    print("Loading linguistic features...")
    df = pd.read_csv(IN_PATH)

    feature_cols = [c for c in df.columns if c not in DROP_COLUMNS]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = df["label"].astype(int)

    print(f"Rows: {len(df):,}")
    print(f"Features: {len(feature_cols)}")
    print(feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    all_results = []

    for name, model in models.items():
        metrics = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )
        all_results.append(metrics)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUT_DIR / "model_metrics_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)
    print(results_df)

    plot_metrics(results_df)

    print("\nSaved results to:")
    print(OUT_DIR)

    print("\nSaved models to:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()