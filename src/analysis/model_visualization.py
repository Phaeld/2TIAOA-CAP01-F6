"""
Project: Multi-Agent Cardiac Risk Prediction System
AUTOR: Raphael da Silva RM561452
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    learning_curve
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

DATA_PATH = SRC_DIR / "data" / "synthetic_data.csv"
TEST_PATH = SRC_DIR / "data" / "test_data.csv"
MODEL_PATH = SRC_DIR / "models" / "risk_model.joblib"

OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"
METRICS_PATH = OUTPUT_DIR / "model_metrics.json"

TRAIN_PATH = SRC_DIR / "data" / "train_data.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (9, 6),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11
})


def save_chart(filename):
    """Save the current chart and release memory."""

    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Chart saved: {output_path}")


def load_resources():
    """Load the dataset, test records and trained model."""

    dataset = pd.read_csv(DATA_PATH)
    test_data = pd.read_csv(TEST_PATH)

    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    features = artifact["features"]
    target = artifact["target"]

    x_test = test_data[features]
    y_test = test_data[target]

    predictions = model.predict(x_test)

    positive_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(x_test)[
        :, positive_index
    ]

    return (
        dataset,
        model,
        features,
        target,
        x_test,
        y_test,
        predictions,
        probabilities
    )


def plot_class_distribution(dataset, target):
    """Visualize the target class distribution."""

    counts = dataset[target].value_counts().reindex(
        [0, 1], fill_value=0
    )

    fig, ax = plt.subplots()

    bars = ax.bar(
        ["Low Risk (0)", "High Risk (1)"],
        counts.values
    )

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom"
        )

    ax.set_title("Synthetic Risk Class Distribution")
    ax.set_xlabel("Risk Class")
    ax.set_ylabel("Number of Records")
    ax.set_ylim(0, max(counts.values) * 1.15)

    save_chart("01_class_distribution.png")


def plot_confusion_matrix(y_test, predictions):
    """Visualize correct and incorrect classifications."""

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    fig, ax = plt.subplots()

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Low Risk", "High Risk"]
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title("Random Forest Confusion Matrix")

    save_chart("02_confusion_matrix.png")


def plot_model_metrics(y_test, predictions):
    """Compare classification metrics."""

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    fig, ax = plt.subplots()

    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values())
    )

    for bar, value in zip(bars, metrics.values()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center"
        )

    ax.set_ylim(0, 1.12)
    ax.set_title("Random Forest Classification Metrics")
    ax.set_ylabel("Metric Score")

    save_chart("03_model_metrics.png")

    return metrics


def plot_feature_importance(model, features):
    """Visualize Random Forest feature importance."""

    importances = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values()

    fig, ax = plt.subplots()

    ax.barh(
        importances.index,
        importances.values
    )

    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Mean Decrease in Impurity")

    save_chart("04_feature_importance.png")


def plot_roc_curve(y_test, probabilities):
    """Visualize the ROC curve and calculate AUC."""

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    fig, ax = plt.subplots()

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Random Forest (AUC = {auc_score:.4f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title("Receiver Operating Characteristic")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.25)

    save_chart("05_roc_curve.png")

    return auc_score


def plot_probability_distribution(y_test, probabilities):
    """Visualize probability estimates by true class."""

    labels = np.asarray(y_test)

    fig, ax = plt.subplots()

    ax.hist(
        probabilities[labels == 0],
        bins=20,
        alpha=0.65,
        label="True Low Risk"
    )

    ax.hist(
        probabilities[labels == 1],
        bins=20,
        alpha=0.65,
        label="True High Risk"
    )

    ax.axvline(
        0.5,
        linestyle="--",
        label="Default Threshold (0.5)"
    )

    ax.set_title("Predicted Risk Probability Distribution")
    ax.set_xlabel("Predicted Probability of Class 1")
    ax.set_ylabel("Number of Records")
    ax.legend()

    save_chart("06_probability_distribution.png")


def plot_learning_curve(model, dataset, features, target):
    """Evaluate learning behavior using stratified CV."""

    # Recreate the original training split
    x_all = dataset[features]
    y_all = dataset[target]

    x, _, y, _ = train_test_split(
        x_all,
        y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all
)

    cross_validation = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42
    )

    train_sizes, train_scores, validation_scores = (
        learning_curve(
            estimator=model,
            X=x,
            y=y,
            train_sizes=np.linspace(0.2, 1.0, 5),
            cv=cross_validation,
            scoring="f1",
            n_jobs=1
        )
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    validation_mean = np.mean(
        validation_scores, axis=1
    )
    validation_std = np.std(
        validation_scores, axis=1
    )

    fig, ax = plt.subplots()

    ax.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training F1"
    )

    ax.plot(
        train_sizes,
        validation_mean,
        marker="s",
        label="Cross-Validation F1"
    )

    ax.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15
    )

    ax.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15
    )

    ax.set_title("Random Forest Learning Curve")
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.25)

    save_chart("07_learning_curve.png")


def main():
    """Execute the complete visualization pipeline."""

    print("\nMODEL PERFORMANCE ANALYSIS\n")

    (
        dataset,
        model,
        features,
        target,
        x_test,
        y_test,
        predictions,
        probabilities
    ) = load_resources()

    print(f"Dataset records: {len(dataset)}")
    print(f"Test records: {len(x_test)}")
    print(f"Features: {len(features)}")

    plot_class_distribution(dataset, target)
    plot_confusion_matrix(y_test, predictions)

    metrics = plot_model_metrics(
        y_test, predictions
    )

    plot_feature_importance(model, features)

    auc_score = plot_roc_curve(
        y_test, probabilities
    )

    plot_probability_distribution(
        y_test, probabilities
    )

    plot_learning_curve(
        model, dataset, features, target
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    report = {
        "dataset_records": int(len(dataset)),
        "test_records": int(len(x_test)),
        "accuracy": float(metrics["Accuracy"]),
        "precision": float(metrics["Precision"]),
        "recall": float(metrics["Recall"]),
        "f1_score": float(metrics["F1 Score"]),
        "roc_auc": float(auc_score),
        "confusion_matrix": matrix.tolist()
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    print("\nFINAL METRICS")

    for name, value in report.items():
        print(f"{name}: {value}")

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
