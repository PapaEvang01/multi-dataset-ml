"""
Breast Cancer Classification — Phase 1 (Baseline Models)

This script implements Phase 1 of a structured machine learning pipeline using the
Breast Cancer Wisconsin (Diagnostic) dataset. The objective is to establish reliable,
interpretable baseline models and validate the data and evaluation setup before
moving to more advanced approaches.

Task:
- Binary classification of tumors as malignant (0) or benign (1) based on
  30 numerical biopsy features.

Methodology:
- Stratified 70/30 train–test split (test set held out from training)
- Feature scaling via sklearn Pipelines to prevent data leakage
- Baseline models:
    • Logistic Regression (interpretable)
    • k-Nearest Neighbors (non-parametric)

Evaluation:
- Training vs test accuracy
- Precision, recall, F1-score
- ROC-AUC (when available)
- Confusion matrices
- False negatives (malignant → benign), emphasized due to medical relevance

Outputs:
- Saved plots (confusion matrices, ROC curves, accuracy comparisons)
- Summary CSV with key metrics for all models

Phase 1 focuses on correctness, interpretability, and evaluation discipline,
providing a validated baseline for Phase 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

# -----------------------------
# Configuration
# -----------------------------
@dataclass(frozen=True)
class Config:
    random_state: int = 42
    train_ratio: float = 0.70  # 70/30 split
    results_dir: str = "results"
    preview_rows: int = 5
    feature_histograms: Tuple[str, ...] = (
        "mean radius",
        "mean texture",
        "mean perimeter",
        "mean area",
        "mean smoothness",
    )


# -----------------------------
# Utility helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def safe_filename(name: str) -> str:
    """Make a safe filename segment from model name."""
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


# -----------------------------
# Data loading + inspection
# -----------------------------
def load_dataset_as_df() -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Loads Breast Cancer Wisconsin (Diagnostic) from sklearn.
    Target convention (sklearn):
      - 0 => malignant
      - 1 => benign
    """
    data = load_breast_cancer()
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    target_names = data.target_names  # ['malignant', 'benign']
    return X_df, y, target_names


def print_dataset_overview(cfg: Config, X_df: pd.DataFrame, y: pd.Series, target_names: np.ndarray) -> None:
    """Show dataset shape, class distribution, and a small preview."""
    print("\n" + "=" * 90)
    print("DATASET OVERVIEW — Breast Cancer Wisconsin (Diagnostic) [TABULAR]")
    print("=" * 90)
    print(f"Samples:  {X_df.shape[0]}")
    print(f"Features: {X_df.shape[1]}")
    print(f"Target names: {list(target_names)} (sklearn mapping: 0=malignant, 1=benign)\n")

    counts = y.value_counts().sort_index()
    dist = pd.DataFrame({
        "class_id": counts.index,
        "class_name": [target_names[i] for i in counts.index],
        "count": counts.values,
        "percent": (counts.values / len(y) * 100).round(2),
    })
    print("Class distribution:")
    print(dist.to_string(index=False))

    preview = X_df.copy()
    preview["target"] = y
    preview["target_name"] = preview["target"].map({0: target_names[0], 1: target_names[1]})

    print(f"\nPreview (first {cfg.preview_rows} rows):")
    print(preview.head(cfg.preview_rows).to_string(index=False))


def plot_class_distribution(cfg: Config, y: pd.Series, target_names: np.ndarray) -> None:
    """Save a simple bar chart of class counts."""
    counts = y.value_counts().sort_index()
    labels = [target_names[i] for i in counts.index]

    plt.figure()
    plt.bar(labels, counts.values)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "class_distribution.png"), dpi=200)
    plt.close()


def plot_feature_histograms(cfg: Config, X_df: pd.DataFrame) -> None:
    """
    Save histograms for a few important features.
    Useful to show: "we inspected the data distribution".
    """
    features = [f for f in cfg.feature_histograms if f in X_df.columns]
    if not features:
        return

    for feat in features:
        plt.figure()
        plt.hist(X_df[feat].values, bins=30)
        plt.title(f"Histogram — {feat}")
        plt.xlabel(feat)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg.results_dir, f"hist_{safe_filename(feat)}.png"), dpi=200)
        plt.close()


# -----------------------------
# Split + models
# -----------------------------
def split_data(
    cfg: Config,
    X_df: pd.DataFrame,
    y: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified split preserves class balance across train/test.
    Using 70/30
    """
    test_size = 1.0 - cfg.train_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y,
        test_size=test_size,
        random_state=cfg.random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def build_models(cfg: Config) -> Dict[str, Pipeline]:
    """
    Phase 1 baselines:
      - Logistic Regression (interpretable)
      - k-NN (simple non-parametric baseline)

    NOTE: StandardScaler is applied inside the pipeline to avoid leakage.
    """
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=cfg.random_state))
        ]),
        "KNN_k5": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
    }


# -----------------------------
# Evaluation + plots
# -----------------------------
def get_score_vector(model: Pipeline, X_test: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Return a score/probability vector for ROC curve / ROC-AUC.
    For binary classification, we use score for class 1 (benign) by sklearn convention.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def save_confusion_matrix(cfg: Config, name: str, y_true: pd.Series, y_pred: np.ndarray, target_names: np.ndarray) -> None:
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot()
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, f"confusion_matrix_{safe_filename(name)}.png"), dpi=200)
    plt.close()


def evaluate_model(
    cfg: Config,
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: np.ndarray
) -> Dict[str, float]:
    """
    Train and evaluate a model.
    Adds:
      - Training accuracy
      - Test accuracy
      - ROC-AUC (if available)
      - False negatives count (malignant predicted as benign)
    """
    print("\n" + "-" * 90)
    print(f"MODEL: {name}")
    print("-" * 90)
    print(f"Training on {len(X_train)} samples (train split only). Evaluating on {len(X_test)} held-out samples.\n")

    # ---- TRAINING (this is the training step)
    model.fit(X_train, y_train)

    # ---- Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # ---- Accuracies
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    # ---- Classification report (on test set)
    print("Test set classification report:")
    print(classification_report(y_test, y_pred_test, target_names=target_names, digits=4))

    # ---- False negatives (medical-critical): malignant (0) predicted as benign (1)
    # Confusion matrix layout: rows=true, cols=pred
    cm = confusion_matrix(y_test, y_pred_test, labels=[0, 1])
    false_negatives = int(cm[0, 1])

    # ---- ROC-AUC (if score vector exists)
    score_vec = get_score_vector(model, X_test)
    auc = roc_auc_score(y_test, score_vec) if score_vec is not None else np.nan

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    if not np.isnan(auc):
        print(f"ROC-AUC (positive class=benign [1]): {auc:.4f}")
    print(f"False negatives (malignant → predicted benign): {false_negatives}")

    # ---- Save confusion matrix
    save_confusion_matrix(cfg, name, y_test, y_pred_test, target_names)

    # ---- Save per-model report file
    report_path = os.path.join(cfg.results_dir, f"report_{safe_filename(name)}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"MODEL: {name}\n\n")
        f.write("Test set classification report:\n")
        f.write(classification_report(y_test, y_pred_test, target_names=target_names, digits=4))
        f.write("\n")
        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Test  accuracy: {test_acc:.4f}\n")
        if not np.isnan(auc):
            f.write(f"ROC-AUC (positive class=benign [1]): {auc:.4f}\n")
        f.write(f"False negatives (malignant → predicted benign): {false_negatives}\n")
        f.write("\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(np.array2string(cm))

    return {
        "model": name,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
        "false_negatives": float(false_negatives),
    }


def plot_roc_curves(
    cfg: Config,
    roc_items: List[Tuple[str, np.ndarray, np.ndarray, float]]
) -> None:
    """
    Save a combined ROC plot for models that can provide a score vector.
    Each item: (model_name, fpr, tpr, auc)
    """
    if not roc_items:
        return

    plt.figure()
    for name, fpr, tpr, auc in roc_items:
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    # Diagonal reference
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title("ROC Curves (Test Set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "roc_curves.png"), dpi=200)
    plt.close()


def plot_accuracy_comparison(cfg: Config, summary_df: pd.DataFrame) -> None:
    """Save a bar chart comparing train vs test accuracy per model."""
    plt.figure()
    x = np.arange(len(summary_df))
    width = 0.35

    plt.bar(x - width / 2, summary_df["train_accuracy"].values, width, label="Train")
    plt.bar(x + width / 2, summary_df["test_accuracy"].values, width, label="Test")

    plt.xticks(x, summary_df["model"].values, rotation=0)
    plt.title("Train vs Test Accuracy (Phase 1 Baselines)")
    plt.ylabel("Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "train_test_accuracy.png"), dpi=200)
    plt.close()


def plot_false_negatives(cfg: Config, summary_df: pd.DataFrame) -> None:
    """Save a bar chart of false negatives per model (medical-critical)."""
    plt.figure()
    plt.bar(summary_df["model"].values, summary_df["false_negatives"].values)
    plt.title("False Negatives per Model (malignant → predicted benign)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.results_dir, "false_negatives.png"), dpi=200)
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    cfg = Config()
    ensure_dir(cfg.results_dir)

    # Load dataset
    X_df, y, target_names = load_dataset_as_df()

    # Print overview in console (shows you understand the data)
    print_dataset_overview(cfg, X_df, y, target_names)

    # Simple EDA plots (saved to results/)
    plot_class_distribution(cfg, y, target_names)
    plot_feature_histograms(cfg, X_df)

    # Split (70/30)
    X_train, X_test, y_train, y_test = split_data(cfg, X_df, y)
    print("\n" + "=" * 90)
    print("SPLIT INFO")
    print("=" * 90)
    print(f"Train: {len(X_train)} samples ({int(cfg.train_ratio * 100)}%)")
    print(f"Test : {len(X_test)} samples ({int((1 - cfg.train_ratio) * 100)}%)")
    print("Note: Training is performed ONLY on the training set. Test set is held out.\n")

    # Build and evaluate baseline models
    models = build_models(cfg)

    summary_rows = []
    roc_items = []

    for name, model in models.items():
        metrics = evaluate_model(cfg, name, model, X_train, y_train, X_test, y_test, target_names)
        summary_rows.append(metrics)

        # ROC curve for models with probability/score output
        score_vec = get_score_vector(model, X_test)
        if score_vec is not None:
            fpr, tpr, _ = roc_curve(y_test, score_vec)
            auc = roc_auc_score(y_test, score_vec)
            roc_items.append((name, fpr, tpr, float(auc)))

    # Save summary CSV for GitHub results
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(cfg.results_dir, "phase1_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Comparison plots (saved to results/)
    plot_roc_curves(cfg, roc_items)
    plot_accuracy_comparison(cfg, summary_df)
    plot_false_negatives(cfg, summary_df)

    print("\n" + "=" * 90)
    print("PHASE 1 COMPLETE")
    print("=" * 90)
    print(f"Saved outputs in: {cfg.results_dir}/")
    print(f"- Summary CSV: {summary_csv}")
    print("- Plots: class distribution, feature histograms, confusion matrices, ROC curves, accuracy comparison, false negatives")


if __name__ == "__main__":
    main()
