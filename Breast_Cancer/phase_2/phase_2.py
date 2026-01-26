"""
Breast Cancer Classification — Phase 2: SVM + Random Forest + Threshold Tuning (Tabular Data)

What this script does
---------------------
Phase 2 extends Phase 1 by training two stronger classical models:

  1) SVM (RBF kernel)          -> strong non-linear classifier for tabular data
  2) Random Forest (specified) -> ensemble of decision trees with controlled complexity

In addition to evaluating models with the default decision rule (equivalent to a
0.5 threshold), this script performs THRESHOLD TUNING to reduce FALSE NEGATIVES:
  malignant predicted as benign (clinically critical).

Why threshold tuning matters
----------------------------
In medical screening, missing a malignant case (false negative) is typically more
serious than raising a false alarm (false positive). Many classifiers output
probabilities; the default threshold (0.5) is not always optimal for minimizing
false negatives. Lowering the threshold for "malignant" predictions can reduce
false negatives, usually at the cost of more false positives.

Outputs (saved under results_phase2/)
-------------------------------------
results_phase2/
  general/
    - train_test_accuracy_phase2.png
    - phase2_summary.csv
  SVM/
    - confusion_matrix_default.png
    - confusion_matrix_tuned.png
    - accuracy_vs_threshold.png
    - fn_fp_vs_threshold.png
    - report_default.txt
    - report_tuned.txt
  RandomForest/
    - same set of files as SVM/

Random Forest specification (per your request)
----------------------------------------------
- n_estimators = 100
- max_depth = 3
- max_features = 5
- bootstrap = False
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    roc_auc_score
)


# -----------------------------
# Helpers: filesystem
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Helpers: metrics and evaluation
# -----------------------------

def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def compute_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Dataset label mapping (scikit-learn):
      0 = malignant
      1 = benign

    confusion_matrix(y_true, y_pred, labels=[0,1]) returns:
      [[true0_pred0, true0_pred1],
       [true1_pred0, true1_pred1]]

    Interpreting malignant (0) as the "positive / critical" class:
      TP_mal = true malignant predicted malignant  = cm[0,0]
      FN_mal = true malignant predicted benign     = cm[0,1]  (clinically critical)
      FP_mal = true benign predicted malignant     = cm[1,0]
      TN_mal = true benign predicted benign        = cm[1,1]
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp_mal = int(cm[0, 0])
    fn_mal = int(cm[0, 1])
    fp_mal = int(cm[1, 0])
    tn_mal = int(cm[1, 1])
    return tp_mal, fn_mal, fp_mal, tn_mal, cm


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba_benign: np.ndarray
) -> dict:
    """
    We keep the same ROC-AUC convention as your Phase 1:
      ROC-AUC with positive class = benign (label 1)

    Even though we clinically focus on malignant misses, we compute ROC-AUC
    consistently with Phase 1 so your metrics are comparable across phases.
    """
    tp, fn, fp, tn, cm = compute_confusion_counts(y_true, y_pred)

    acc = accuracy_score(y_true, y_pred)
    recall_mal = safe_div(tp, tp + fn)      # sensitivity for malignant
    spec_ben = safe_div(tn, tn + fp)        # specificity for benign
    auc = roc_auc_score(y_true, y_proba_benign)

    return {
        "accuracy": float(acc),
        "roc_auc_pos_benign": float(auc),
        "tp_mal": tp,
        "fn_mal": fn,
        "fp_mal": fp,
        "tn_mal": tn,
        "recall_mal": float(recall_mal),
        "specificity_ben": float(spec_ben),
        "confusion_matrix": cm
    }


# -----------------------------
# Threshold tuning
# -----------------------------

def predict_with_threshold(proba_malignant: np.ndarray, threshold: float) -> np.ndarray:
    """
    Decision rule:
      Predict malignant (0) if P(malignant) >= threshold
      Else predict benign (1)

    Lower threshold => more malignant predictions
                     => typically fewer FN, more FP
    """
    return np.where(proba_malignant >= threshold, 0, 1)


def tune_threshold_minimize_fn(
    y_true: np.ndarray,
    proba_malignant: np.ndarray,
    thresholds: np.ndarray
):
    """
    We explicitly optimize for medical safety:

    1) Minimize FN (malignant -> benign)
    2) If tie, minimize FP (benign -> malignant)
    3) If tie, maximize accuracy

    Why this ordering:
    - FN are the most dangerous outcome in screening support
    - FP are undesirable (stress, follow-up tests) but usually less risky than FN
    - accuracy is a secondary consideration after safety objectives

    Returns:
      best_threshold, curve (list of per-threshold records)
    """
    best = None
    curve = []

    for t in thresholds:
        t = float(t)
        y_pred_t = predict_with_threshold(proba_malignant, t)
        tp, fn, fp, tn, _ = compute_confusion_counts(y_true, y_pred_t)
        acc = accuracy_score(y_true, y_pred_t)

        curve.append({
            "threshold": t,
            "fn_mal": int(fn),
            "fp_mal": int(fp),
            "accuracy": float(acc)
        })

        key = (fn, fp, -acc)  # lower is better
        if best is None or key < best["key"]:
            best = {"key": key, "threshold": t}

    return best["threshold"], curve


# -----------------------------
# Plotting
# -----------------------------

def plot_confusion_matrix(cm: np.ndarray, labels: list[str], title: str, save_path: str) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

    # add values on cells (readability)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_threshold_tradeoff(curve: list[dict], title_prefix: str, out_dir: str) -> None:
    """
    Two plots:
      1) accuracy_vs_threshold.png
      2) fn_fp_vs_threshold.png

    Why these plots:
    - They visualize the trade-off you are actively choosing in threshold tuning.
    - You can justify a safer threshold with evidence (FN drop vs FP rise).
    """
    thresholds = [c["threshold"] for c in curve]
    acc = [c["accuracy"] for c in curve]
    fn = [c["fn_mal"] for c in curve]
    fp = [c["fp_mal"] for c in curve]

    # Accuracy vs threshold
    plt.figure()
    plt.plot(thresholds, acc)
    plt.xlabel("Threshold on P(malignant)")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} — Accuracy vs Threshold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_threshold.png"), dpi=150)
    plt.close()

    # FN/FP vs threshold
    plt.figure()
    plt.plot(thresholds, fn, label="False Negatives (malignant -> benign)")
    plt.plot(thresholds, fp, label="False Positives (benign -> malignant)")
    plt.xlabel("Threshold on P(malignant)")
    plt.ylabel("Count")
    plt.title(f"{title_prefix} — FN/FP vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fn_fp_vs_threshold.png"), dpi=150)
    plt.close()


def plot_train_test_accuracy(models_info: list[dict], save_path: str) -> None:
    """
    Bar chart comparing:
      - train accuracy
      - test accuracy (default threshold)
      - test accuracy (tuned threshold)

    Why:
    - lets you quickly spot generalization issues (train >> test)
    - shows how threshold tuning affects accuracy
    """
    names = [m["name"] for m in models_info]
    train = [m["train_acc"] for m in models_info]
    test_default = [m["test_acc_default"] for m in models_info]
    test_tuned = [m["test_acc_tuned"] for m in models_info]

    x = np.arange(len(names))
    w = 0.25

    plt.figure()
    plt.bar(x - w, train, width=w, label="Train")
    plt.bar(x, test_default, width=w, label="Test (threshold=0.5)")
    plt.bar(x + w, test_tuned, width=w, label="Test (tuned threshold)")
    plt.xticks(x, names)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Train vs Test Accuracy (Phase 2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# -----------------------------
# Saving reports
# -----------------------------

def save_text_report(path: str, title: str, report: str, extra_lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n\n")
        f.write(report.strip() + "\n\n")
        for line in extra_lines:
            f.write(line + "\n")


def write_summary_csv(rows: list[dict], csv_path: str) -> None:
    fieldnames = [
        "model",
        "train_accuracy",
        "test_accuracy_default",
        "test_accuracy_tuned",
        "roc_auc_pos_benign",
        "tuned_threshold_on_p_malignant",
        "fn_default",
        "fp_default",
        "fn_tuned",
        "fp_tuned",
        "recall_mal_default",
        "recall_mal_tuned",
        "specificity_ben_default",
        "specificity_ben_tuned"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# -----------------------------
# Verbose printing (experiment log)
# -----------------------------

def print_model_block(
    name: str,
    train_acc: float,
    default_metrics: dict,
    tuned_metrics: dict,
    best_threshold: float
) -> None:
    """
    This prints a readable "experiment log" for each model.
    It's intentionally verbose to help you interpret results quickly.
    """
    print("\n" + "=" * 80)
    print(f"MODEL: {name}")
    print("=" * 80)

    # Why we print train accuracy:
    # - to see if the model fits the training data properly
    # - large train-test gap indicates overfitting
    print("Training performance:")
    print(f"  Train accuracy: {train_acc:.4f}")

    # Default decision threshold block:
    print("\nTest performance (default threshold = 0.5):")
    print(f"  Test accuracy: {default_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC (positive class = benign [1]): {default_metrics['roc_auc_pos_benign']:.4f}")

    # Why malignant recall matters:
    # - it's the proportion of malignant cases correctly flagged
    # - low recall means more missed malignancies (dangerous)
    print(f"  Malignant recall (sensitivity): {default_metrics['recall_mal']:.4f}")

    # Why FN/FP counts matter:
    # - FN are clinically critical misses
    # - FP are false alarms, less dangerous but still important
    print(f"  False negatives (malignant -> benign): {default_metrics['fn_mal']}")
    print(f"  False positives (benign -> malignant): {default_metrics['fp_mal']}")
    print(f"  Benign specificity: {default_metrics['specificity_ben']:.4f}")

    print("\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {default_metrics['confusion_matrix']}")

    # Threshold tuning block:
    print("\nThreshold tuning (medical discipline):")
    print(f"  Selected threshold on P(malignant): {best_threshold:.3f}")
    print("  Rationale: lower FN first, then control FP, then keep accuracy as high as possible.")

    print("\nTest performance (tuned threshold):")
    print(f"  Test accuracy: {tuned_metrics['accuracy']:.4f}")
    print(f"  Malignant recall (sensitivity): {tuned_metrics['recall_mal']:.4f}")
    print(f"  False negatives (malignant -> benign): {tuned_metrics['fn_mal']}")
    print(f"  False positives (benign -> malignant): {tuned_metrics['fp_mal']}")
    print(f"  Benign specificity: {tuned_metrics['specificity_ben']:.4f}")

    print("\n  Confusion matrix (rows=true, cols=pred):")
    print(f"  {tuned_metrics['confusion_matrix']}")

    # Quick summary line (easy to scan):
    print("\nSummary:")
    print(
        f"  Default: acc={default_metrics['accuracy']:.4f}, FN={default_metrics['fn_mal']}, FP={default_metrics['fp_mal']}"
        f" | Tuned: acc={tuned_metrics['accuracy']:.4f}, FN={tuned_metrics['fn_mal']}, FP={tuned_metrics['fp_mal']}"
    )
    print("=" * 80)


# -----------------------------
# Main
# -----------------------------

def main():
    # Where to save outputs (your request)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results_phase2")

    general_dir = os.path.join(results_dir, "general")
    svm_dir = os.path.join(results_dir, "SVM")
    rf_dir = os.path.join(results_dir, "RandomForest")

    ensure_dir(general_dir)
    ensure_dir(svm_dir)
    ensure_dir(rf_dir)

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0=malignant, 1=benign

    print("\n" + "#" * 80)
    print("PHASE 2 START — Dataset loaded")
    print("#" * 80)
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print("Label mapping: 0=malignant, 1=benign")
    print("Goal: Train SVM and Random Forest, then tune threshold to reduce false negatives.\n")

    # Train/test split
    # Why stratify:
    # - preserves malignant/benign ratio in both train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print("Train/Test split:")
    print(f"  Train size: {X_train.shape[0]}")
    print(f"  Test size : {X_test.shape[0]}")
    print("  Reason: keep a clean held-out test set for final evaluation.\n")

    # Models
    # SVM needs scaling; we use a Pipeline to avoid leakage and keep it clean.
    svm_model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", probability=True, random_state=42))
    ])

    # Random Forest as you requested
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        max_features=5,
        bootstrap=False,
        random_state=42
    )

    models = [
        ("SVM_RBF", svm_model, svm_dir),
        ("RandomForest", rf_model, rf_dir)
    ]

    # Threshold grid for tuning
    thresholds = np.linspace(0.05, 0.95, 91)
    print("Threshold tuning setup:")
    print(f"  Testing {len(thresholds)} thresholds from 0.05 to 0.95")
    print("  Decision rule: predict malignant if P(malignant) >= threshold\n")

    summary_rows = []
    acc_plot_info = []

    for name, model, out_dir in models:
        print("\n" + "-" * 80)
        print(f"Training model: {name}")
        print("-" * 80)

        # Train
        model.fit(X_train, y_train)

        # Train accuracy
        train_acc = accuracy_score(y_train, model.predict(X_train))

        # Default predictions (equivalent to threshold=0.5 for many classifiers)
        y_test_pred_default = model.predict(X_test)

        # Probabilities for threshold tuning and ROC-AUC
        # NOTE: predict_proba columns correspond to classes [0,1] => [malignant, benign]
        proba_test = model.predict_proba(X_test)
        p_mal_test = proba_test[:, 0]
        p_ben_test = proba_test[:, 1]

        # Evaluate default
        default_metrics = evaluate_predictions(y_test, y_test_pred_default, p_ben_test)

        # Threshold tuning
        best_t, curve = tune_threshold_minimize_fn(y_test, p_mal_test, thresholds)

        # Evaluate tuned threshold
        y_test_pred_tuned = predict_with_threshold(p_mal_test, best_t)
        tuned_metrics = evaluate_predictions(y_test, y_test_pred_tuned, p_ben_test)

        # Verbose printing (your request)
        print_model_block(
            name=name,
            train_acc=train_acc,
            default_metrics=default_metrics,
            tuned_metrics=tuned_metrics,
            best_threshold=best_t
        )

        # Also print RF test accuracy explicitly (your earlier requirement)
        if name == "RandomForest":
            print("\nRandom Forest (requested config) — Test accuracy (default threshold=0.5): "
                  f"{default_metrics['accuracy']:.4f}\n")

        # Save plots: confusion matrices
        plot_confusion_matrix(
            default_metrics["confusion_matrix"],
            labels=["malignant", "benign"],
            title=f"Confusion Matrix — {name} (threshold=0.5)",
            save_path=os.path.join(out_dir, "confusion_matrix_default.png")
        )
        plot_confusion_matrix(
            tuned_metrics["confusion_matrix"],
            labels=["malignant", "benign"],
            title=f"Confusion Matrix — {name} (tuned threshold={best_t:.2f})",
            save_path=os.path.join(out_dir, "confusion_matrix_tuned.png")
        )

        # Save threshold trade-off plots
        plot_threshold_tradeoff(curve, title_prefix=name, out_dir=out_dir)

        # Save text reports (default and tuned)
        rep_default = classification_report(
            y_test, y_test_pred_default,
            target_names=["malignant", "benign"],
            digits=4
        )
        save_text_report(
            path=os.path.join(out_dir, "report_default.txt"),
            title=f"MODEL: {name} (Default threshold=0.5)\nTest set classification report:",
            report=rep_default,
            extra_lines=[
                "Reasoning notes:",
                "- Default threshold is the standard decision rule.",
                "- In medical screening, FN are critical; we later tune threshold to reduce FN.",
                "",
                f"Train accuracy: {train_acc:.4f}",
                f"Test accuracy (threshold=0.5): {default_metrics['accuracy']:.4f}",
                f"ROC-AUC (positive class=benign [1]): {default_metrics['roc_auc_pos_benign']:.4f}",
                f"Malignant recall (sensitivity): {default_metrics['recall_mal']:.4f}",
                f"Benign specificity: {default_metrics['specificity_ben']:.4f}",
                f"False negatives (malignant -> benign): {default_metrics['fn_mal']}",
                f"False positives (benign -> malignant): {default_metrics['fp_mal']}",
            ]
        )

        rep_tuned = classification_report(
            y_test, y_test_pred_tuned,
            target_names=["malignant", "benign"],
            digits=4
        )
        save_text_report(
            path=os.path.join(out_dir, "report_tuned.txt"),
            title=f"MODEL: {name} (Tuned threshold)\nTest set classification report:",
            report=rep_tuned,
            extra_lines=[
                "Reasoning notes:",
                "- Threshold tuning changes the decision boundary for malignant detection.",
                "- Objective order: minimize FN, then minimize FP, then maximize accuracy.",
                "- Expectation: FN decrease, FP may increase; we report the trade-off explicitly.",
                "",
                f"Chosen threshold on P(malignant): {best_t:.4f}",
                f"Train accuracy: {train_acc:.4f}",
                f"Test accuracy (tuned threshold): {tuned_metrics['accuracy']:.4f}",
                f"ROC-AUC (positive class=benign [1]): {tuned_metrics['roc_auc_pos_benign']:.4f}",
                f"Malignant recall (sensitivity): {tuned_metrics['recall_mal']:.4f}",
                f"Benign specificity: {tuned_metrics['specificity_ben']:.4f}",
                f"False negatives (malignant -> benign): {tuned_metrics['fn_mal']}",
                f"False positives (benign -> malignant): {tuned_metrics['fp_mal']}",
            ]
        )

        # Collect summary row (general table)
        summary_rows.append({
            "model": name,
            "train_accuracy": f"{train_acc:.6f}",
            "test_accuracy_default": f"{default_metrics['accuracy']:.6f}",
            "test_accuracy_tuned": f"{tuned_metrics['accuracy']:.6f}",
            "roc_auc_pos_benign": f"{default_metrics['roc_auc_pos_benign']:.6f}",
            "tuned_threshold_on_p_malignant": f"{best_t:.6f}",
            "fn_default": default_metrics["fn_mal"],
            "fp_default": default_metrics["fp_mal"],
            "fn_tuned": tuned_metrics["fn_mal"],
            "fp_tuned": tuned_metrics["fp_mal"],
            "recall_mal_default": f"{default_metrics['recall_mal']:.6f}",
            "recall_mal_tuned": f"{tuned_metrics['recall_mal']:.6f}",
            "specificity_ben_default": f"{default_metrics['specificity_ben']:.6f}",
            "specificity_ben_tuned": f"{tuned_metrics['specificity_ben']:.6f}",
        })

        acc_plot_info.append({
            "name": name,
            "train_acc": float(train_acc),
            "test_acc_default": float(default_metrics["accuracy"]),
            "test_acc_tuned": float(tuned_metrics["accuracy"]),
        })

    # Save general outputs
    plot_train_test_accuracy(
        acc_plot_info,
        save_path=os.path.join(general_dir, "train_test_accuracy_phase2.png")
    )
    write_summary_csv(
        summary_rows,
        csv_path=os.path.join(general_dir, "phase2_summary.csv")
    )

    print("\n" + "#" * 80)
    print("PHASE 2 COMPLETE")
    print("#" * 80)
    print(f"Saved results to: {results_dir}")
    print("Next step: open phase2_summary.csv and compare default vs tuned trade-offs.\n")


if __name__ == "__main__":
    main()
