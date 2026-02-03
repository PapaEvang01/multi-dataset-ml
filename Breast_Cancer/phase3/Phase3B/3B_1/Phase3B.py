"""
Breast Cancer Classification — Phase 3B.1: Explainability (Global Feature Importance)

Goal
----
Phase 3B explains *why* models make decisions. In 3B.1 (global explainability),
we estimate which input features matter most overall for each model.

Reproducibility
---------------
We reuse the exact Phase 2 model definitions via imports from phase_2.py.
No model logic is re-implemented here.

Method
------
Permutation Feature Importance:
- Shuffle one feature column at a time on the test set
- Measure how much performance drops
- Bigger drop => more important feature

We compute importance under two scoring views:
1) Accuracy (general performance)
2) Malignant recall (sensitivity for class 0; clinically important)

Outputs
-------
results_phase3/explainability/<MODEL_NAME>/
  - perm_importance_accuracy.csv
  - perm_importance_recall_malignant.csv
  - global_feature_importance_accuracy.png
  - global_feature_importance_recall_malignant.png
  - global_feature_importance_summary.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import recall_score, make_scorer, accuracy_score

# ------------------------------------------------------------
# Import from Phase 2 (NO duplication of model definitions)
# ------------------------------------------------------------
from phase_2 import ensure_dir, get_models


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_top_importances(feature_names, importances_mean, importances_std, title, save_path, top_k=15):
    """
    Simple horizontal bar plot for top-k features by importance mean.
    """
    idx = np.argsort(importances_mean)[::-1][:top_k]
    top_features = [feature_names[i] for i in idx]
    top_means = importances_mean[idx]
    top_stds = importances_std[idx]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_features))[::-1], top_means, xerr=top_stds)
    plt.yticks(range(len(top_features))[::-1], top_features)
    plt.xlabel("Importance (performance drop after shuffling)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_importance_table(feature_names, importances_mean, importances_std, save_csv_path):
    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": importances_mean,
        "importance_std": importances_std
    }).sort_values("importance_mean", ascending=False)
    df.to_csv(save_csv_path, index=False)
    return df


def write_summary_txt(path, model_name, top_acc_df, top_rec_df):
    """
    Write a short, human-readable summary per model.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Global Explainability Summary — {model_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("What this means:\n")
        f.write("- Permutation importance measures how much performance drops when a feature is randomized.\n")
        f.write("- Higher drop => the model relied more on that feature.\n\n")

        f.write("Top features by ACCURACY importance:\n")
        for i, row in top_acc_df.iterrows():
            f.write(f"  - {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}\n")

        f.write("\nTop features by MALIGNANT RECALL importance (class 0 sensitivity):\n")
        for i, row in top_rec_df.iterrows():
            f.write(f"  - {row['feature']}: {row['importance_mean']:.6f} ± {row['importance_std']:.6f}\n")

        f.write("\nNotes:\n")
        f.write("- If the malignant-recall top features differ from accuracy top features, it indicates\n")
        f.write("  that the model uses different signals for overall correctness vs catching malignancies.\n")
        f.write("- This global view is a foundation for Phase 3B.2 (local explanations for hard cases).\n")


# -----------------------------
# Main
# -----------------------------
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, "results_phase3", "explainability")
    ensure_dir(out_root)

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 malignant, 1 benign
    feature_names = list(data.feature_names)

    # Same split policy as Phase 2 / Phase 3A
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Models from Phase 2 (reused)
    models = get_models(random_state=42)

    # Scorers
    acc_scorer = "accuracy"
    recall_malignant_scorer = make_scorer(recall_score, pos_label=0)  # sensitivity for malignant

    print("\n" + "#" * 80)
    print("PHASE 3B.1 — GLOBAL EXPLAINABILITY (PERMUTATION IMPORTANCE)")
    print("#" * 80)
    print(f"Test set size: {X_test.shape[0]}")
    print("Scorers: accuracy, malignant recall (pos_label=0)\n")

    for model_name, model in models.items():
        model_dir = os.path.join(out_root, model_name)
        ensure_dir(model_dir)

        print("=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        # Train (same as earlier phases)
        model.fit(X_train, y_train)

        # Quick sanity: base performance
        y_pred = model.predict(X_test)
        base_acc = accuracy_score(y_test, y_pred)
        base_rec_mal = recall_score(y_test, y_pred, pos_label=0)

        print(f"Base test accuracy: {base_acc:.4f}")
        print(f"Base malignant recall (sensitivity): {base_rec_mal:.4f}")

        # Permutation importance: Accuracy
        perm_acc = permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            scoring=acc_scorer,
            n_repeats=15,
            random_state=42,
            n_jobs=-1
        )

        df_acc = save_importance_table(
            feature_names,
            perm_acc.importances_mean,
            perm_acc.importances_std,
            save_csv_path=os.path.join(model_dir, "perm_importance_accuracy.csv")
        )

        plot_top_importances(
            feature_names,
            perm_acc.importances_mean,
            perm_acc.importances_std,
            title=f"{model_name} — Permutation Importance (Accuracy)",
            save_path=os.path.join(model_dir, "global_feature_importance_accuracy.png"),
            top_k=15
        )

        # Permutation importance: Malignant Recall
        perm_rec = permutation_importance(
            estimator=model,
            X=X_test,
            y=y_test,
            scoring=recall_malignant_scorer,
            n_repeats=15,
            random_state=42,
            n_jobs=-1
        )

        df_rec = save_importance_table(
            feature_names,
            perm_rec.importances_mean,
            perm_rec.importances_std,
            save_csv_path=os.path.join(model_dir, "perm_importance_recall_malignant.csv")
        )

        plot_top_importances(
            feature_names,
            perm_rec.importances_mean,
            perm_rec.importances_std,
            title=f"{model_name} — Permutation Importance (Malignant Recall)",
            save_path=os.path.join(model_dir, "global_feature_importance_recall_malignant.png"),
            top_k=15
        )

        # Write short summary for GitHub/thesis discussion
        top_acc = df_acc.head(10)
        top_rec = df_rec.head(10)

        write_summary_txt(
            path=os.path.join(model_dir, "global_feature_importance_summary.txt"),
            model_name=model_name,
            top_acc_df=top_acc,
            top_rec_df=top_rec
        )

        print("Saved:")
        print("  - perm_importance_accuracy.csv")
        print("  - perm_importance_recall_malignant.csv")
        print("  - global_feature_importance_accuracy.png")
        print("  - global_feature_importance_recall_malignant.png")
        print("  - global_feature_importance_summary.txt\n")

    print("#" * 80)
    print("PHASE 3B.1 COMPLETE")
    print("#" * 80)
    print(f"Outputs saved under: {out_root}\n")


if __name__ == "__main__":
    main()
