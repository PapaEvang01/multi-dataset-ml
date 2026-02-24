"""
evaluate.py

Evaluation utilities.

Responsibilities:
- Compute performance metrics
- Save metrics to file
- Save confusion matrix plot
- Extract & save most influential words (Top Words Analysis)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def _save_top_words_logreg(model, vectorizer, target_names, output_dir, top_n=25):
    """
    Extract top positive/negative words from a trained Logistic Regression model
    using its learned coefficients.

    For binary classification, LogisticRegression.coef_ has shape (1, n_features).
    Positive coefficients push towards class 1; negative towards class 0.
    """
    # Safety checks
    if not hasattr(model, "coef_"):
        raise ValueError("Model has no 'coef_' attribute. Top words require a linear model with coefficients.")

    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_.ravel()  # shape: (n_features,)

    # Identify which label is class 0 and class 1 (for correct interpretation)
    # In your run: target_names = ['neg', 'pos'] and sklearn's y=0->neg, y=1->pos.
    class0_name = target_names[0]
    class1_name = target_names[1]

    top_pos_idx = np.argsort(coef)[-top_n:][::-1]  # largest -> most positive
    top_neg_idx = np.argsort(coef)[:top_n]         # smallest -> most negative

    top_words_path = os.path.join(output_dir, "top_words.txt")
    with open(top_words_path, "w", encoding="utf-8") as f:
        f.write("IMDB Sentiment Analysis - Top Words (Logistic Regression)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Class 1 (positive coefficient direction): {class1_name}\n")
        f.write(f"Class 0 (negative coefficient direction): {class0_name}\n\n")

        f.write(f"Top {top_n} words pushing towards '{class1_name}'\n")
        f.write("-" * 60 + "\n")
        for i in top_pos_idx:
            f.write(f"{feature_names[i]:<25}  coef={coef[i]:+.6f}\n")

        f.write("\n")
        f.write(f"Top {top_n} words pushing towards '{class0_name}'\n")
        f.write("-" * 60 + "\n")
        for i in top_neg_idx:
            f.write(f"{feature_names[i]:<25}  coef={coef[i]:+.6f}\n")

    print(f"[INFO] Top words saved to: {top_words_path}")


def evaluate(model, vectorizer, X_test_vec, y_test, target_names, top_n=25):
    """
    Evaluate a trained model and save:
    - outputs/metrics.txt
    - outputs/confusion_matrix.png
    - outputs/top_words.txt (for Logistic Regression)
    """

    # ---------------------------------------------------
    # Create outputs directory (if it doesn't exist)
    # ---------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------
    # Predictions
    # ---------------------------------------------------
    y_pred = model.predict(X_test_vec)

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    # ---------------------------------------------------
    # Save metrics to text file
    # ---------------------------------------------------
    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("IMDB Sentiment Analysis - Phase 1\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    # ---------------------------------------------------
    # Save confusion matrix as image
    # ---------------------------------------------------
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(plot_path)
    plt.close()

    # ---------------------------------------------------
    # Top Words Analysis (Logistic Regression)
    # ---------------------------------------------------
    _save_top_words_logreg(
        model=model,
        vectorizer=vectorizer,
        target_names=target_names,
        output_dir=output_dir,
        top_n=top_n,
    )

    # ---------------------------------------------------
    # Console summary
    # ---------------------------------------------------
    print("\n[INFO] Evaluation completed.")
    print(f"[INFO] Accuracy: {acc:.4f}")
    print(f"[INFO] Metrics saved to: {metrics_path}")
    print(f"[INFO] Confusion matrix saved to: {plot_path}")