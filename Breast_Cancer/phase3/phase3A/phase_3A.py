"""
Breast Cancer Classification — Phase 3A: Error Analysis

This phase goes beyond accuracy and investigates *where and why* the models fail,
with emphasis on clinically critical errors.

Using the exact models and threshold-tuning logic from Phase 2 (for full
reproducibility), the script analyzes:
- False Negatives (malignant → benign), the most dangerous failure mode
- False Positives (benign → malignant), false alarms
- Borderline cases near the tuned decision threshold (ambiguous predictions)

Enhancements in this phase:
- Top-case previews (Top 10 false positives and borderline cases) saved as text
- More informative plots:
  • Probability histograms with tuned threshold, mean, and median
  • FN/FP comparison (default vs tuned thresholds)

The goal is to treat errors as first-class outputs and prepare the ground for
Phase 3B (model explainability).

Outputs are saved under:
results_phase3/error_analysis/<MODEL_NAME>/
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# ------------------------------------------------------------
# Import from Phase 2 (NO duplication of model/tuning logic)
# ------------------------------------------------------------
from phase_2 import (
    ensure_dir,
    get_models,
    tune_threshold_medical,
    predict_with_threshold
)


# -----------------------------
# Local helpers (Phase 3 only)
# -----------------------------

def compute_counts(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Labels: 0=malignant, 1=benign
    confusion_matrix(y_true, y_pred, labels=[0,1]) returns:
      [[true0_pred0, true0_pred1],
       [true1_pred0, true1_pred1]]

    FN (critical): cm[0,1]  malignant -> benign
    FP:            cm[1,0]  benign -> malignant
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = int(cm[0, 0])
    fn = int(cm[0, 1])
    fp = int(cm[1, 0])
    tn = int(cm[1, 1])
    return tp, fn, fp, tn, cm


def build_case_tables(
    X_test: np.ndarray,
    y_test: np.ndarray,
    p_malignant: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    tuned_threshold: float,
    borderline_margin: float
):
    """
    Create dataframes for:
      - false positives
      - false negatives
      - borderline cases (prob close to tuned threshold)
      - hard cases (union of FP + FN + borderline)
    """
    df = pd.DataFrame(X_test, columns=feature_names)
    df["true_label"] = y_test
    df["true_name"] = np.where(y_test == 0, "malignant", "benign")

    df["p_malignant"] = p_malignant
    df["pred_label"] = y_pred
    df["pred_name"] = np.where(y_pred == 0, "malignant", "benign")

    df["threshold_used"] = float(tuned_threshold)
    df["abs_dist_to_threshold"] = np.abs(df["p_malignant"] - tuned_threshold)

    fn_df = df[(df["true_label"] == 0) & (df["pred_label"] == 1)].copy()
    fp_df = df[(df["true_label"] == 1) & (df["pred_label"] == 0)].copy()

    borderline_df = df[df["abs_dist_to_threshold"] <= borderline_margin].copy()
    borderline_df["borderline_margin"] = float(borderline_margin)

    hard_df = pd.concat([fn_df, fp_df, borderline_df], axis=0).drop_duplicates().copy()

    # Helpful sorting: highest P(malignant) first
    for d in (hard_df, fp_df, fn_df):
        d.sort_values(by="p_malignant", ascending=False, inplace=True)

    # Borderline: closest to threshold first
    borderline_df.sort_values(by="abs_dist_to_threshold", ascending=True, inplace=True)

    return hard_df, fp_df, fn_df, borderline_df


def save_top_fp_preview(fp_df: pd.DataFrame, feature_names: list[str], out_path: str, top_k: int = 10) -> None:
    """
    Save the top-K most confident false positives (highest P(malignant)).
    """
    cols = ["true_name", "pred_name", "p_malignant", "abs_dist_to_threshold"] + feature_names[:10]
    preview = fp_df.head(top_k)[cols].copy()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Top False Positives (most confident benign -> predicted malignant)\n")
        f.write("=" * 75 + "\n\n")
        if len(fp_df) == 0:
            f.write("No false positives found.\n")
            return

        f.write(f"Showing top {min(top_k, len(fp_df))} cases by highest P(malignant).\n")
        f.write("Columns include probability + a small subset of features (first 10).\n\n")
        f.write(preview.to_string(index=False))


def save_top_borderline_preview(borderline_df: pd.DataFrame, feature_names: list[str], out_path: str, top_k: int = 10) -> None:
    """
    Save the top-K borderline cases closest to the tuned threshold.
    """
    cols = ["true_name", "pred_name", "p_malignant", "threshold_used", "abs_dist_to_threshold"] + feature_names[:10]
    preview = borderline_df.head(top_k)[cols].copy()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Top Borderline Cases (closest to tuned threshold)\n")
        f.write("=" * 75 + "\n\n")
        if len(borderline_df) == 0:
            f.write("No borderline cases found.\n")
            return

        f.write(f"Showing top {min(top_k, len(borderline_df))} cases by smallest |P(malignant) - threshold|.\n")
        f.write("Columns include probability + a small subset of features (first 10).\n\n")
        f.write(preview.to_string(index=False))


def plot_p_malignant_hist(
    p_malignant: np.ndarray,
    threshold: float,
    title: str,
    save_path: str,
    show_mean_median: bool = True
) -> None:
    """
    Histogram of P(malignant) with tuned threshold marker.
    Added:
      - threshold label
      - mean/median lines (optional)
    """
    mean_val = float(np.mean(p_malignant))
    median_val = float(np.median(p_malignant))

    plt.figure()
    plt.hist(p_malignant, bins=25)

    # Tuned threshold line
    plt.axvline(threshold, linestyle="--")
    plt.text(threshold, 0.95, f"threshold={threshold:.2f}",
             transform=plt.gca().get_xaxis_transform(),
             ha="left", va="top")

    if show_mean_median:
        plt.axvline(mean_val, linestyle="--")
        plt.axvline(median_val, linestyle="--")

        plt.text(mean_val, 0.90, f"mean={mean_val:.2f}",
                 transform=plt.gca().get_xaxis_transform(),
                 ha="left", va="top")

        plt.text(median_val, 0.85, f"median={median_val:.2f}",
                 transform=plt.gca().get_xaxis_transform(),
                 ha="left", va="top")

    plt.title(title)
    plt.xlabel("P(malignant)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fp_feature_shift(
    df_all: pd.DataFrame,
    df_fp: pd.DataFrame,
    feature: str,
    title: str,
    save_path: str
) -> None:
    """
    Compare distribution of a selected feature:
      - all test samples
      - false positives only
    """
    plt.figure()
    plt.hist(df_all[feature].values, bins=25, alpha=0.6, label="All test samples")
    if len(df_fp) > 0:
        plt.hist(df_fp[feature].values, bins=25, alpha=0.6, label="False positives")
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_fn_fp_bar(default_fn: int, default_fp: int, tuned_fn: int, tuned_fp: int, title: str, save_path: str) -> None:
    """
    Small bar plot comparing FN/FP counts for default vs tuned threshold.
    """
    labels = ["FN", "FP"]
    default_vals = [default_fn, default_fp]
    tuned_vals = [tuned_fn, tuned_fp]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure()
    plt.bar(x - w/2, default_vals, width=w, label="Default")
    plt.bar(x + w/2, tuned_vals, width=w, label="Tuned")
    plt.xticks(x, labels)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def write_summary_txt(
    save_path: str,
    model_name: str,
    default_stats: dict,
    tuned_stats: dict,
    borderline_count: int,
    borderline_margin: float
) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Error Analysis Summary — {model_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write("Label convention:\n")
        f.write("  0 = malignant\n")
        f.write("  1 = benign\n\n")

        f.write("Default decision rule (threshold ~ 0.5):\n")
        f.write(f"  Accuracy: {default_stats['acc']:.4f}\n")
        f.write(f"  FN (malignant -> benign): {default_stats['fn']}\n")
        f.write(f"  FP (benign -> malignant): {default_stats['fp']}\n")
        f.write(f"  Confusion matrix:\n{default_stats['cm']}\n\n")

        f.write("Tuned decision rule (medical objective: minimize FN first):\n")
        f.write(f"  Tuned threshold on P(malignant): {tuned_stats['threshold']:.3f}\n")
        f.write(f"  Accuracy: {tuned_stats['acc']:.4f}\n")
        f.write(f"  FN (malignant -> benign): {tuned_stats['fn']}\n")
        f.write(f"  FP (benign -> malignant): {tuned_stats['fp']}\n")
        f.write(f"  Confusion matrix:\n{tuned_stats['cm']}\n\n")

        f.write("Borderline cases:\n")
        f.write(f"  Definition: |P(malignant) - tuned_threshold| <= {borderline_margin:.2f}\n")
        f.write(f"  Count: {borderline_count}\n\n")

        f.write("Notes:\n")
        f.write("- False negatives represent missed malignant cases and are clinically critical.\n")
        f.write("- Threshold tuning typically reduces FN at the cost of increased FP.\n")
        f.write("- Borderline cases lie near the decision boundary and are often ambiguous.\n")


# -----------------------------
# Main (Phase 3A)
# -----------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_root = os.path.join(base_dir, "results_phase3", "error_analysis")
    ensure_dir(out_root)

    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target  # 0 malignant, 1 benign
    feature_names = list(data.feature_names)

    # Same split policy as Phase 2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Phase 2 models (imported)
    models = get_models(random_state=42)

    thresholds = np.linspace(0.05, 0.95, 91)
    borderline_margin = 0.05

    print("\n" + "#" * 80)
    print("PHASE 3A — ERROR ANALYSIS START")
    print("#" * 80)
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Borderline margin: ±{borderline_margin:.2f} around tuned threshold\n")

    for model_name, model in models.items():
        model_dir = os.path.join(out_root, model_name)
        ensure_dir(model_dir)

        print("\n" + "=" * 80)
        print(f"MODEL: {model_name}")
        print("=" * 80)

        # Train
        model.fit(X_train, y_train)

        # Default predictions
        y_pred_default = model.predict(X_test)
        acc_default = accuracy_score(y_test, y_pred_default)
        _, fn_d, fp_d, _, cm_d = compute_counts(y_test, y_pred_default)

        print("Default decision rule:")
        print(f"  Accuracy: {acc_default:.4f}")
        print(f"  FN (malignant -> benign): {fn_d}")
        print(f"  FP (benign -> malignant): {fp_d}")
        print(f"  Confusion matrix:\n{cm_d}")

        # Probabilities
        proba_test = model.predict_proba(X_test)
        p_mal = proba_test[:, 0]

        # Tune threshold (Phase 2 objective)
        best = tune_threshold_medical(y_test, p_mal, thresholds)
        tuned_t = float(best["threshold"])

        # Tuned predictions
        y_pred_tuned = predict_with_threshold(p_mal, tuned_t)
        acc_tuned = accuracy_score(y_test, y_pred_tuned)
        _, fn_t, fp_t, _, cm_t = compute_counts(y_test, y_pred_tuned)

        print("\nTuned decision rule (medical objective):")
        print(f"  Selected threshold on P(malignant): {tuned_t:.3f}")
        print(f"  Accuracy: {acc_tuned:.4f}")
        print(f"  FN (malignant -> benign): {fn_t}")
        print(f"  FP (benign -> malignant): {fp_t}")
        print(f"  Confusion matrix:\n{cm_t}")

        # Build tables
        hard_df, fp_df, fn_df, borderline_df = build_case_tables(
            X_test=X_test,
            y_test=y_test,
            p_malignant=p_mal,
            y_pred=y_pred_tuned,
            feature_names=feature_names,
            tuned_threshold=tuned_t,
            borderline_margin=borderline_margin
        )

        # Save CSVs
        hard_df.to_csv(os.path.join(model_dir, f"hard_cases_{model_name}.csv"), index=False)
        fp_df.to_csv(os.path.join(model_dir, f"false_positives_{model_name}.csv"), index=False)
        fn_df.to_csv(os.path.join(model_dir, f"false_negatives_{model_name}.csv"), index=False)
        borderline_df.to_csv(os.path.join(model_dir, f"borderline_cases_{model_name}.csv"), index=False)

        # Save previews (new)
        save_top_fp_preview(
            fp_df=fp_df,
            feature_names=feature_names,
            out_path=os.path.join(model_dir, f"top_false_positives_{model_name}.txt"),
            top_k=10
        )
        save_top_borderline_preview(
            borderline_df=borderline_df,
            feature_names=feature_names,
            out_path=os.path.join(model_dir, f"top_borderline_{model_name}.txt"),
            top_k=10
        )

        print("\nSaved case tables:")
        print(f"  Hard cases: {len(hard_df)}")
        print(f"  False positives: {len(fp_df)}")
        print(f"  False negatives: {len(fn_df)}")
        print(f"  Borderline cases: {len(borderline_df)}")

        # Plot 1: histogram (improved)
        plot_p_malignant_hist(
            p_malignant=p_mal,
            threshold=tuned_t,
            title=f"{model_name} — P(malignant) distribution (tuned threshold={tuned_t:.2f})",
            save_path=os.path.join(model_dir, "p_malignant_hist.png"),
            show_mean_median=True
        )

        # Plot 2: feature shift for FP
        feature_to_plot = "mean concave points" if "mean concave points" in feature_names else feature_names[0]
        df_all = pd.DataFrame(X_test, columns=feature_names)

        df_fp_features = fp_df[[feature_to_plot]].copy() if len(fp_df) > 0 else pd.DataFrame(columns=[feature_to_plot])

        plot_fp_feature_shift(
            df_all=df_all,
            df_fp=df_fp_features,
            feature=feature_to_plot,
            title=f"{model_name} — Feature shift for false positives ({feature_to_plot})",
            save_path=os.path.join(model_dir, "false_positive_feature_shift.png")
        )

        # Plot 3: FN/FP bars (new)
        plot_fn_fp_bar(
            default_fn=fn_d,
            default_fp=fp_d,
            tuned_fn=fn_t,
            tuned_fp=fp_t,
            title=f"{model_name} — FN/FP (Default vs Tuned)",
            save_path=os.path.join(model_dir, "fn_fp_default_vs_tuned.png")
        )

        # Summary file
        write_summary_txt(
            save_path=os.path.join(model_dir, f"summary_{model_name}.txt"),
            model_name=model_name,
            default_stats={"acc": acc_default, "fn": fn_d, "fp": fp_d, "cm": cm_d},
            tuned_stats={"acc": acc_tuned, "fn": fn_t, "fp": fp_t, "cm": cm_t, "threshold": tuned_t},
            borderline_count=len(borderline_df),
            borderline_margin=borderline_margin
        )

        print(f"\nSaved summary: summary_{model_name}.txt")
        print("Saved plots: p_malignant_hist.png, false_positive_feature_shift.png, fn_fp_default_vs_tuned.png")
        print("Saved previews: top_false_positives_*.txt, top_borderline_*.txt")

    print("\n" + "#" * 80)
    print("PHASE 3A — ERROR ANALYSIS COMPLETE")
    print("#" * 80)
    print(f"Saved all outputs under: {out_root}\n")


if __name__ == "__main__":
    main()