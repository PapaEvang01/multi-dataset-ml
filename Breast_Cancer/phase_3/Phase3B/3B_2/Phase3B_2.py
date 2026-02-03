"""
Breast Cancer Classification — Phase 3B.2: Local Explainability (Hard Cases)

Goal
----
Explain *why* the model made specific decisions for difficult samples identified
in Phase 3A (false positives and borderline cases).

This phase produces local, sample-level explanations using permutation-based
feature influence, fully compatible with both SVM and Random Forest.

Reproducibility
---------------
- Reuses Phase 2 model definitions
- Uses the same train/test split as Phase 3A
- Explains only real hard cases already identified
"""

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# Import Phase 2 logic (NO duplication)
# ------------------------------------------------------------
from phase_2 import ensure_dir, get_models


# -----------------------------
# Local explanation function
# -----------------------------
def explain_single_sample(
    model,
    x_sample: np.ndarray,
    feature_names: list[str],
    baseline_prob: float,
    X_reference: np.ndarray,
    n_repeats: int = 20
):
    """
    Permutation-based local explanation.

    For one sample:
      - shuffle each feature using values from the reference set
      - measure average change in P(malignant)

    Returns: sorted list of (feature, mean_delta)
    """
    rng = np.random.default_rng(42)
    contributions = []

    for j, fname in enumerate(feature_names):
        deltas = []

        for _ in range(n_repeats):
            x_perturbed = x_sample.copy()
            x_perturbed[j] = rng.choice(X_reference[:, j])

            p_new = model.predict_proba(x_perturbed.reshape(1, -1))[0, 0]
            deltas.append(baseline_prob - p_new)

        contributions.append((fname, float(np.mean(deltas))))

    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions


def write_local_explanation(
    save_path: str,
    model_name: str,
    case_type: str,
    case_index: int,
    baseline_prob: float,
    contributions: list[tuple]
):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Local Explanation — {model_name}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Case type: {case_type}\n")
        f.write(f"Test-set index: {case_index}\n")
        f.write(f"Baseline P(malignant): {baseline_prob:.4f}\n\n")

        f.write("Top contributing features (permutation-based):\n")
        f.write("(Positive = increases malignancy probability)\n\n")

        for feat, delta in contributions[:10]:
            sign = "↑ increases malignancy" if delta > 0 else "↓ decreases malignancy"
            f.write(f"- {feat}: {delta:+.5f} ({sign})\n")

        f.write("\nInterpretation:\n")
        f.write(
            "The listed features had the strongest influence on this individual "
            "prediction. Large positive contributions pushed the model toward a "
            "malignant decision, while negative contributions counteracted it.\n"
        )


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
    y = data.target
    feature_names = list(data.feature_names)

    # Same split as Phase 3A
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Load Phase 2 models
    models = get_models(random_state=42)

    print("\n" + "#" * 80)
    print("PHASE 3B.2 — LOCAL EXPLAINABILITY (HARD CASES)")
    print("#" * 80)

    for model_name, model in models.items():
        print(f"\nMODEL: {model_name}")
        model.fit(X_train, y_train)

        model_dir = os.path.join(out_root, model_name, "local_explanations")
        ensure_dir(model_dir)

        # Load Phase 3A hard-case tables
        phase3_dir = os.path.join(
            base_dir, "results_phase3", "error_analysis", model_name
        )

        fp_path = os.path.join(phase3_dir, f"false_positives_{model_name}.csv")
        borderline_path = os.path.join(phase3_dir, f"borderline_cases_{model_name}.csv")

        fp_df = pd.read_csv(fp_path)
        borderline_df = pd.read_csv(borderline_path)

        # Explain top cases
        for case_type, df in [("false_positive", fp_df), ("borderline", borderline_df)]:
            for i in range(min(3, len(df))):  # top 3 per type
                idx = int(df.iloc[i].name)
                x_sample = X_test[idx]
                baseline_prob = model.predict_proba(x_sample.reshape(1, -1))[0, 0]

                contributions = explain_single_sample(
                    model=model,
                    x_sample=x_sample,
                    feature_names=feature_names,
                    baseline_prob=baseline_prob,
                    X_reference=X_train
                )

                save_path = os.path.join(
                    model_dir, f"explanation_{case_type}_{i}.txt"
                )

                write_local_explanation(
                    save_path=save_path,
                    model_name=model_name,
                    case_type=case_type,
                    case_index=idx,
                    baseline_prob=baseline_prob,
                    contributions=contributions
                )

                print(f"  Saved explanation: {os.path.basename(save_path)}")

    print("\n" + "#" * 80)
    print("PHASE 3B.2 COMPLETE")
    print("#" * 80)


if __name__ == "__main__":
    main()
