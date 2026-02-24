"""
train.py

Pipeline orchestration.

Responsibilities:
- Load config
- Load and preprocess data
- Split into train/test
- Vectorize text with TF-IDF
- Train Logistic Regression model
- Evaluate results

This file controls the full workflow.
"""

from sklearn.model_selection import train_test_split

from config import Config
from data import load_imdb_dataset
from model import build_vectorizer, build_model
from evaluate import evaluate


def main():
    """
    Run the  training pipeline.
    """

    print("\n========== IMDB Sentiment Analysis ==========\n")

    # ---------------------------------------------------
    # 1) Load configuration
    # ---------------------------------------------------
    cfg = Config()
    print("[INFO] Configuration loaded successfully.")
    print(f"[INFO] Dataset path: {cfg.dataset_path}")

    # ---------------------------------------------------
    # 2) Load dataset
    # ---------------------------------------------------
    print("\n[INFO] Loading dataset...")
    X, y, target_names = load_imdb_dataset(cfg.dataset_path)

    print("[INFO] Dataset loaded successfully.")
    print(f"[INFO] Total samples: {len(X)}")
    print(f"[INFO] Class labels: {target_names}")

    # ---------------------------------------------------
    # 3) Train/Test split
    # ---------------------------------------------------
    print("\n[INFO] Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    print("[INFO] Split completed.")
    print(f"[INFO] Train size: {len(X_train)}")
    print(f"[INFO] Test size: {len(X_test)}")

    # ---------------------------------------------------
    # 4) TF-IDF Vectorization
    # ---------------------------------------------------
    print("\n[INFO] Building TF-IDF vectorizer...")
    vectorizer = build_vectorizer(cfg)

    print("[INFO] Transforming training data...")
    X_train_vec = vectorizer.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_vec = vectorizer.transform(X_test)

    print("[INFO] Vectorization completed.")
    print(f"[INFO] TF-IDF train shape: {X_train_vec.shape}")
    print(f"[INFO] TF-IDF test shape: {X_test_vec.shape}")

    # ---------------------------------------------------
    # 5) Model Training
    # ---------------------------------------------------
    print("\n[INFO] Training Logistic Regression model...")
    model = build_model(cfg)
    model.fit(X_train_vec, y_train)
    print("[INFO] Model training completed successfully.")

    # ---------------------------------------------------
    # 6) Evaluation
    # ---------------------------------------------------
    print("\n[INFO] Evaluating model...\n")
    evaluate(model, vectorizer, X_test_vec, y_test, target_names, top_n=25)

    print("\n========== Pipeline Finished Successfully ==========\n")


if __name__ == "__main__":
    main()