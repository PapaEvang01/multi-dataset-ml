
"""
model.py

Model components for Phase 1 (baseline).

Responsibilities:
- Build the TF-IDF vectorizer (text -> numeric features).
- Build the classification model (Logistic Regression).

Important:
- This file should NOT contain training loops or evaluation printing.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def build_vectorizer(cfg):
    """
    Create a TF-IDF vectorizer using config settings.

    TF-IDF converts text into sparse numerical vectors.
    This is the classic baseline representation for sentiment analysis.
    """
    return TfidfVectorizer(
        max_features=cfg.max_features,
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
    )


def build_model(cfg):
    """
    Create the baseline classifier.

    Logistic Regression is a strong, interpretable model for sparse text features.
    """
    return LogisticRegression(max_iter=cfg.max_iter)
