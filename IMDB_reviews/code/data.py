# src/data.py
"""
data.py

Data loading + text preprocessing.

Responsibilities:
- Load IMDB reviews from disk (train/test folders).
- Clean raw review text into a consistent format for TF-IDF.

Important:
- This file should NOT contain model training or evaluation logic.
"""

import re
import numpy as np
from sklearn.datasets import load_files


def clean_text(text: str) -> str:
    """
    Minimal text cleaning for IMDB reviews.

    Steps:
    1) lowercase
    2) remove common HTML breaks and tags
    3) keep only English letters and spaces
    4) collapse multiple spaces

    Note:
    We intentionally keep this simple for Phase 1 (baseline).
    """
    text = text.lower()

    # IMDB often contains <br /> for new lines
    text = re.sub(r"<br\s*/?>", " ", text)

    # Remove any remaining HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep only letters and whitespace (removes digits, punctuation, etc.)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_imdb_dataset(dataset_path: str):
    """
    Load the Stanford IMDB dataset from the given folder.

    Expected structure:
      data/aclImdb/train/pos
      data/aclImdb/train/neg
      data/aclImdb/test/pos
      data/aclImdb/test/neg

    Returns:
      X (list[str]): cleaned review texts
      y (np.ndarray): labels (0/1) assigned by sklearn load_files
      target_names (list[str]): class names in order used by sklearn
    """
    # Load training split from disk
    train = load_files(
        f"{dataset_path}/train",
        categories=["pos", "neg"],
        encoding="utf-8",
        decode_error="replace",
    )

    # Load test split from disk
    test = load_files(
        f"{dataset_path}/test",
        categories=["pos", "neg"],
        encoding="utf-8",
        decode_error="replace",
    )

    # Combine train + test so we control our own split (reproducible baseline)
    X = np.concatenate([train.data, test.data])
    y = np.concatenate([train.target, test.target])

    # Clean all reviews
    X = [clean_text(t) for t in X]

    # target_names describes the label order (e.g., ["neg","pos"] or ["pos","neg"])
    target_names = train.target_names

    return X, y, target_names
