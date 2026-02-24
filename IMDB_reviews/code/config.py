"""
config.py

Project configuration for the IMDB Sentiment Analysis pipeline.

Purpose:
- Centralizes all experiment parameters.
- Avoids hard-coded values inside training logic.
- Ensures reproducibility and portability across environments.
"""

import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    Configuration container used throughout the pipeline.

    All adjustable parameters should be defined here.
    """

    # ---------------------------------------------------
    # Project & Dataset Paths
    # ---------------------------------------------------

    # Determine absolute project root (independent of run location)
    project_root: str = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    # Absolute dataset path (prevents working-directory issues)
    dataset_path: str = os.path.join(project_root, "data", "aclImdb")

    # ---------------------------------------------------
    # Train/Test Split
    # ---------------------------------------------------

    test_size: float = 0.2
    random_state: int = 42

    # ---------------------------------------------------
    # TF-IDF Parameters
    # ---------------------------------------------------

    max_features: int = 30000          # Maximum vocabulary size
    ngram_range: tuple = (1, 1)        # (1,1)=unigrams | (1,2)=add bigrams later
    min_df: int = 2                    # Ignore extremely rare words
    max_df: float = 0.95               # Ignore overly common words

    # ---------------------------------------------------
    # Model Parameters
    # ---------------------------------------------------

    max_iter: int = 2000               # Ensures convergence for sparse features