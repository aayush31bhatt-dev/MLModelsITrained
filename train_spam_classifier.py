"""
SMS Spam Classifier
===================

Trains a baseline Multinomial Naive Bayes model with a TF-IDF feature
extractor on the SMS Spam Collection (spam.csv), evaluates it on a held-out
test set, and plots a confusion matrix.

Dataset schema (spam.csv):
    v1  - label ("ham" or "spam")
    v2  - raw SMS text
    Unnamed: 2, 3, 4 - mostly empty, ignored

Usage:
    pip install -r requirements.txt
    python train_spam_classifier.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).parent / "spam.csv"
TEST_SIZE = 0.20
RANDOM_STATE = 42
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
CM_PLOT_PATH = ARTIFACT_DIR / "confusion_matrix.png"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load spam.csv and return a clean DataFrame with `label` and `text` columns."""
    if not csv_path.exists():
        sys.exit(f"ERROR: dataset not found at {csv_path}")

    # The file uses latin-1 encoding in the original SMS Spam Collection.
    df = pd.read_csv(csv_path, encoding="latin-1")

    # Keep only the two meaningful columns and rename them.
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})

    # Normalize labels: ham -> 0, spam -> 1.
    df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})

    # Drop rows with missing text or label.
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)

    # Strip whitespace from text.
    df["text"] = df["text"].astype(str).str.strip()

    # Drop empty messages.
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    """TF-IDF (1-2 grams) -> Multinomial Naive Bayes."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", MultinomialNB(alpha=0.1)),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model: Pipeline, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1),
        "recall": recall_score(y_test, y_pred, pos_label=1),
        "f1": f1_score(y_test, y_pred, pos_label=1),
    }
    return metrics, y_pred


def plot_confusion_matrix(y_test, y_pred, out_path: Path) -> None:
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["ham", "spam"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix - Multinomial NB + TF-IDF")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading dataset from {DATA_PATH} ...")
    df = load_dataset(DATA_PATH)
    print(f"  rows: {len(df)}")
    print(f"  class distribution:\n{df['label'].value_counts().rename({0: 'ham', 1: 'spam'})}")

    X = df["text"].values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

    print("\nBuilding pipeline: TF-IDF (1-2 grams) -> MultinomialNB ...")
    model = build_pipeline()

    print("Fitting model ...")
    model.fit(X_train, y_train)

    print("\nEvaluating on held-out test set ...")
    metrics, y_pred = evaluate(model, X_test, y_test)

    print("\n=== Metrics ===")
    for name, value in metrics.items():
        print(f"  {name:>9}: {value:.4f}")

    print("\n=== Classification report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["ham", "spam"],
            digits=4,
        )
    )

    print(f"Saving confusion matrix to {CM_PLOT_PATH} ...")
    plot_confusion_matrix(y_test, y_pred, CM_PLOT_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
