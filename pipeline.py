"""
Reusable SMS Spam classification pipeline.

Extracted from `train_spam_classifier.py` so the FastAPI service can train
and evaluate models in-process (no subprocess spawning) and persist the
fitted pipeline to disk via joblib.

Public API:
    load_dataset(csv_path: Path) -> pd.DataFrame
    build_pipeline(config: PipelineConfig | None = None) -> Pipeline
    train(csv_path: Path, config: PipelineConfig | None = None,
          progress_cb: Callable[[float, str], None] | None = None) -> TrainResult
    predict(model: Pipeline, texts: list[str]) -> list[Prediction]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATH = Path(__file__).parent / "spam.csv"
DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42


@dataclass
class PipelineConfig:
    """Hyperparameters for the TF-IDF + classifier pipeline."""

    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    sublinear_tf: bool = True
    alpha: float = 0.1
    test_size: float = DEFAULT_TEST_SIZE
    random_state: int = DEFAULT_RANDOM_STATE
    algorithm: str = "multinomial_nb"  # multinomial_nb | complement_nb | logistic_regression
    max_iter: int = 1000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Load spam.csv and return a clean DataFrame with `label` and `text` columns."""
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding="latin-1")
    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df["label"] = df["label"].str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label", "text"]).reset_index(drop=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_pipeline(config: PipelineConfig | None = None) -> Pipeline:
    """TF-IDF -> classifier pipeline.

    Supported algorithms (via `config.algorithm`):
      - "multinomial_nb"      -> MultinomialNB
      - "complement_nb"       -> ComplementNB
      - "logistic_regression" -> LogisticRegression
    """
    cfg = config or PipelineConfig()
    algo = (cfg.algorithm or "multinomial_nb").lower()
    if algo == "multinomial_nb":
        clf = MultinomialNB(alpha=cfg.alpha)
    elif algo == "complement_nb":
        clf = ComplementNB(alpha=cfg.alpha)
    elif algo == "logistic_regression":
        clf = LogisticRegression(
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
            solver="liblinear",
        )
    else:
        raise ValueError(
            f"unknown algorithm: {cfg.algorithm!r} "
            "(expected multinomial_nb | complement_nb | logistic_regression)"
        )
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=cfg.ngram_range,
                    min_df=cfg.min_df,
                    sublinear_tf=cfg.sublinear_tf,
                ),
            ),
            ("clf", clf),
        ]
    )


def algorithm_label(algo: str) -> str:
    """Human-readable algorithm name for display."""
    algo = (algo or "").lower()
    return {
        "multinomial_nb": "MultinomialNB",
        "complement_nb": "ComplementNB",
        "logistic_regression": "LogisticRegression",
    }.get(algo, algo)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ClassMetrics:
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class TrainResult:
    model: Pipeline
    metrics: dict[str, float]
    per_class: dict[str, ClassMetrics]
    confusion_matrix: list[list[int]]
    train_size: int
    test_size: int
    class_distribution: dict[str, int]
    duration_seconds: float
    config: PipelineConfig


@dataclass
class Prediction:
    text: str
    label: str  # "ham" | "spam"
    label_int: int  # 0 | 1
    spam_probability: float


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
ProgressCB = Callable[[float, str], None]


def _noop_progress(_pct: float, _msg: str) -> None:
    return None


def train(
    csv_path: Path,
    config: PipelineConfig | None = None,
    progress_cb: ProgressCB | None = None,
) -> TrainResult:
    """Train + evaluate the pipeline. Calls `progress_cb(percent, message)`."""
    cfg = config or PipelineConfig()
    cb = progress_cb or _noop_progress

    started = time.perf_counter()

    cb(5.0, "loading_dataset")
    df = load_dataset(csv_path)
    class_distribution = {
        "ham": int((df["label"] == 0).sum()),
        "spam": int((df["label"] == 1).sum()),
    }

    X = df["text"].values
    y = df["label"].astype(int).values

    cb(20.0, "splitting")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    cb(35.0, "building_pipeline")
    model = build_pipeline(cfg)

    cb(45.0, "fitting")
    model.fit(X_train, y_train)

    cb(75.0, "predicting")
    y_pred = model.predict(X_test)

    cb(85.0, "computing_metrics")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
    }

    report = classification_report(
        y_test, y_pred, target_names=["ham", "spam"], output_dict=True, zero_division=0
    )
    per_class = {
        name: ClassMetrics(
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            f1=float(row["f1-score"]),
            support=int(row["support"]),
        )
        for name, row in report.items()
        if name in ("ham", "spam")
    }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    cb(100.0, "done")
    duration = time.perf_counter() - started

    return TrainResult(
        model=model,
        metrics=metrics,
        per_class=per_class,
        confusion_matrix=cm,
        train_size=int(len(X_train)),
        test_size=int(len(X_test)),
        class_distribution=class_distribution,
        duration_seconds=duration,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(model: Pipeline, texts: Iterable[str]) -> list[Prediction]:
    """Run inference on a list of texts using a fitted pipeline."""
    texts = list(texts)
    if not texts:
        return []

    label_int = model.predict(texts)
    # Convert class probabilities to a spam probability in a numerically stable way.
    spam_idx = list(model.classes_).index(1)
    log_proba = model.predict_log_proba(texts)
    spam_log_prob = log_proba[:, spam_idx]
    spam_prob = np.exp(spam_log_prob - np.max(log_proba, axis=1, keepdims=True))
    spam_prob = spam_prob / spam_prob.sum(axis=1, keepdims=True)

    out: list[Prediction] = []
    for text, li, sp in zip(texts, label_int, spam_prob):
        out.append(
            Prediction(
                text=text,
                label="spam" if int(li) == 1 else "ham",
                label_int=int(li),
                spam_probability=float(sp),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"model artifact not found at {path}")
    return joblib.load(path)
