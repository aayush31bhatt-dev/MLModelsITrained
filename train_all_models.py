"""
Train 3 SMS Spam classifiers on the SMS Spam Collection and persist artifacts.

Models trained:
    1. Multinomial Naive Bayes  (TF-IDF -> MultinomialNB)
    2. Complement Naive Bayes   (TF-IDF -> ComplementNB)
    3. Logistic Regression      (TF-IDF -> LogisticRegression)

For each model we:
    * train + evaluate on the same stratified 80/20 split (random_state=42)
    * save the fitted pipeline to artifacts/model_<algo>_<ts>.joblib
    * save a confusion matrix PNG to artifacts/confusion_matrix_<algo>.png
    * append a summary to artifacts/all_models_report.json

Usage:
    python train_all_models.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend — no GUI needed
import matplotlib.pyplot as plt
import numpy as np

from pipeline import (
    DEFAULT_DATA_PATH,
    PipelineConfig,
    algorithm_label,
    build_pipeline,
    load_dataset,
    save_model,
)
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = ARTIFACT_DIR / "all_models_report.json"

ALGORITHMS = [
    "multinomial_nb",
    "complement_nb",
    "logistic_regression",
]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def plot_confusion_matrix(y_test, y_pred, out_path: Path, title: str) -> None:
    cm = __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_test, y_score, out_path: Path, title: str, auc: float) -> None:
    """Plot and save an ROC curve for a single model."""
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#5ce4c6", lw=2.0, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#6b7480", lw=1.2, linestyle="--", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc_curves_combined(roc_curves: list[dict], out_path: Path, title: str) -> None:
    """Plot all model ROC curves on a single chart for comparison."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    palette = ["#7c9cff", "#5ce4c6", "#ffb86b", "#ff7a90", "#c792ea"]
    for idx, entry in enumerate(roc_curves):
        fpr = entry["fpr"]
        tpr = entry["tpr"]
        ax.plot(
            fpr,
            tpr,
            color=palette[idx % len(palette)],
            lw=2.0,
            label=f"{entry['algorithm_label']} (AUC = {entry['auc']:.4f})",
        )
    ax.plot([0, 1], [0, 1], color="#6b7480", lw=1.2, linestyle="--", label="Random baseline")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def train_one(algo: str, X_train, X_test, y_train, y_test) -> dict:
    """Train + evaluate one algorithm. Returns a summary dict."""
    cfg = PipelineConfig(algorithm=algo)
    label = algorithm_label(algo)

    print(f"\n=== Training {label} ===")
    started = time.perf_counter()
    model = build_pipeline(cfg)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    duration = time.perf_counter() - started

    # Metrics (positive class = spam)
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

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
        name: {
            "precision": float(row["precision"]),
            "recall": float(row["recall"]),
            "f1": float(row["f1-score"]),
            "support": int(row["support"]),
        }
        for name, row in report.items()
        if name in ("ham", "spam")
    }
    aggregates = {
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
            "support": int(report["macro avg"]["support"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
            "support": int(report["weighted avg"]["support"]),
        },
    }

    # ROC AUC + curve (positive class = spam)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: use decision_function if available, else use hard predictions.
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            y_score = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_score = y_pred.astype(float)
    auc = float(roc_auc_score(y_test, y_score))
    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)

    # Persist artifacts
    ts = int(time.time() * 1000)
    artifact_path = ARTIFACT_DIR / f"model_{algo}_{ts}.joblib"
    save_model(model, artifact_path)

    cm_path = ARTIFACT_DIR / f"confusion_matrix_{algo}.png"
    plot_confusion_matrix(
        y_test, y_pred, cm_path, title=f"Confusion Matrix - {label}"
    )

    roc_path = ARTIFACT_DIR / f"roc_curve_{algo}.png"
    plot_roc_curve(
        y_test,
        y_score,
        roc_path,
        title=f"ROC Curve - {label}",
        auc=auc,
    )

    print(f"  accuracy : {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall   : {metrics['recall']:.4f}")
    print(f"  f1       : {metrics['f1']:.4f}")
    print(f"  roc_auc  : {auc:.4f}")
    print(f"  artifact : {artifact_path}")
    print(f"  cm plot  : {cm_path}")
    print(f"  roc plot : {roc_path}")

    return {
        "algorithm": algo,
        "algorithm_label": label,
        "metrics": metrics,
        "per_class": per_class,
        "aggregates": aggregates,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "duration_seconds": float(duration),
        "artifact_path": str(artifact_path),
        "confusion_matrix_path": str(cm_path),
        "roc_auc": auc,
        "roc_curve_path": str(roc_path),
        "roc_curve_points": {
            "fpr": [float(x) for x in fpr.tolist()],
            "tpr": [float(x) for x in tpr.tolist()],
        },
        "trained_at": _utcnow_iso(),
    }


def main() -> None:
    print(f"Loading dataset from {DEFAULT_DATA_PATH} ...")
    df = load_dataset(DEFAULT_DATA_PATH)
    print(f"  rows: {len(df)}")
    class_distribution = {
        "ham": int((df["label"] == 0).sum()),
        "spam": int((df["label"] == 1).sum()),
    }
    print(f"  class distribution: {class_distribution}")

    X = df["text"].values
    y = df["label"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  Test size: {len(X_test)}")

    summaries = []
    for algo in ALGORITHMS:
        summaries.append(train_one(algo, X_train, X_test, y_train, y_test))

    # Combined ROC curve plot for all models
    combined_roc_path = ARTIFACT_DIR / "roc_curve_all_models.png"
    plot_roc_curves_combined(
        [
            {
                "algorithm_label": s["algorithm_label"],
                "fpr": s["roc_curve_points"]["fpr"],
                "tpr": s["roc_curve_points"]["tpr"],
                "auc": s["roc_auc"],
            }
            for s in summaries
        ],
        combined_roc_path,
        title="ROC Curves — All Models",
    )
    print(f"\nWrote combined ROC plot to {combined_roc_path}")

    report = {
        "generated_at": _utcnow_iso(),
        "dataset": {
            "path": str(DEFAULT_DATA_PATH),
            "total_rows": int(len(df)),
            "class_distribution": class_distribution,
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "random_state": 42,
            "test_size_pct": 0.20,
        },
        "models": summaries,
        "roc_curve_combined_path": str(combined_roc_path),
    }

    REPORT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\nWrote summary report to {REPORT_PATH}")

    # Quick comparison table
    print("\n=== Comparison ===")
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC AUC':>9}")
    for s in summaries:
        m = s["metrics"]
        print(
            f"{s['algorithm_label']:<22} "
            f"{m['accuracy']:>9.4f} "
            f"{m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} "
            f"{s['roc_auc']:>9.4f}"
        )
    print("Done.")


if __name__ == "__main__":
    main()
