"""
Success-criteria verification for the SMS Spam Classifier.

Trains the baseline pipeline on `spam.csv`, evaluates on the held-out test
set, and asserts the agreed success thresholds. Prints a clear PASS/FAIL
report and exits non-zero if any criterion is missed.

Default thresholds (tuned for the SMS Spam Collection baseline):
    accuracy  >= 0.97
    precision >= 0.95
    recall    >= 0.90
    f1        >= 0.92

Override via CLI flags, e.g.:
    python scripts/verify_success.py --min-accuracy 0.98 --min-f1 0.93

Run with:
    python scripts/verify_success.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import DEFAULT_DATA_PATH, PipelineConfig, train as pipeline_train  # noqa: E402


DEFAULT_CRITERIA = {
    "accuracy": 0.97,
    "precision": 0.95,
    "recall": 0.90,
    "f1": 0.92,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify SMS Spam Classifier success criteria.")
    p.add_argument("--min-accuracy", type=float, default=DEFAULT_CRITERIA["accuracy"])
    p.add_argument("--min-precision", type=float, default=DEFAULT_CRITERIA["precision"])
    p.add_argument("--min-recall", type=float, default=DEFAULT_CRITERIA["recall"])
    p.add_argument("--min-f1", type=float, default=DEFAULT_CRITERIA["f1"])
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    thresholds = {
        "accuracy": args.min_accuracy,
        "precision": args.min_precision,
        "recall": args.min_recall,
        "f1": args.min_f1,
    }

    print("Success-criteria verification")
    print(f"  dataset: {args.data}")
    print(f"  thresholds: {json.dumps(thresholds)}")

    if not args.data.exists():
        print(f"ERROR: dataset not found at {args.data}", file=sys.stderr)
        return 2

    result = pipeline_train(args.data, PipelineConfig())
    metrics = {k: float(v) for k, v in result.metrics.items()}

    print("\nMeasured metrics:")
    for k, v in metrics.items():
        print(f"  {k:>9}: {v:.4f}")

    print("\nPer-class:")
    for name, cm in result.per_class.items():
        print(
            f"  {name:>4}: precision={cm.precision:.4f} "
            f"recall={cm.recall:.4f} f1={cm.f1:.4f} support={cm.support}"
        )

    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    for row in result.confusion_matrix:
        print(f"  {row}")

    print("\nVerdict:")
    all_ok = True
    for key, threshold in thresholds.items():
        actual = metrics[key]
        ok = actual >= threshold
        all_ok = all_ok and ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {key}: {actual:.4f} >= {threshold:.4f}")

    print("\nOverall:", "SUCCESS" if all_ok else "FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
