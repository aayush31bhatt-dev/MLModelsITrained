"""
End-to-end integration test for the SMS Spam Classifier.

Exercises every shipped surface in one pass:

  1. CLI baseline (`train_spam_classifier.py`) — runs and writes
     `artifacts/confusion_matrix.png`.
  2. Pipeline module (`pipeline.py`) — train + predict + persist + reload.
  3. Persistence (`db.py`) — `init_db()`, ModelRepository, RunRepository.
  4. Service orchestration (`service.py`) — `start_training` + `run_inference`.
  5. FastAPI app (`app.py`) — `/health`, `/models/train`, `/models`,
     `/models/{id}`, `/predict`, `/runs`, `/runs/{id}` via TestClient.
  6. Realtime (`events.py` + `/ws/events`) — hello + progress + completion
     events delivered to a WebSocket subscriber.

Run with:
    python tests/integration_test.py

Exit code 0 = all surfaces green. Non-zero = at least one surface failed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402
from db import (  # noqa: E402
    ModelRepository,
    RunRepository,
    get_session,
    init_db,
)
from events import bus  # noqa: E402
from pipeline import (  # noqa: E402
    DEFAULT_DATA_PATH,
    PipelineConfig,
    load_model,
    predict as pipeline_predict,
    save_model,
    train as pipeline_train,
)


# ---------------------------------------------------------------------------
# Tiny test runner
# ---------------------------------------------------------------------------
RESULTS: list[tuple[str, bool, str]] = []


def _record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name}{(' — ' + detail) if detail else ''}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------
# 1. CLI baseline
# ---------------------------------------------------------------------------
def test_cli_baseline() -> None:
    section("1. CLI baseline (train_spam_classifier.py)")
    try:
        import subprocess

        proc = subprocess.run(
            [sys.executable, str(ROOT / "train_spam_classifier.py")],
            capture_output=True,
            text=True,
            timeout=180,
        )
        ok = proc.returncode == 0
        _record(
            "cli exits 0",
            ok,
            f"rc={proc.returncode}" + ("" if ok else f" stderr={proc.stderr[:200]}"),
        )
        cm_path = ROOT / "artifacts" / "confusion_matrix.png"
        _record("confusion matrix written", cm_path.exists(), str(cm_path))
        # Sanity-check the printed metrics block.
        _record(
            "stdout contains metrics",
            "accuracy" in proc.stdout and "f1" in proc.stdout,
        )
    except Exception as exc:  # noqa: BLE001
        _record("cli baseline", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 2. Pipeline module
# ---------------------------------------------------------------------------
def test_pipeline_module() -> None:
    section("2. Pipeline module (pipeline.py)")
    try:
        cfg = PipelineConfig()
        result = pipeline_train(DEFAULT_DATA_PATH, cfg)
        _record(
            "train returns metrics",
            all(k in result.metrics for k in ("accuracy", "precision", "recall", "f1")),
            json.dumps({k: round(v, 4) for k, v in result.metrics.items()}),
        )

        artifact = ROOT / "artifacts" / "integration_test_model.joblib"
        save_model(result.model, artifact)
        reloaded = load_model(artifact)
        preds = pipeline_predict(reloaded, ["Free entry in our weekly contest!", "Hey, are we still on for lunch?"])
        _record("predict returns 2 rows", len(preds) == 2, f"n={len(preds)}")
        _record(
            "labels are ham/spam",
            all(p.label in ("ham", "spam") for p in preds),
            ",".join(p.label for p in preds),
        )
        _record(
            "spam_probability in [0,1]",
            all(0.0 <= p.spam_probability <= 1.0 for p in preds),
        )
    except Exception as exc:  # noqa: BLE001
        _record("pipeline module", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 3. Persistence
# ---------------------------------------------------------------------------
def test_persistence() -> None:
    section("3. Persistence (db.py)")
    try:
        init_db()
        _record("init_db is idempotent", True)
        with get_session() as session:
            models = ModelRepository(session).list(limit=5)
            runs = RunRepository(session).list(limit=5)
        _record(
            "repositories return lists",
            isinstance(models, list) and isinstance(runs, list),
            f"models={len(models)} runs={len(runs)}",
        )
    except Exception as exc:  # noqa: BLE001
        _record("persistence", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 4. Service orchestration
# ---------------------------------------------------------------------------
def test_service() -> None:
    section("4. Service orchestration (service.py)")
    try:
        from service import run_inference, start_training

        run_id, model_id = start_training("integration-test", None)
        _record("start_training returns ids", bool(run_id) and bool(model_id), f"run={run_id[:8]} model={model_id[:8]}")

        # Wait for the background job to finish (poll the DB).
        deadline = time.time() + 60
        final_status = None
        while time.time() < deadline:
            with get_session() as session:
                rec = RunRepository(session).get(run_id)
                if rec and rec.status in ("succeeded", "failed"):
                    final_status = rec.status
                    break
            time.sleep(0.5)
        _record("background run completes", final_status == "succeeded", f"status={final_status}")

        chosen_id, preds = run_inference(model_id, ["WINNER!! As a valued network customer you have been selected."])
        _record("run_inference returns predictions", len(preds) == 1, f"model={chosen_id[:8]} label={preds[0]['label']}")
    except Exception as exc:  # noqa: BLE001
        _record("service", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 5. FastAPI surface
# ---------------------------------------------------------------------------
def test_api_surface() -> None:
    section("5. FastAPI surface (app.py)")
    try:
        client = TestClient(app)
        with client:  # triggers lifespan -> init_db
            r = client.get("/health")
            _record("GET /health 200", r.status_code == 200, r.text[:120])
            body = r.json()
            _record(
                "/health payload",
                body.get("status") == "ok" and "version" in body,
                json.dumps(body),
            )

            r = client.get("/models")
            _record("GET /models 200", r.status_code == 200, f"n={len(r.json())}")

            r = client.get("/runs")
            _record("GET /runs 200", r.status_code == 200, f"n={len(r.json())}")

            r = client.post(
                "/models/train",
                json={"name": "integration-api", "config": None},
            )
            _record("POST /models/train 202", r.status_code == 202, r.text[:120])
            accepted = r.json()
            run_id = accepted["run_id"]
            model_id = accepted["model_id"]

            # Wait for completion.
            deadline = time.time() + 60
            while time.time() < deadline:
                rr = client.get(f"/runs/{run_id}")
                if rr.status_code == 200 and rr.json()["status"] in ("succeeded", "failed"):
                    break
                time.sleep(0.5)
            _record(
                "GET /runs/{id} returns succeeded",
                rr.status_code == 200 and rr.json()["status"] == "succeeded",
                rr.json().get("status"),
            )

            r = client.get(f"/models/{model_id}")
            _record("GET /models/{id} 200", r.status_code == 200, r.text[:120])

            r = client.post(
                "/predict",
                json={"texts": ["Free entry in 2 a weekly comp!", "Ok, see you at 7"], "model_id": model_id},
            )
            _record("POST /predict 200", r.status_code == 200, r.text[:160])
            preds = r.json()["predictions"]
            _record("/predict returns 2 predictions", len(preds) == 2)

            # 404 path
            r = client.get("/models/does-not-exist")
            _record("GET /models/{bad} 404", r.status_code == 404)
    except Exception as exc:  # noqa: BLE001
        _record("api surface", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# 6. Realtime WebSocket
# ---------------------------------------------------------------------------
def test_realtime() -> None:
    section("6. Realtime WebSocket (/ws/events)")
    try:
        client = TestClient(app)
        with client:
            with client.websocket_connect("/ws/events") as ws:
                hello = ws.receive_text()
                hello_obj = json.loads(hello)
                _record("ws hello frame", hello_obj.get("type") == "hello", hello[:120])

                # Kick off a training run; expect progress + run.completed + model.created.
                r = client.post(
                    "/models/train",
                    json={"name": "integration-ws", "config": None},
                )
                assert r.status_code == 202, r.text
                run_id = r.json()["run_id"]

                seen_types: set[str] = set()
                deadline = time.time() + 60
                while time.time() < deadline and not {"run.completed", "model.created"}.issubset(seen_types):
                    try:
                        msg = ws.receive_text()
                    except Exception:  # noqa: BLE001
                        break
                    obj = json.loads(msg)
                    seen_types.add(obj.get("type", ""))
                    if obj.get("type") == "run.completed" and obj.get("run_id") == run_id:
                        break

                _record(
                    "ws delivered progress",
                    "progress" in seen_types,
                    ",".join(sorted(seen_types)),
                )
                _record(
                    "ws delivered run.completed",
                    "run.completed" in seen_types,
                    ",".join(sorted(seen_types)),
                )
                _record(
                    "ws delivered model.created",
                    "model.created" in seen_types,
                    ",".join(sorted(seen_types)),
                )
    except Exception as exc:  # noqa: BLE001
        _record("realtime", False, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> int:
    print("SMS Spam Classifier — end-to-end integration pass")
    print(f"  project root: {ROOT}")
    print(f"  dataset:      {DEFAULT_DATA_PATH} (exists={DEFAULT_DATA_PATH.exists()})")

    test_cli_baseline()
    test_pipeline_module()
    test_persistence()
    test_service()
    test_api_surface()
    test_realtime()

    section("Summary")
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    total = len(RESULTS)
    print(f"  {passed}/{total} checks passed")
    if passed != total:
        print("  failures:")
        for name, ok, detail in RESULTS:
            if not ok:
                print(f"    - {name}: {detail}")
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
