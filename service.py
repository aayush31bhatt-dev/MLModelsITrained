"""
Service layer: orchestrates training, persistence, and realtime events.

This sits between the FastAPI routes (`app.py`) and the lower-level
`pipeline.py` + `db.py` modules. It owns:

  * background training jobs (asyncio tasks)
  * progress + completion event publishing
  * artifact persistence (joblib)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from db import (
    ModelRecord,
    ModelRepository,
    RunRecord,
    RunRepository,
    config_to_dict,
    dict_to_pipeline_config,
    get_session,
)
from events import bus
from pipeline import (
    DEFAULT_DATA_PATH,
    PipelineConfig,
    predict as pipeline_predict,
    save_model,
    train as pipeline_train,
)
from schemas import (
    ModelCreatedEvent,
    ProgressEvent,
    RunCompletedEvent,
)

logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _config_from_request(cfg_in) -> PipelineConfig:
    """Translate a PipelineConfigIn (Pydantic) into a PipelineConfig (dataclass)."""
    if cfg_in is None:
        return PipelineConfig()
    return PipelineConfig(
        ngram_range=(cfg_in.ngram_min, cfg_in.ngram_max),
        min_df=cfg_in.min_df,
        sublinear_tf=cfg_in.sublinear_tf,
        alpha=cfg_in.alpha,
        test_size=cfg_in.test_size,
        random_state=cfg_in.random_state,
    )


def _config_to_request(cfg: PipelineConfig):
    """Translate a PipelineConfig back into a PipelineConfigIn for API responses."""
    from schemas import PipelineConfigIn

    ngram_min, ngram_max = cfg.ngram_range
    return PipelineConfigIn(
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        alpha=cfg.alpha,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------
def start_training(name: str, cfg_in) -> tuple[str, str]:
    """Create Model + Run records and kick off a background training task.

    Returns (run_id, model_id).
    """
    cfg = _config_from_request(cfg_in)

    with get_session() as session:
        model_repo = ModelRepository(session)
        run_repo = RunRepository(session)

        artifact_path = ARTIFACT_DIR / f"model_{int(_utcnow().timestamp() * 1000)}.joblib"

        model_rec = ModelRecord(
            name=name,
            algorithm="MultinomialNB",
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            train_size=0,
            test_size=0,
            class_distribution={"ham": 0, "spam": 0},
            config=config_to_dict(cfg),
            artifact_path=str(artifact_path),
        )
        model_repo.add(model_rec)

        run_rec = RunRecord(
            model_id=model_rec.id,
            status="queued",
            progress_percent=0.0,
            stage="queued",
        )
        run_repo.add(run_rec)

        run_id = run_rec.id
        model_id = model_rec.id
        artifact_str = str(artifact_path)

    # Schedule the background task on the running event loop.
    loop = asyncio.get_running_loop()
    loop.create_task(_run_training_job(run_id, model_id, cfg, artifact_str))

    return run_id, model_id


async def _run_training_job(
    run_id: str,
    model_id: str,
    cfg: PipelineConfig,
    artifact_path: str,
) -> None:
    """Background coroutine: train, persist, publish events."""
    started = _utcnow()

    # Mark running
    with get_session() as session:
        RunRepository(session).update_progress(run_id, 1.0, "starting")

    await bus.publish(
        ProgressEvent(
            run_id=run_id,
            model_id=model_id,
            percent=1.0,
            stage="starting",
            timestamp=_utcnow(),
        ).model_dump(mode="json")
    )

    # Run the (blocking) training in a thread so the event loop stays responsive.
    loop = asyncio.get_running_loop()
    error: str | None = None
    result = None
    try:
        result = await loop.run_in_executor(
            None,
            _train_blocking,
            run_id,
            model_id,
            cfg,
            artifact_path,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("training failed for run %s", run_id)
        error = f"{type(exc).__name__}: {exc}"

    finished = _utcnow()
    duration = (finished - started).total_seconds()

    with get_session() as session:
        run_repo = RunRepository(session)
        if error is not None:
            run_repo.complete(run_id, "failed", duration, error)
        else:
            run_repo.complete(run_id, "succeeded", duration, None)

    if error is not None:
        await bus.publish(
            RunCompletedEvent(
                run_id=run_id,
                model_id=model_id,
                status="failed",
                duration_seconds=duration,
                metrics=None,
                error=error,
                timestamp=finished,
            ).model_dump(mode="json")
        )
        return

    # Publish completion + model.created
    from schemas import MetricsOut

    metrics_payload = MetricsOut(
        accuracy=result["accuracy"],
        precision=result["precision"],
        recall=result["recall"],
        f1=result["f1"],
    )

    await bus.publish(
        RunCompletedEvent(
            run_id=run_id,
            model_id=model_id,
            status="succeeded",
            duration_seconds=duration,
            metrics=metrics_payload,
            error=None,
            timestamp=finished,
        ).model_dump(mode="json")
    )

    # Build a full ModelOut for the model.created event
    from schemas import ModelOut

    with get_session() as session:
        rec = ModelRepository(session).get(model_id)
        if rec is not None:
            cfg_obj = dict_to_pipeline_config(rec.config)
            model_out = ModelOut(
                id=rec.id,
                name=rec.name,
                algorithm=rec.algorithm,
                created_at=rec.created_at,
                metrics=metrics_payload,
                train_size=rec.train_size,
                test_size=rec.test_size,
                class_distribution=rec.class_distribution,
                config=_config_to_request(cfg_obj),
                artifact_path=rec.artifact_path,
            )
            await bus.publish(
                ModelCreatedEvent(
                    model=model_out,
                    timestamp=_utcnow(),
                ).model_dump(mode="json")
            )


def _train_blocking(
    run_id: str,
    model_id: str,
    cfg: PipelineConfig,
    artifact_path: str,
) -> dict[str, Any]:
    """Synchronous training routine executed in a worker thread."""

    def progress_cb(percent: float, stage: str) -> None:
        # Persist progress + publish event from the worker thread.
        with get_session() as session:
            RunRepository(session).update_progress(run_id, float(percent), stage)

        # Schedule the async publish on the main loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                bus.publish(
                    ProgressEvent(
                        run_id=run_id,
                        model_id=model_id,
                        percent=float(percent),
                        stage=stage,
                        timestamp=_utcnow(),
                    ).model_dump(mode="json")
                ),
                loop,
            )

    result = pipeline_train(DEFAULT_DATA_PATH, cfg, progress_cb=progress_cb)

    # Persist artifact
    save_model(result.model, Path(artifact_path))

    # Update model record with final metrics
    with get_session() as session:
        rec = ModelRepository(session).get(model_id)
        if rec is not None:
            rec.accuracy = result.metrics["accuracy"]
            rec.precision = result.metrics["precision"]
            rec.recall = result.metrics["recall"]
            rec.f1 = result.metrics["f1"]
            rec.train_size = result.train_size
            rec.test_size = result.test_size
            rec.class_distribution = result.class_distribution
            session.commit()

    return result.metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(model_id: str | None, texts: list[str]) -> tuple[str, list[dict[str, Any]]]:
    """Load a model artifact and run prediction. Returns (model_id, predictions)."""
    with get_session() as session:
        repo = ModelRepository(session)
        if model_id:
            rec = repo.get(model_id)
            if rec is None:
                raise LookupError(f"model {model_id} not found")
        else:
            # Default: most recent model
            recent = repo.list(limit=1)
            if not recent:
                raise LookupError("no trained models available — train one first")
            rec = recent[0]

        artifact = Path(rec.artifact_path)
        chosen_id = rec.id

    from pipeline import load_model

    model = load_model(artifact)
    preds = pipeline_predict(model, texts)
    return chosen_id, [
        {
            "text": p.text,
            "label": p.label,
            "label_int": p.label_int,
            "spam_probability": p.spam_probability,
        }
        for p in preds
    ]
