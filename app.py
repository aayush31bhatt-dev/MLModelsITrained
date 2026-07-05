"""
FastAPI application — Phase 2 Backend Core.

Exposes the agreed API contract:

  GET    /health
  POST   /models/train          -> kick off a background training run
  GET    /models                -> list trained models
  GET    /models/{model_id}     -> model detail
  POST   /predict               -> run inference
  GET    /runs                  -> list training runs
  GET    /runs/{run_id}         -> run detail
  WS     /ws/events             -> realtime progress + run/model events

Run with:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware

from db import (
    ModelRepository,
    RunRepository,
    dict_to_pipeline_config,
    get_session,
    init_db,
)
from events import bus
from schemas import (
    HealthOut,
    MetricsOut,
    ModelOut,
    ModelSummary,
    PredictRequest,
    PredictResponse,
    PredictionOut,
    RunOut,
    TrainAccepted,
    TrainRequest,
)
from service import _config_to_request, run_inference, start_training

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

API_VERSION = "0.2.0"


# ---------------------------------------------------------------------------
# Lifespan: initialize DB on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("database initialized")
    yield


app = FastAPI(
    title="SMS Spam Classifier API",
    version=API_VERSION,
    description="Phase 2 backend: train, persist, predict, and stream events.",
    lifespan=lifespan,
)

# Permissive CORS for local dev / on-prem technical users.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_to_out(rec) -> ModelOut:
    cfg = dict_to_pipeline_config(rec.config)
    return ModelOut(
        id=rec.id,
        name=rec.name,
        algorithm=rec.algorithm,
        created_at=rec.created_at,
        metrics=MetricsOut(
            accuracy=rec.accuracy,
            precision=rec.precision,
            recall=rec.recall,
            f1=rec.f1,
        ),
        train_size=rec.train_size,
        test_size=rec.test_size,
        class_distribution=rec.class_distribution,
        config=_config_to_request(cfg),
        artifact_path=rec.artifact_path,
    )


def _model_to_summary(rec) -> ModelSummary:
    return ModelSummary(
        id=rec.id,
        name=rec.name,
        algorithm=rec.algorithm,
        created_at=rec.created_at,
        metrics=MetricsOut(
            accuracy=rec.accuracy,
            precision=rec.precision,
            recall=rec.recall,
            f1=rec.f1,
        ),
    )


def _run_to_out(rec) -> RunOut:
    return RunOut(
        id=rec.id,
        model_id=rec.model_id,
        status=rec.status,
        started_at=rec.started_at,
        finished_at=rec.finished_at,
        duration_seconds=rec.duration_seconds,
        progress_percent=rec.progress_percent,
        stage=rec.stage,
        error=rec.error,
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthOut, tags=["meta"])
def health() -> HealthOut:
    with get_session() as session:
        models = ModelRepository(session).count()
        runs = len(RunRepository(session).list(limit=10000))
    return HealthOut(status="ok", version=API_VERSION, models_loaded=models, runs_total=runs)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@app.post("/models/train", response_model=TrainAccepted, status_code=202, tags=["models"])
def train_model(req: TrainRequest) -> TrainAccepted:
    try:
        run_id, model_id = start_training(req.name, req.config)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return TrainAccepted(run_id=run_id, model_id=model_id, status="running")


@app.get("/models", response_model=list[ModelSummary], tags=["models"])
def list_models(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[ModelSummary]:
    with get_session() as session:
        recs = ModelRepository(session).list(limit=limit, offset=offset)
        return [_model_to_summary(r) for r in recs]


@app.get("/models/{model_id}", response_model=ModelOut, tags=["models"])
def get_model(model_id: str) -> ModelOut:
    with get_session() as session:
        rec = ModelRepository(session).get(model_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"model {model_id} not found")
        return _model_to_out(rec)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    try:
        chosen_id, preds = run_inference(req.model_id, req.texts)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return PredictResponse(
        model_id=chosen_id,
        predictions=[PredictionOut(**p) for p in preds],
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
@app.get("/runs", response_model=list[RunOut], tags=["runs"])
def list_runs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    model_id: str | None = Query(None),
) -> list[RunOut]:
    with get_session() as session:
        repo = RunRepository(session)
        if model_id:
            recs = repo.list_for_model(model_id)
        else:
            recs = repo.list(limit=limit, offset=offset)
        return [_run_to_out(r) for r in recs]


@app.get("/runs/{run_id}", response_model=RunOut, tags=["runs"])
def get_run(run_id: str) -> RunOut:
    with get_session() as session:
        rec = RunRepository(session).get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"run {run_id} not found")
        return _run_to_out(rec)


# ---------------------------------------------------------------------------
# Realtime: WebSocket event stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """Stream realtime events to clients.

    Event types:
      - progress       (training progress updates)
      - run.completed  (training finished, succeeded or failed)
      - model.created  (new model persisted)
    """
    await websocket.accept()
    queue = await bus.subscribe()
    logger.info("ws: client connected (subscribers=%d)", bus.subscriber_count)
    try:
        # Send a hello frame so clients can confirm the channel is live.
        await websocket.send_text(
            json.dumps(
                {
                    "type": "hello",
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "api_version": API_VERSION,
                }
            )
        )
        while True:
            event = await queue.get()
            await websocket.send_text(json.dumps(event, default=str))
    except WebSocketDisconnect:
        logger.info("ws: client disconnected")
    except Exception:  # noqa: BLE001
        logger.exception("ws: unexpected error")
    finally:
        await bus.unsubscribe(queue)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
