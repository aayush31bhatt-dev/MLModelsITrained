"""
Pydantic schemas (API contract) for the SMS Spam Classifier backend.

These define the wire format for every REST endpoint and WebSocket event.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------
class PipelineConfigIn(BaseModel):
    """Hyperparameters accepted by POST /models/train."""

    ngram_min: int = Field(1, ge=1, le=5)
    ngram_max: int = Field(2, ge=1, le=5)
    min_df: int = Field(2, ge=1)
    sublinear_tf: bool = True
    alpha: float = Field(0.1, gt=0.0)
    test_size: float = Field(0.2, gt=0.0, lt=1.0)
    random_state: int = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class ClassMetricsOut(BaseModel):
    precision: float
    recall: float
    f1: float
    support: int


class MetricsOut(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    algorithm: str
    created_at: datetime
    metrics: MetricsOut
    train_size: int
    test_size: int
    class_distribution: dict[str, int]
    config: PipelineConfigIn
    artifact_path: str


class ModelSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    algorithm: str
    created_at: datetime
    metrics: MetricsOut


class TrainRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    config: PipelineConfigIn | None = None


class TrainAccepted(BaseModel):
    run_id: str
    model_id: str
    status: Literal["queued", "running"] = "queued"
    message: str = "training started"


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
class RunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    model_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    started_at: datetime
    finished_at: datetime | None
    duration_seconds: float | None
    progress_percent: float
    stage: str
    error: str | None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=1000)
    model_id: str | None = None


class PredictionOut(BaseModel):
    text: str
    label: Literal["ham", "spam"]
    label_int: Literal[0, 1]
    spam_probability: float


class PredictResponse(BaseModel):
    model_id: str
    predictions: list[PredictionOut]


# ---------------------------------------------------------------------------
# Realtime events (WebSocket payloads)
# ---------------------------------------------------------------------------
class ProgressEvent(BaseModel):
    type: Literal["progress"] = "progress"
    run_id: str
    model_id: str
    percent: float
    stage: str
    timestamp: datetime


class RunCompletedEvent(BaseModel):
    type: Literal["run.completed"] = "run.completed"
    run_id: str
    model_id: str
    status: Literal["succeeded", "failed"]
    duration_seconds: float | None
    metrics: MetricsOut | None
    error: str | None
    timestamp: datetime


class ModelCreatedEvent(BaseModel):
    type: Literal["model.created"] = "model.created"
    model: ModelOut
    timestamp: datetime


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    models_loaded: int
    runs_total: int
