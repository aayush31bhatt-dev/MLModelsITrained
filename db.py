"""
Persistence layer: SQLAlchemy ORM models, engine/session factory, and
repositories for `models` and `runs`.

Uses SQLite by default for zero-ops on-prem deployment. The engine URL is
configurable via the `DATABASE_URL` environment variable.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)


# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------
DEFAULT_DB_PATH = Path(__file__).parent / "artifacts" / "spam_classifier.db"
DEFAULT_DB_URL = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


def make_engine(url: str | None = None):
    url = url or get_database_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = make_engine()
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=get_engine(), autoflush=False, autocommit=False, expire_on_commit=False
        )
    return _SessionLocal


def init_db() -> None:
    """Create tables. Idempotent — safe to call on every startup."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    factory = get_session_factory()
    return factory()


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ModelRecord(Base):
    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(60), nullable=False, default="MultinomialNB")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Metrics
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision: Mapped[float] = mapped_column(Float, nullable=False)
    recall: Mapped[float] = mapped_column(Float, nullable=False)
    f1: Mapped[float] = mapped_column(Float, nullable=False)

    # Dataset / split info
    train_size: Mapped[int] = mapped_column(Integer, nullable=False)
    test_size: Mapped[int] = mapped_column(Integer, nullable=False)
    class_distribution: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Pipeline config (stored as JSON for forward-compat)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    # Artifact on disk
    artifact_path: Mapped[str] = mapped_column(String(400), nullable=False)


class RunRecord(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued")
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    progress_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage: Mapped[str] = mapped_column(String(60), nullable=False, default="queued")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------
class ModelRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, record: ModelRecord) -> ModelRecord:
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get(self, model_id: str) -> ModelRecord | None:
        return self.session.get(ModelRecord, model_id)

    def list(self, limit: int = 100, offset: int = 0) -> list[ModelRecord]:
        stmt = select(ModelRecord).order_by(ModelRecord.created_at.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def count(self) -> int:
        stmt = select(ModelRecord)
        return len(list(self.session.scalars(stmt)))


class RunRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, record: RunRecord) -> RunRecord:
        self.session.add(record)
        self.session.commit()
        self.session.refresh(record)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        return self.session.get(RunRecord, run_id)

    def list(self, limit: int = 100, offset: int = 0) -> list[RunRecord]:
        stmt = select(RunRecord).order_by(RunRecord.started_at.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt))

    def list_for_model(self, model_id: str) -> list[RunRecord]:
        stmt = (
            select(RunRecord)
            .where(RunRecord.model_id == model_id)
            .order_by(RunRecord.started_at.desc())
        )
        return list(self.session.scalars(stmt))

    def update_progress(self, run_id: str, percent: float, stage: str) -> None:
        rec = self.session.get(RunRecord, run_id)
        if rec is None:
            return
        rec.progress_percent = percent
        rec.stage = stage
        rec.status = "running"
        self.session.commit()

    def complete(self, run_id: str, status: str, duration: float | None, error: str | None) -> None:
        rec = self.session.get(RunRecord, run_id)
        if rec is None:
            return
        rec.status = status
        rec.finished_at = _utcnow()
        rec.duration_seconds = duration
        rec.error = error
        rec.progress_percent = 100.0 if status == "succeeded" else rec.progress_percent
        self.session.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def config_to_dict(cfg) -> dict:
    """Serialize a PipelineConfig (dataclass) to a JSON-safe dict."""
    if cfg is None:
        return {}
    if hasattr(cfg, "__dict__"):
        d = dict(cfg.__dict__)
        # tuple -> list for JSON
        if "ngram_range" in d and isinstance(d["ngram_range"], tuple):
            d["ngram_range"] = list(d["ngram_range"])
        return d
    if isinstance(cfg, dict):
        return cfg
    return json.loads(json.dumps(cfg, default=str))


def dict_to_pipeline_config(d: dict | None):
    """Convert a stored config dict back into a PipelineConfig."""
    from pipeline import PipelineConfig  # local import to avoid cycle

    if not d:
        return PipelineConfig()
    ngram = d.get("ngram_range") or (1, 2)
    if isinstance(ngram, list):
        ngram = tuple(ngram)
    return PipelineConfig(
        ngram_range=ngram,
        min_df=int(d.get("min_df", 2)),
        sublinear_tf=bool(d.get("sublinear_tf", True)),
        alpha=float(d.get("alpha", 0.1)),
        test_size=float(d.get("test_size", 0.2)),
        random_state=int(d.get("random_state", 42)),
    )
