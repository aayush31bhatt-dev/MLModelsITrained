# Release Notes — SMS Spam Classifier (Phase 3)

**Release:** v0.3.0 — Integration, Verification & Launch
**Audience:** Internal technical team (data scientists / ML engineers)
**Deployment target:** Local workstation or on-prem server (Linux/Windows)

---

## 1. What ships

| Surface | Entry point | Notes |
| --- | --- | --- |
| CLI baseline | `python train_spam_classifier.py` | Prints metrics + classification report, writes `artifacts/confusion_matrix.png`. |
| REST API | `uvicorn app:app --host 0.0.0.0 --port 8000` | FastAPI + Uvicorn. |
| WebSocket realtime | `ws://<host>:8000/ws/events` | `hello` → `progress` → `run.completed` → `model.created`. |
| Persistence | SQLite at `artifacts/spam_classifier.db` | Override via `DATABASE_URL`. |
| Model artifacts | `artifacts/model_<ts>.joblib` | joblib-serialized sklearn `Pipeline`. |

### API surface (stable contract)

- `GET  /health`
- `POST /models/train` → `202 Accepted` with `run_id` + `model_id`
- `GET  /models`, `GET /models/{model_id}`
- `POST /predict`
- `GET  /runs`, `GET /runs/{run_id}`
- `WS   /ws/events`

Full request/response schemas live in `schemas.py` (Pydantic v2).

---

## 2. Build & run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Train + evaluate (CLI baseline)
python train_spam_classifier.py

# 3. Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000
# or
python app.py
```

### Environment variables

| Var | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///./artifacts/spam_classifier.db` | SQLAlchemy engine URL. |

---

## 3. Success criteria (Phase 3 acceptance)

The baseline `MultinomialNB + TF-IDF (1-2 grams)` pipeline must meet:

| Metric | Threshold |
| --- | --- |
| accuracy  | ≥ 0.97 |
| precision | ≥ 0.95 |
| recall    | ≥ 0.90 |
| f1        | ≥ 0.92 |

Verify with:

```bash
python scripts/verify_success.py
```

Override thresholds via flags, e.g. `--min-f1 0.93`.

---

## 4. End-to-end integration pass

Run the full integration suite (CLI + pipeline + persistence + service + REST + WebSocket):

```bash
python tests/integration_test.py
```

Expected output: `N/N checks passed` and exit code `0`.

---

## 5. Release checklist

- [x] `pip install -r requirements.txt` succeeds on a clean venv.
- [x] `python train_spam_classifier.py` exits 0 and writes `artifacts/confusion_matrix.png`.
- [x] `uvicorn app:app` boots, `GET /health` returns `{"status": "ok", ...}`.
- [x] `POST /models/train` returns `202` and the run completes (`succeeded`).
- [x] `POST /predict` returns per-text `label` + `spam_probability`.
- [x] `WS /ws/events` delivers `hello`, `progress`, `run.completed`, `model.created`.
- [x] `python scripts/verify_success.py` reports `SUCCESS` against the thresholds above.
- [x] `python tests/integration_test.py` reports `N/N checks passed`.
- [x] SQLite DB and joblib artifacts are written under `artifacts/`.

---

## 6. Handoff notes

### Architecture (one-liner)
`app.py` (FastAPI routes + lifespan + CORS + WS) → `service.py` (orchestration) → `pipeline.py` (training/inference) → `db.py` (SQLAlchemy ORM + repositories). Realtime events flow through `events.py` (in-process asyncio pub/sub bus).

### Key files

- `train_spam_classifier.py` — single-file CLI baseline.
- `pipeline.py` — reusable training/inference (`PipelineConfig`, `train`, `predict`, `save_model`, `load_model`).
- `schemas.py` — Pydantic v2 request/response contracts.
- `db.py` — ORM models (`ModelRecord`, `RunRecord`) + repositories.
- `events.py` — `EventBus` singleton.
- `service.py` — `start_training`, `run_inference`, background job orchestration.
- `app.py` — FastAPI app, routes, WebSocket endpoint.

### Operational notes

- **Background training** runs in `loop.run_in_executor` so the event loop stays responsive; progress events stream live to `/ws/events`.
- **Inference** loads the most recent model by default; pass `model_id` to `POST /predict` to target a specific artifact.
- **Persistence** is SQLite by default for zero-ops on-prem. Switch to Postgres/MySQL by setting `DATABASE_URL`.
- **Concurrency** is single-process; the in-process event bus does not span multiple workers. If you scale out, replace `events.py` with Redis pub/sub or similar.

### Known limitations (Phase 3 scope)

- Single algorithm (`MultinomialNB`). Phase 4+ may add SVM / logistic regression / ensembles.
- No auth on the API. Local / on-prem technical-team use only.
- No model registry UI; use `GET /models` and `GET /runs`.

### Quick smoke test (after deploy)

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/models/train \
  -H 'content-type: application/json' \
  -d '{"name":"smoke","config":null}'
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"texts":["Free entry in our weekly contest!", "Hey, lunch at noon?"]}'
```

---

## 7. Sign-off

Phase 3 acceptance is met when:

1. `scripts/verify_success.py` exits `0`.
2. `tests/integration_test.py` exits `0`.
3. The release checklist above is fully ticked.
