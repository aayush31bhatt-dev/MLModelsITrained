# SMS Spam Classifier

A multi-model SMS spam classification system comparing **Multinomial Naive Bayes**, **Complement Naive Bayes**, and **Logistic Regression** — all using TF-IDF feature extraction. Includes a FastAPI backend with WebSocket realtime events, SQLite persistence, and a minimal model report dashboard.

> **Release:** v0.3.0 — see [RELEASE.md](./RELEASE.md) for the launch package, success criteria, integration test, and handoff notes.

---

## Models

| # | Algorithm | Pipeline |
|---|-----------|----------|
| 1 | Multinomial Naive Bayes | TF-IDF → `MultinomialNB(alpha=0.1)` |
| 2 | Complement Naive Bayes | TF-IDF → `ComplementNB(alpha=0.1)` |
| 3 | Logistic Regression | TF-IDF → `LogisticRegression(max_iter=1000)` |

All models share the same TF-IDF vectorizer config: `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True`.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train all 3 models + generate artifacts
python train_all_models.py

# Train baseline only (Multinomial NB)
python train_spam_classifier.py

# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Artifacts Generated

After running `train_all_models.py`, the `artifacts/` directory contains:

| File | Description |
|------|-------------|
| `all_models_report.json` | Full metrics, per-class stats, and ROC data for all models |
| `confusion_matrix_multinomial_nb.png` | Confusion matrix — Multinomial NB |
| `confusion_matrix_complement_nb.png` | Confusion matrix — Complement NB |
| `confusion_matrix_logistic_regression.png` | Confusion matrix — Logistic Regression |
| `roc_curve_multinomial_nb.png` | ROC curve — Multinomial NB |
| `roc_curve_complement_nb.png` | ROC curve — Complement NB |
| `roc_curve_logistic_regression.png` | ROC curve — Logistic Regression |
| `roc_curve_all_models.png` | Combined ROC curves (all models) |
| `model_<algo>_<timestamp>.joblib` | Serialized fitted pipeline for each model |

---

## Model Report (Dashboard)

Open `index.html` in a browser to view an interactive model report page featuring:

- Dataset statistics (total samples, class distribution, train/test split)
- Per-model metrics: accuracy, precision, recall, F1, ROC AUC
- Per-class breakdown (ham vs spam)
- Confusion matrix visualizations
- ROC curve plots (per-model + combined comparison)

The report page reads from the embedded JSON snapshot and references images in `artifacts/`.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/models/train` | Start a training run (returns 202 with `run_id`) |
| `GET` | `/models` | List all trained models |
| `GET` | `/models/{model_id}` | Get a specific model's details |
| `POST` | `/predict` | Classify SMS texts (accepts `texts[]` + optional `model_id`) |
| `GET` | `/runs` | List training runs |
| `GET` | `/runs/{run_id}` | Get a specific run's status |
| `WS` | `/ws/events` | Realtime event stream (progress, run.completed, model.created) |

---

## Verify the Release

```bash
# Success-criteria check (accuracy/precision/recall/F1 thresholds)
python scripts/verify_success.py

# End-to-end integration pass (CLI + REST + WS + persistence)
python tests/integration_test.py
```

---

## Project Layout

```
├── train_all_models.py       # Train all 3 models, generate plots + report JSON
├── train_spam_classifier.py  # CLI baseline (Multinomial NB only)
├── pipeline.py               # Reusable training/inference pipeline
├── app.py                    # FastAPI routes, lifespan, CORS, WebSocket
├── service.py                # Orchestration layer
├── schemas.py                # Pydantic v2 API contracts
├── db.py                     # SQLAlchemy ORM + repositories
├── events.py                 # In-process async pub/sub bus
├── index.html                # Static model report dashboard
├── requirements.txt          # Python dependencies
├── RELEASE.md                # Release notes & handoff
├── spam.csv                  # SMS Spam Collection dataset
├── artifacts/                # Generated models, plots, and reports
├── scripts/
│   └── verify_success.py     # Success-criteria verification
└── tests/
    └── integration_test.py   # End-to-end integration test
```

---

## Dataset

- **Source:** SMS Spam Collection (5,572 messages)
- **Encoding:** latin-1
- **Labels:** `ham` → 0, `spam` → 1
- **Split:** 80% train / 20% test (stratified, `random_state=42`)

---

## Tech Stack

- **ML:** scikit-learn (TF-IDF + classifiers)
- **Data:** pandas, numpy
- **Visualization:** matplotlib
- **API:** FastAPI + Uvicorn
- **Validation:** Pydantic v2
- **Persistence:** SQLAlchemy 2.x (SQLite)
- **Realtime:** WebSockets + asyncio pub/sub
- **Serialization:** joblib
