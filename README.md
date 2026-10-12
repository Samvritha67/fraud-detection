# PayGuard — Payment Fraud Detection

A small full-stack PaySim demonstration: a responsive frontend, FastAPI backend, and leakage-aware fraud classification pipeline. The deployed scoring path accepts pre-authorization features only.

## Project layout

- `frontend/` — dashboard HTML, styles, and browser logic.
- `backend/main.py` — FastAPI routes and trained model inference.
- `models/best_model.joblib` — trained model loaded when the backend starts.
- `models/metrics.json` — model comparison and training metadata used by the dashboard.
- `features.py` — feature transformation required by the serialized model.
- `train.py` and `model_comparison.ipynb` — model training workflows.

## Run locally

Python 3.10+ is recommended. From this project directory:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000>. The same backend serves the static frontend and API, so no separate frontend build or CORS setup is needed. API documentation is at <http://127.0.0.1:8000/docs>.

To retrain, place the PaySim CSV in `data/PS_20174392719_1491204439457_log.csv` and run `python train.py`. Training draws a reproducible 1,000,000-row sample, creates stratified 70/15/15 train/validation/test partitions, compares Logistic Regression, Decision Tree, Random Forest, Extra Trees, and XGBoost under both pre-authorization and post-authorization features, and selects the deployed pre-authorization model by validation PR-AUC. The test set is used only for final reporting; threshold-based metrics use 0.5. It saves the model to `models/best_model.joblib` and validation/test results to `models/metrics.json`.

The CSV is not included in the repository. Retrain before starting the backend so the included model artifact is replaced with a pre-authorization model compatible with the current API inputs.

## API

- `GET /health` — backend and active model status.
- `GET /metrics` — saved holdout model comparison.
- `GET /history` — recent screenings in this server process.
- `POST /predict` — score one transaction.
- `POST /predict/batch` — score 1 to 5,000 transactions.
