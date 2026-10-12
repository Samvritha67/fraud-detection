"""FastAPI backend for the PaySim fraud detection dashboard."""
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# The pipeline serializes features.add_features from this root module.
import features  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
THRESHOLD = 0.5

if not MODEL_PATH.is_file():
    raise RuntimeError(f"Trained model not found at {MODEL_PATH}. Run the notebook or `python train.py` first.")

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    raise RuntimeError(f"Could not load the trained model at {MODEL_PATH}: {exc}") from exc

try:
    metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.is_file() else {}
except (OSError, json.JSONDecodeError) as exc:
    raise RuntimeError(f"Could not read model metrics at {METRICS_PATH}: {exc}") from exc

history = deque(maxlen=50)
app = FastAPI(
    title="PayGuard | Payment Fraud Detection",
    description="Score payment transactions using the trained PaySim fraud detection model.",
    version="1.0.0",
)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


class Transaction(BaseModel):
    step: int = Field(ge=1, le=10000, description="Simulation time step")
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
    amount: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceOrg: float = Field(ge=0, allow_inf_nan=False)
    oldbalanceDest: float = Field(ge=0, allow_inf_nan=False)


def risk_level(probability: float) -> str:
    if probability >= 0.8:
        return "HIGH"
    if probability >= THRESHOLD:
        return "MEDIUM"
    if probability >= 0.2:
        return "LOW"
    return "VERY LOW"


def score(transactions: list[Transaction]) -> list[dict]:
    frame = pd.DataFrame([transaction.model_dump() for transaction in transactions])
    try:
        probabilities = model.predict_proba(frame)[:, 1]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model scoring failed: {exc}") from exc

    predictions = []
    for probability in probabilities:
        probability = float(probability)
        flagged = probability >= THRESHOLD
        predictions.append({
            "prediction": int(flagged),
            "label": "Fraudulent" if flagged else "Legitimate",
            "fraud_probability": round(probability, 6),
            "risk_level": risk_level(probability),
            "action": "Block & alert analyst" if flagged else "Approve & settle",
        })
    return predictions


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": metrics.get("best_model", "unknown")}


@app.get("/metrics")
def get_metrics():
    return {**metrics, "threshold": THRESHOLD}


@app.get("/history")
def get_history():
    return list(history)


@app.post("/predict")
def predict(transaction: Transaction):
    result = score([transaction])[0]
    history.appendleft({
        "time": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
        "type": transaction.type,
        "amount": transaction.amount,
        **result,
    })
    return result


@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    if not 1 <= len(transactions) <= 5000:
        raise HTTPException(status_code=400, detail="Send between 1 and 5000 transactions.")
    return score(transactions)
