from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from fraud_mlops.monitoring_metrics import (
    MISSING_VALUE_RATIO,
    PREDICTION_CONFIDENCE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)

ARTIFACT_PATH = Path("artifacts/best_model.joblib")


class FraudRequest(BaseModel):
    records: list[dict]


app = FastAPI(title="Fraud Detection API")


def load_model():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError("Model artifact not found. Train the model first.")
    return joblib.load(ARTIFACT_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict")
def predict(payload: FraudRequest):
    start = time.perf_counter()
    try:
        model_bundle = load_model()
        frame = pd.DataFrame(payload.records)
        frame = frame.reindex(columns=model_bundle["feature_names"], fill_value=None)
        MISSING_VALUE_RATIO.set(float(frame.isna().mean().mean()))
        transformed = model_bundle["preprocessor"].transform(frame)
        selected = model_bundle["selector"].transform(transformed)
        probabilities = model_bundle["model"].predict_proba(selected)[:, 1]
        predictions = (probabilities >= model_bundle["threshold"]).astype(int)
        for probability in probabilities:
            PREDICTION_CONFIDENCE.observe(float(probability))
        return_payload = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }
        REQUEST_COUNT.labels("/predict", "POST", "200").inc()
        return return_payload
    except FileNotFoundError as exc:
        REQUEST_COUNT.labels("/predict", "POST", "500").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.labels("/predict").observe(time.perf_counter() - start)
