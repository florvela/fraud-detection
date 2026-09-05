"""API REST que sirve el modelo de fraude: /v1/predict y /health"""

from __future__ import annotations

from enum import Enum

import pandas as pd
from fastapi import FastAPI, HTTPException

from fraud.api.model_loader import ModelStore
from fraud.api.schemas import HealthResponse, PredictionResponse, Transaction

store = ModelStore()
store.load()

app = FastAPI(
    title="Fraud Detection API",
    version=store.version if store.loaded else "0.0.0",
)


@app.get("/health", response_model=HealthResponse, tags=["Infra"])
def health() -> HealthResponse:
    status = "ok" if store.loaded else "model_not_loaded"
    version = store.version if store.loaded else "unknown"
    return HealthResponse(status=status, model_version=version)


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Model v1"])
def predict(transaction: Transaction) -> PredictionResponse:
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")

    # DataFrame de una fila , respetando el orden de columnas del modelo
    row = {name: getattr(transaction, name) for name in store.feature_order}
    row = {k: (v.value if isinstance(v, Enum) else v) for k, v in row.items()}
    X = pd.DataFrame([row], columns=store.feature_order)

    try:
        probability = float(store.pipeline.predict_proba(X)[0][1])
        is_fraud = bool(store.pipeline.predict(X)[0])
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {err}")

    return PredictionResponse(
        is_fraud=is_fraud,
        probability=round(probability, 4),
        model_version=store.version,
    )
