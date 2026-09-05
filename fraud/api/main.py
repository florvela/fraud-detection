"""API REST que sirve el modelo de fraude: /v1/predict y /health"""

from __future__ import annotations

import os
from enum import Enum

import pandas as pd
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from strawberry.fastapi import GraphQLRouter

from fraud.api.graphql_schema import schema as graphql_schema
from fraud.api.model_loader import ModelStore
from fraud.api.schemas import HealthResponse, PredictionResponse, Transaction

store = ModelStore()
store.load()

app = FastAPI(
    title="Fraud Detection API",
    version=store.version if store.loaded else "0.0.0",
)

api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)
_valid_tokens = os.getenv("API_KEYS", "token-secreto-123")
AUTHORIZED_CLIENTS = {t.strip() for t in _valid_tokens.split(",") if t.strip()}


def validate_token(api_key: str = Security(api_key_header)) -> str:
    if api_key not in AUTHORIZED_CLIENTS:
        raise HTTPException(status_code=403, detail="API Key inválida o ausente")
    return api_key


@app.get("/health", response_model=HealthResponse, tags=["Infra"])
def health() -> HealthResponse:
    status = "ok" if store.loaded else "model_not_loaded"
    version = store.version if store.loaded else "unknown"
    return HealthResponse(status=status, model_version=version)


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Model v1"])
def predict(
    transaction: Transaction,
    _client: str = Security(validate_token),
) -> PredictionResponse:
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

@app.get("/v1/model-info", tags=["Model v1"])
def model_info() -> dict:
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"name": "fraud-detection", **store.info}


app.include_router(GraphQLRouter(graphql_schema), prefix="/graphql")