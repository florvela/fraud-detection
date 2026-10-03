"""Servicio REST (FastAPI) del TP1: sirve el modelo de fraude.

Autocontenido (no depende del repo raíz): el modelo viaja en `model.joblib` al
lado de este archivo. Expone el contrato que documenta `docs/01_REST.md`:

  GET  /health         -> estado + versión del modelo          (pública)
  POST /v1/predict     -> predicción para una transacción      (X-API-KEY)
  GET  /v1/model-info  -> metadatos completos del modelo        (pública)
  POST /predict        -> igual que v1 pero sin auth, lo usa el cliente
                          comparador de latencia (client/client.py)
  GET  /docs           -> Swagger UI (lo da FastAPI)
"""

import os

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, ConfigDict, field_validator

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# El modelo se carga UNA sola vez al iniciar.
ARTIFACT = joblib.load(MODEL_PATH)
PIPELINE = ARTIFACT["pipeline"]
VERSION = ARTIFACT["version"]
FEATURE_ORDER = ARTIFACT["feature_order"]
METRICS = ARTIFACT.get("metrics", {"roc_auc": 0.0, "pr_auc": 0.0})
CATEGORIES = ARTIFACT.get("categories", {})
VALID_CATEGORIES = set(CATEGORIES.get("category", []))
VALID_GENDERS = set(CATEGORIES.get("gender", ["F", "M"]))

# --- Seguridad (API Key) --------------------------------------------------
# Los tokens válidos salen de la env API_KEYS (separados por coma). Sin definirla
# se usa el token de desarrollo. Nunca se hardcodea en el código.
AUTHORIZED_CLIENTS = set(os.getenv("API_KEYS", "token-secreto-123").split(","))
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


def validate_token(api_key: str = Security(api_key_header)) -> str:
    if api_key not in AUTHORIZED_CLIENTS:
        raise HTTPException(status_code=403, detail="Token ausente o inválido")
    return api_key


# --- Contrato de entrada --------------------------------------------------
class Transaction(BaseModel):
    # extra="forbid": campos desconocidos -> 422 (ver docs/01_REST.md).
    model_config = ConfigDict(extra="forbid")

    amt: float
    category: str
    gender: str
    city_pop: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    hour: int
    age: int

    @field_validator("amt")
    @classmethod
    def _amt_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amt debe ser > 0")
        return v

    @field_validator("category")
    @classmethod
    def _category_known(cls, v: str) -> str:
        if VALID_CATEGORIES and v not in VALID_CATEGORIES:
            raise ValueError(f"category inválida: {v}")
        return v

    @field_validator("gender")
    @classmethod
    def _gender_known(cls, v: str) -> str:
        if v not in VALID_GENDERS:
            raise ValueError("gender debe ser F o M")
        return v

    @field_validator("hour")
    @classmethod
    def _hour_range(cls, v: int) -> int:
        if not 0 <= v <= 23:
            raise ValueError("hour fuera de rango 0-23")
        return v

    @field_validator("age")
    @classmethod
    def _age_range(cls, v: int) -> int:
        if not 0 <= v <= 120:
            raise ValueError("age fuera de rango 0-120")
        return v


app = FastAPI(title="TP1 - Fraud REST service")


def _score(tx: Transaction) -> dict:
    row = {name: getattr(tx, name) for name in FEATURE_ORDER}
    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    probability = float(PIPELINE.predict_proba(X)[0][1])
    is_fraud = bool(PIPELINE.predict(X)[0])
    return {
        "is_fraud": is_fraud,
        "probability": round(probability, 4),
        "model_version": VERSION,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_version": VERSION}


@app.post("/v1/predict")
def predict_v1(tx: Transaction, _: str = Depends(validate_token)):
    return _score(tx)


@app.get("/v1/model-info")
def model_info():
    """Metadatos completos del modelo (REST devuelve todo -> over-fetching)."""
    return {
        "name": "fraud-detection",
        "version": VERSION,
        "trained_at": ARTIFACT.get("trained_at"),
        "metrics": METRICS,
        "feature_order": FEATURE_ORDER,
        "categories": CATEGORIES,
    }


@app.post("/predict")
def predict(tx: Transaction):
    """Sin auth ni versión: lo usa el cliente comparador de latencia."""
    return _score(tx)


@app.get("/")
def read_root():
    return {"message": "REST ML Service is running. Post to /predict o /v1/predict"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
