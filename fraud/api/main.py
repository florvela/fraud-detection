"""API REST que sirve el modelo de fraude: /v1/predict y /health"""

from __future__ import annotations

from enum import Enum
import os

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from fraud.api.model_loader import ModelStore
from fraud.api.schemas import (
    DecisionRequest,
    HealthResponse,
    PredictionResponse,
    Transaction,
)

store = ModelStore()
store.load()

app = FastAPI(
    title="Fraud Detection API",
    version=store.version if store.loaded else "0.0.0",
)

# Métricas Prometheus en /metrics (para el monitoreo de latencia/throughput).
# Tolerante: si la lib no está instalada (dev local), simplemente no expone /metrics.
try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
except Exception:
    pass

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


FEATURE_ORDER = [
    "amt", "category", "gender", "city_pop", "lat",
    "long", "merch_lat", "merch_long", "hour", "age",
]


@app.post("/v1/predict", response_model=PredictionResponse, tags=["Model v1"])
def predict(
    transaction: Transaction,
    _client: str = Security(validate_token),
) -> PredictionResponse:
    # Normaliza la transacción a un dict plano (resuelve los Enum de Pydantic)
    feature_order = store.feature_order if store.loaded else FEATURE_ORDER
    row = {name: getattr(transaction, name) for name in feature_order}
    row = {k: (v.value if isinstance(v, Enum) else v) for k, v in row.items()}

    # El champion decide y el challenger (si hay) corre en sombra; todo se registra.
    from fraud.api.scoring import get_scorer

    try:
        result = get_scorer().score(row)
    except Exception as err:  # noqa: BLE001 - cualquier fallo del scoring -> 502
        raise HTTPException(status_code=502, detail=f"No se pudo puntuar la transacción: {err}")

    return PredictionResponse(**result)


@app.get("/v1/model-info", tags=["Model v1"])
def model_info() -> dict:
    if not store.loaded:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    return {"name": "fraud-detection", **store.info}


# --------------------------------------------------------------------------- #
# Endpoints del analista (UC2 revisión · UC3 denuncia post-mortem)
# --------------------------------------------------------------------------- #

@app.get("/v1/reviews", tags=["Analista"])
def list_reviews(
    status: str = "PENDING",
    limit: int = 100,
    _client: str = Security(validate_token),
) -> dict:
    """Lista las transacciones en un estado dado (por defecto las retenidas)."""
    from fraud.api.store import get_store

    items = get_store().list_by_status(status=status, limit=limit)
    return {"status": status, "count": len(items), "items": items}


@app.get("/v1/transactions/{transaction_id}", tags=["Analista"])
def get_transaction(
    transaction_id: str,
    _client: str = Security(validate_token),
) -> dict:
    """Detalle de una transacción (raw + score champion + challenger + label)."""
    from fraud.api.store import get_store

    tx = get_store().get(transaction_id)
    if tx is None:
        raise HTTPException(status_code=404, detail="Transacción inexistente")
    return tx


@app.post("/v1/transactions/{transaction_id}/decision", tags=["Analista"])
def decide_transaction(
    transaction_id: str,
    body: DecisionRequest,
    _client: str = Security(validate_token),
) -> dict:
    """Resuelve una tx retenida: approve (legítima) o reject (fraude). Guarda label."""
    from fraud.api.store import get_store

    ok = get_store().resolve(transaction_id, body.decision)
    if not ok:
        raise HTTPException(status_code=404, detail="Transacción inexistente")
    return {"transaction_id": transaction_id, "decision": body.decision, "status": "resolved"}


@app.post("/v1/transactions/{transaction_id}/report-fraud", status_code=201, tags=["Analista"])
def report_fraud(
    transaction_id: str,
    _client: str = Security(validate_token),
) -> dict:
    """Denuncia post-mortem (UC3): pega un label tardío de fraude por transaction_id."""
    from fraud.api.store import SOURCE_POST_MORTEM, get_store

    s = get_store()
    if not s.exists(transaction_id):
        raise HTTPException(status_code=404, detail="Transacción inexistente")
    s.add_label(transaction_id, label=1, source=SOURCE_POST_MORTEM)
    return {"transaction_id": transaction_id, "label": 1, "source": SOURCE_POST_MORTEM}


# --------------------------------------------------------------------------- #
# Endpoints del Ingeniero MLOps (UC5 evaluar · UC6 deploy · UC4 entrenar)
# --------------------------------------------------------------------------- #

@app.post("/v1/mlops/evaluate", tags=["MLOps"])
def mlops_evaluate(_client: str = Security(validate_token)) -> dict:
    """Compara challenger vs champion contra la ground truth acumulada (UC5)."""
    from fraud.api import evaluation

    return evaluation.evaluate()


@app.post("/v1/mlops/deploy", tags=["MLOps"])
def mlops_deploy(
    action: str = "promote",
    _client: str = Security(validate_token),
) -> dict:
    """Promueve el challenger a champion (o revierte): action = promote | rollback (UC6)."""
    from fraud.api import deploy

    try:
        if action == "promote":
            result = deploy.promote()
        elif action == "rollback":
            result = deploy.rollback()
        else:
            raise HTTPException(status_code=422, detail="action debe ser promote|rollback")
    except FileNotFoundError as err:
        raise HTTPException(status_code=409, detail=str(err))
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Deploy falló: {err}")

    # El serving ya recargó el Scorer; recargamos también el store de /model-info.
    store.reload()
    return result


@app.post("/v1/mlops/train", status_code=202, tags=["MLOps"])
def mlops_train(_client: str = Security(validate_token)) -> dict:
    """Dispara el reentrenamiento (UC4).

    Si hay un Airflow con API REST configurado (`AIRFLOW_API_URL`), dispara el DAG
    `fraud_pipeline`; si no, devuelve la instrucción equivalente para correrlo.
    """
    airflow_url = os.getenv("AIRFLOW_API_URL")
    dag_id = os.getenv("AIRFLOW_DAG_ID", "fraud_pipeline")
    if airflow_url:
        import requests

        try:
            resp = requests.post(
                f"{airflow_url.rstrip('/')}/dags/{dag_id}/dagRuns",
                json={"conf": {}},
                auth=(os.getenv("AIRFLOW_USER", "airflow"), os.getenv("AIRFLOW_PASSWORD", "airflow")),
                timeout=10,
            )
            resp.raise_for_status()
            return {"triggered": True, "dag_id": dag_id, "airflow_response": resp.json()}
        except Exception as err:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"No se pudo disparar el DAG: {err}")
    return {
        "triggered": False,
        "dag_id": dag_id,
        "hint": "Corré el pipeline con `airflow dags trigger fraud_pipeline` "
        "o localmente `python -m fraud.dataset && python -m fraud.features && "
        "python -m fraud.modeling.train`.",
    }


