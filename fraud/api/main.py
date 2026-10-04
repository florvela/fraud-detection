"""API REST que sirve el modelo de fraude: /v1/predict y /health"""

from __future__ import annotations

from enum import Enum
import os

from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from loguru import logger

from fraud.api.model_loader import ModelStore
from fraud.api.schemas import (
    DecisionRequest,
    HealthResponse,
    PredictionResponse,
    Transaction,
)

# Orden de features canónico. Lo tomamos de `federated_model` (módulo liviano,
# solo numpy/pandas) y NO de `fraud.features`, que arrastra typer/sklearn del
# pipeline de datos y no está instalado en la imagen de serving.
from fraud.federated_model import FEATURE_ORDER

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
except Exception as exc:  # noqa: BLE001 - /metrics es opcional (p.ej. dev local sin la lib)
    logger.debug(f"Prometheus instrumentator no disponible, sigo sin /metrics: {exc}")

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


# Tipado nativo por columna para serializar a JSON/SQLite sin tipos numpy.
_INT_COLS = {"city_pop", "hour", "age"}
_FLOAT_COLS = {"amt", "lat", "long", "merch_lat", "merch_long"}


def _coerce_row(raw: dict) -> dict:
    """Normaliza una fila cruda a tipos nativos de Python en el orden de features."""
    row: dict = {}
    for col in FEATURE_ORDER:
        val = raw[col]
        if col in _INT_COLS:
            row[col] = int(val)
        elif col in _FLOAT_COLS:
            row[col] = float(val)
        else:
            row[col] = str(val)
    return row


def _load_seed_frauds(n: int, seed: int) -> tuple[list[dict], str]:
    """Devuelve `n` fraudes para sembrar y la fuente usada ('dataset' | 'fallback').

    Prioriza casos REALES del split de test (`data/processed/test.parquet`); si no está
    disponible, cae al archivo curado `seed_frauds.json`.
    """
    from fraud.config import PROCESSED_DATA_DIR

    test_path = PROCESSED_DATA_DIR / "test.parquet"
    if test_path.exists():
        import pandas as pd

        df = pd.read_parquet(test_path)
        frauds = df[df["is_fraud"] == 1]
        if len(frauds) > 0:
            take = min(n, len(frauds))
            sample = frauds.sample(n=take, random_state=seed)
            return [_coerce_row(r) for r in sample[FEATURE_ORDER].to_dict("records")], "dataset"

    import json
    from pathlib import Path

    seed_path = Path(__file__).parent / "seed_frauds.json"
    rows = json.loads(seed_path.read_text())[:n]
    return [_coerce_row(r) for r in rows], "fallback"


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
    # El champion que REALMENTE decide es el del núcleo gRPC (si hay delegación);
    # lo exponemos aparte para que la UI/health no mientan si el store de REST difiere.
    from fraud.api import grpc_client

    serving = grpc_client.get_model_info()
    serving_champion_version = serving[1] if serving else None
    return {
        "name": "fraud-detection",
        **store.info,
        "serving_champion_version": serving_champion_version,
    }


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


@app.post("/v1/mlops/compare-federated", tags=["MLOps"])
def mlops_compare_federated(
    threshold: float = 0.5,
    _client: str = Security(validate_token),
) -> dict:
    """Benchmark offline sobre el test: **federado vs champion (centralizado)**.

    Responde la pregunta del TP: ¿cuánto del PR-AUC del modelo centralizado recupera el
    federado SIN que los bancos compartan datos? Es un benchmark a demanda (NO parte del
    reentrenamiento): evalúa ambos modelos sobre `data/processed/test.parquet` y devuelve
    PR-AUC / ROC-AUC / recall de cada uno + el % recuperado.
    """
    from fraud.api.model_loader import make_champion, make_federated
    from fraud.config import PROCESSED_DATA_DIR

    test_path = PROCESSED_DATA_DIR / "test.parquet"
    if not test_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe {test_path} para el benchmark")

    federated = make_federated()
    if not federated.load():
        raise HTTPException(
            status_code=409,
            detail="No hay modelo federado activo; corré el DAG fraud_federate_pipeline primero.",
        )
    champion = make_champion()
    champion.load()

    import pandas as pd
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

    df = pd.read_parquet(test_path)
    y = df["is_fraud"].to_numpy(dtype=int)

    def _metrics(store: ModelStore) -> dict:
        proba = store.pipeline.predict_proba(df[store.feature_order])[:, 1]
        pred = (proba >= threshold).astype(int)
        return {
            "model_version": store.version,
            "pr_auc": round(float(average_precision_score(y, proba)), 4),
            "roc_auc": round(float(roc_auc_score(y, proba)), 4),
            "recall_fraude": round(float(recall_score(y, pred, zero_division=0)), 4),
        }

    centralizado = _metrics(champion)
    federado = _metrics(federated)
    pct = (
        round(federado["pr_auc"] / centralizado["pr_auc"] * 100, 1)
        if centralizado["pr_auc"] > 0
        else None
    )
    return {
        "n_test": len(y),
        "fraude_test": round(float(y.mean()), 5),
        "centralizado": centralizado,
        "federado": federado,
        "pct_pr_auc_recuperado": pct,
    }


@app.post("/v1/mlops/deploy", tags=["MLOps"])
def mlops_deploy(
    action: str = "promote",
    source: str = "challenger",
    _client: str = Security(validate_token),
) -> dict:
    """Promueve un modelo en sombra a champion (o revierte) — UC6.

    action = promote | rollback · source = challenger | federated (qué sombra promover).
    """
    from fraud.api import deploy

    try:
        if action == "promote":
            result = deploy.promote(source=source)
        elif action == "rollback":
            result = deploy.rollback()
        else:
            raise HTTPException(status_code=422, detail="action debe ser promote|rollback")
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))
    except FileNotFoundError as err:
        raise HTTPException(status_code=409, detail=str(err))
    except HTTPException:
        raise
    except Exception as err:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Deploy falló: {err}")

    # El serving ya recargó el Scorer; recargamos también el store de /model-info.
    store.reload()
    return result


@app.post("/v1/mlops/reload", tags=["MLOps"])
def mlops_reload(_client: str = Security(validate_token)) -> dict:
    """Recarga champion + challenger en el serving tras un retrain, SIN reiniciar.

    Resetea el Scorer in-process, recarga el store de /model-info y, si el scoring
    se delega al núcleo gRPC, le pide que recargue su champion también.
    """
    from fraud.api import grpc_client
    from fraud.api.scoring import get_scorer, reset_scorer

    reset_scorer()
    store.reload()
    grpc_version = grpc_client.reload_remote()
    scorer = get_scorer()  # fuerza la recarga ya mismo
    return {
        "status": "reloaded",
        "champion_version": store.version if store.loaded else "unknown",
        "has_challenger": scorer.has_challenger,
        "has_federated": scorer.has_federated,
        "grpc_champion_version": grpc_version,
    }


@app.post("/v1/mlops/seed-frauds", tags=["MLOps"])
def mlops_seed_frauds(
    n: int = 30,
    seed: int = 42,
    _client: str = Security(validate_token),
) -> dict:
    """Siembra fraudes post-mortem muestreando casos REALES del dataset para poblar la ground truth.

    Toma `n` transacciones de fraude reales del split de test (`data/processed/test.parquet`),
    las scorea (champion decide + challenger/federado en sombra, todo se registra en
    `fraud.db`) y les pega un label de fraude (post-mortem). Así la ground truth proviene
    de datos reales del sistema, no de un archivo curado a mano.

    Correr DESPUÉS de 'Recargar modelos' (con las sombras activas) para que se registren
    todas las predicciones y la evaluación tenga sentido. Si no hay dataset disponible,
    cae a `seed_frauds.json` (fallback).
    """
    from fraud.api.scoring import get_scorer
    from fraud.api.store import SOURCE_POST_MORTEM, get_store

    rows, source = _load_seed_frauds(n, seed)

    scorer = get_scorer()
    store_ = get_store()
    seeded = 0
    with_challenger = 0
    with_federated = 0
    for row in rows:
        result = scorer.score(row)
        tid = result["transaction_id"]
        store_.add_label(tid, label=1, source=SOURCE_POST_MORTEM)
        seeded += 1
        if result.get("challenger") is not None:
            with_challenger += 1
        if result.get("federated") is not None:
            with_federated += 1

    note = "ground truth sembrada desde datos reales; ya podés evaluar las sombras vs champion."
    if seeded > 0 and with_challenger == 0 and with_federated == 0:
        note = (
            "ningún modelo en sombra activo: recargá modelos (MLOps → Recargar) ANTES de "
            "sembrar para que se registren las predicciones comparables"
        )
    return {
        "seeded": seeded,
        "with_challenger": with_challenger,
        "with_federated": with_federated,
        "source": source,
        "note": note,
    }


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


