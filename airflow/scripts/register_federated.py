"""Registra el MLP federado en MLflow como un modelo con alias ``federated``.

Último paso del DAG ``fraud_federate_pipeline``. Toma los pesos del modelo global
federado (``models/mlp_federado.npz``, que dejaron los contenedores Flower) y el
spec público de preprocesamiento, los **envuelve en un pipeline con interfaz
sklearn** (``fraud.federated_model.FederatedPipeline``) y produce el MISMO artefacto
joblib que usan champion/challenger::

    {"pipeline": FederatedPipeline(...), "feature_order", "version", "metrics"}

Gracias a eso, el federado entra al registry como una versión más del modelo
``fraud-detection`` con alias ``federated``: el serving lo puede correr en sombra y
promoverlo a champion con la maquinaria existente (``fraud/api/deploy.py``), sin
reconstruir imágenes.

No necesita torch: el pipeline guarda los pesos como numpy y arma el modelo de torch
recién en la primera inferencia (en el serving). Las métricas se leen del JSON que
produjo ``evaluate_vs_central.py --metrics-out`` (ese paso sí corre con torch).
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
import tempfile

import joblib
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

from fraud.config import MODELS_DIR, PROJ_ROOT
from fraud.federated_model import (
    CATEGORICAL_FEATURES,
    CATEGORY_VOCAB,
    FEATURE_ORDER,
    NUMERIC_FEATURES,
    build_pipeline,
    load_weights_npz,
)

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")
FEDERATED_ALIAS = os.getenv("MLFLOW_FEDERATED_ALIAS", "federated")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
ARTIFACT_DIR = "model_artifact"  # coincide con MLFLOW_ARTIFACT_PATH del serving

# Entradas producidas por los pasos previos del DAG.
FED_WEIGHTS = MODELS_DIR / "mlp_federado.npz"
PREPROCESS_SPEC = PROJ_ROOT / "services" / "federated" / "preprocess_spec.json"
METRICS_JSON = MODELS_DIR / "federated_metrics.json"

# Fallback local para el serving sin MLflow (análogo a challenger.joblib).
FEDERATED_LOCAL = MODELS_DIR / "federated.joblib"


def _load_metrics() -> dict:
    """Lee las métricas del federado (si existen); best-effort."""
    if METRICS_JSON.exists():
        return json.loads(METRICS_JSON.read_text())
    logger.warning("No existe {}; registro el federado sin métricas.", METRICS_JSON)
    return {}


def main() -> None:
    if not FED_WEIGHTS.exists():
        raise FileNotFoundError(
            f"No existe {FED_WEIGHTS}; corré primero el entrenamiento federado (fed-server + clientes)."
        )
    if not PREPROCESS_SPEC.exists():
        raise FileNotFoundError(f"No existe el spec de preprocesamiento {PREPROCESS_SPEC}.")

    spec = json.loads(PREPROCESS_SPEC.read_text())
    weights = load_weights_npz(FED_WEIGHTS)
    metrics = _load_metrics()
    version = f"federado-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

    pipeline = build_pipeline(spec, weights)
    artifact = {
        "pipeline": pipeline,
        "feature_order": FEATURE_ORDER,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        # Mismo contrato que el artefacto XGBoost: así /v1/model-info y la UI tienen
        # las categorías aunque el federado sea el champion servido.
        "categories": {k: list(v) for k, v in CATEGORY_VOCAB.items()},
        "version": version,
        "metrics": metrics,
        "model_kind": "federated-mlp",
    }

    # Fallback local (serving sin registry).
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, FEDERATED_LOCAL)
    logger.info("Artefacto federado guardado localmente en {}", FEDERATED_LOCAL)

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        logger.warning("MLFLOW_TRACKING_URI no seteada: solo quedó el fallback local.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run(run_name=version) as run:
        mlflow.set_tag("model_kind", "federated-mlp")
        mlflow.log_param("model_version", version)
        mlflow.log_param("fed_rounds", os.getenv("FED_ROUNDS", "unknown"))
        mlflow.log_param("feature_order", FEATURE_ORDER)
        for k, v in metrics.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):  # claves no numéricas: como param
                mlflow.log_param(k, v)
        # Loguear el npz y el spec como artefactos trazables del camino federado.
        mlflow.log_artifact(str(FED_WEIGHTS), artifact_path="federated_raw")
        mlflow.log_artifact(str(PREPROCESS_SPEC), artifact_path="federated_raw")
        # El joblib envuelto, con el nombre que espera el loader del serving.
        with tempfile.TemporaryDirectory() as tmp:
            joblib_path = os.path.join(tmp, "model.joblib")
            joblib.dump(artifact, joblib_path)
            mlflow.log_artifact(joblib_path, artifact_path=ARTIFACT_DIR)

        model_uri = f"runs:/{run.info.run_id}/{ARTIFACT_DIR}"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    client.set_registered_model_alias(MODEL_NAME, FEDERATED_ALIAS, mv.version)
    pr = metrics.get("pr_auc")
    logger.success(
        "Registrada v{} de '{}' como '{}' (PR-AUC federado={}). "
        "Corré en sombra y, si querés, promovelo a champion desde la UI.",
        mv.version, MODEL_NAME, FEDERATED_ALIAS, f"{pr:.4f}" if isinstance(pr, (int, float)) else "n/d",
    )


if __name__ == "__main__":
    main()
