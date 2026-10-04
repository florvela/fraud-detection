"""Registra el modelo entrenado en MLflow como champion (bootstrap) o challenger.

Lo llama el DAG de Airflow como paso final del pipeline. Flujo:
1. Carga `models/model.joblib` (el artefacto que dejó `fraud.modeling.train`).
2. Abre un run de MLflow y loguea params, métricas y el artefacto joblib.
3. Registra una nueva versión del modelo `fraud-detection`.
4. Asigna el alias según el ciclo de vida del sistema:
   - **No hay champion todavía** (arranque en frío / seed) → alias `champion`.
   - **Ya hay champion** → alias `challenger`. NO se auto-promueve: el challenger
     corre en sombra (UC1), se evalúa contra la ground truth (UC5) y recién el
     Ingeniero MLOps lo promueve con un deploy explícito (UC6, ver `fraud/api/deploy.py`).

Esto es lo que hace real al esquema champion/challenger: el modelo nuevo entra como
challenger y la promoción es una decisión informada, no un `>=` automático de PR-AUC.
El serving carga cada alias por su cuenta, así que nada de esto reconstruye imágenes.
"""

from __future__ import annotations

import json
import os

import joblib
import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from fraud.config import MODELS_DIR, PROCESSED_DATA_DIR

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")
CHAMPION_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
CHALLENGER_ALIAS = os.getenv("MLFLOW_CHALLENGER_ALIAS", "challenger")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
ARTIFACT_DIR = "model_artifact"  # coincide con MLFLOW_ARTIFACT_PATH del serving


def _has_champion(client: MlflowClient) -> bool:
    """True si ya hay un modelo con alias champion en el registry."""
    try:
        client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        return True
    except Exception:  # noqa: BLE001 - cualquier fallo = no hay champion todavía
        return False


def main() -> None:
    model_file = MODELS_DIR / "model.joblib"
    if not model_file.exists():
        raise FileNotFoundError(f"No existe {model_file}; corré primero el entrenamiento.")

    artifact = joblib.load(model_file)
    metrics = artifact.get("metrics", {})
    version = artifact.get("version", "unknown")
    pr_auc = float(metrics.get("pr_auc", 0.0))

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    champion_exists = _has_champion(client)

    with mlflow.start_run(run_name=f"train-{version}") as run:
        mlflow.log_param("model_version", version)
        mlflow.log_param("feature_order", artifact.get("feature_order"))
        # Tamaño del dataset usado en el retrain (base original + inyectados).
        meta_file = PROCESSED_DATA_DIR / "train_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            mlflow.log_param("n_rows_base", meta.get("n_base"))
            mlflow.log_param("n_rows_injected", meta.get("n_injected"))
            mlflow.log_param("n_rows_total", meta.get("n_total"))
            logger.info(
                f"Dataset: base={meta.get('n_base')} "
                f"injected={meta.get('n_injected')} total={meta.get('n_total')}"
            )
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        # Logueamos el dict joblib entero como artefacto: el serving lo descarga tal cual
        mlflow.log_artifact(str(model_file), artifact_path=ARTIFACT_DIR)

        model_uri = f"runs:/{run.info.run_id}/{ARTIFACT_DIR}"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        logger.info(f"Registrada versión {mv.version} de '{MODEL_NAME}' (PR-AUC={pr_auc:.4f})")

    # Bootstrap vs retrain: primer modelo = champion; de ahí en más = challenger.
    if not champion_exists:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)
        logger.success(
            f"Arranque en frío: v{mv.version} queda como '{CHAMPION_ALIAS}' (no había champion)."
        )
    else:
        client.set_registered_model_alias(MODEL_NAME, CHALLENGER_ALIAS, mv.version)
        logger.success(
            f"Registrado v{mv.version} como '{CHALLENGER_ALIAS}'. "
            "Corré en sombra, evaluá (UC5) y promové con un deploy explícito (UC6)."
        )


if __name__ == "__main__":
    main()
