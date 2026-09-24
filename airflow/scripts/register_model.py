"""Registra el modelo entrenado en MLflow y promueve champion/challenger.

Lo llama el DAG de Airflow como paso final del pipeline. Flujo:
1. Carga `models/model.joblib` (el artefacto que dejó `fraud.modeling.train`).
2. Abre un run de MLflow y loguea params, métricas y el artefacto joblib.
3. Registra una nueva versión del modelo `fraud-detection`.
4. Compara su PR-AUC contra el champion actual (challenger): si es igual o mejor
   —o si no hay champion todavía— le asigna el alias `champion`.

El serving (gRPC/REST) carga el champion por alias, así que promover un mejor
modelo NO requiere reconstruir ninguna imagen.
"""

from __future__ import annotations

import os

import joblib
import mlflow
from loguru import logger
from mlflow.tracking import MlflowClient

from fraud.config import MODELS_DIR

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")
ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
ARTIFACT_DIR = "model_artifact"  # coincide con MLFLOW_ARTIFACT_PATH del serving


def _current_champion_pr_auc(client: MlflowClient) -> float | None:
    """PR-AUC del champion actual (o None si todavía no hay ninguno)."""
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        return None
    run = client.get_run(mv.run_id)
    val = run.data.metrics.get("pr_auc")
    return float(val) if val is not None else None


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

    champ_pr_auc = _current_champion_pr_auc(client)

    with mlflow.start_run(run_name=f"train-{version}") as run:
        mlflow.log_param("model_version", version)
        mlflow.log_param("feature_order", artifact.get("feature_order"))
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        # Logueamos el dict joblib entero como artefacto: el serving lo descarga tal cual
        mlflow.log_artifact(str(model_file), artifact_path=ARTIFACT_DIR)

        model_uri = f"runs:/{run.info.run_id}/{ARTIFACT_DIR}"
        mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        logger.info(f"Registrada versión {mv.version} de '{MODEL_NAME}' (PR-AUC={pr_auc:.4f})")

    # Champion / challenger
    if champ_pr_auc is None:
        promote = True
        reason = "no había champion previo"
    elif pr_auc >= champ_pr_auc:
        promote = True
        reason = f"PR-AUC {pr_auc:.4f} >= champion {champ_pr_auc:.4f}"
    else:
        promote = False
        reason = f"PR-AUC {pr_auc:.4f} < champion {champ_pr_auc:.4f}"

    if promote:
        client.set_registered_model_alias(MODEL_NAME, ALIAS, mv.version)
        logger.success(f"Promovido a '{ALIAS}' v{mv.version} ({reason}).")
    else:
        logger.info(f"NO se promueve: {reason}. El champion sigue igual.")


if __name__ == "__main__":
    main()
