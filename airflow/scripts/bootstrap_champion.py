"""Siembra un champion inicial DETERMINISTA desde un artefacto versionado en git.

Se usa para la demo: arrancar con un champion a propósito **malo**
(`models/badmodel.joblib`, entrenado sin balanceo de clases) para después mostrar
cómo el retrain del DAG produce un challenger mejor y se promueve.

Lo dispara `airflow-init` SÓLO cuando la env `SEED_BAD_CHAMPION=1`. Registra el
artefacto en MLflow como una versión nueva y le asigna el alias `champion`
(forzado, pisa el que hubiera), y borra el alias `challenger` para dejar un estado
"antes del retrain" limpio. Sin la flag, el bootstrap normal (DAG) no se toca.

Idempotente y tolerante al arranque: reintenta hasta que MLflow/MinIO estén listos.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time

import joblib
from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient

from fraud.config import MODELS_DIR

MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")
CHAMPION_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
CHALLENGER_ALIAS = os.getenv("MLFLOW_CHALLENGER_ALIAS", "challenger")
FEDERATED_ALIAS = os.getenv("MLFLOW_FEDERATED_ALIAS", "federated")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "fraud-detection")
ARTIFACT_DIR = "model_artifact"
BAD_MODEL_FILE = MODELS_DIR / "badmodel.joblib"


def _wait_for_mlflow(client: MlflowClient, attempts: int = 30, delay: int = 5) -> None:
    """Espera a que el tracking server responda (puede tardar en el arranque)."""
    for i in range(1, attempts + 1):
        try:
            client.search_registered_models(max_results=1)
            return
        except Exception as exc:  # noqa: BLE001
            logger.info(f"MLflow no listo aún ({i}/{attempts}): {exc}")
            time.sleep(delay)
    raise RuntimeError("MLflow no respondió a tiempo para sembrar el champion.")


def main() -> None:
    if not BAD_MODEL_FILE.exists():
        raise FileNotFoundError(f"No existe {BAD_MODEL_FILE} (debería estar versionado en git).")

    artifact = joblib.load(BAD_MODEL_FILE)
    metrics = artifact.get("metrics", {})
    version = artifact.get("version", "bootstrap")

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    _wait_for_mlflow(client)

    # Logueamos el artefacto como model.joblib (el serving baja ese nombre).
    with tempfile.TemporaryDirectory() as d:
        staged = Path(d) / "model.joblib"
        joblib.dump(artifact, staged)
        with mlflow.start_run(run_name=f"bootstrap-champion-{version}") as run:
            mlflow.log_param("model_version", version)
            mlflow.log_param("bootstrap", "bad_champion")
            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))
            mlflow.log_artifact(str(staged), artifact_path=ARTIFACT_DIR)
            mv = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/{ARTIFACT_DIR}", name=MODEL_NAME
            )

    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, mv.version)
    # Limpiamos las SOMBRAS (challenger + federado) para arrancar la demo con el champion
    # malo SOLO: así no aparece una sombra "fantasma" de una corrida anterior persistida.
    for alias in (CHALLENGER_ALIAS, FEDERATED_ALIAS):
        try:
            client.delete_registered_model_alias(MODEL_NAME, alias)
        except Exception as exc:  # noqa: BLE001 - puede no existir, no es un error
            logger.debug(f"No se pudo borrar el alias '{alias}' (quizá no existía): {exc}")
    logger.success(
        f"Champion inicial (malo) sembrado: v{mv.version} ({version}) como "
        f"'{CHAMPION_ALIAS}'. Aliases '{CHALLENGER_ALIAS}' y '{FEDERATED_ALIAS}' limpiados."
    )


if __name__ == "__main__":
    main()
