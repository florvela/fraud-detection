"""Carga el modelo entrenado y lo expone a la API.

Dos orígenes posibles, con fallback:
1. **MLflow Model Registry** (producción): si está seteada la env `MLFLOW_TRACKING_URI`,
   se descarga el artefacto `model.joblib` de la versión con alias `champion` del
   modelo registrado (`MLFLOW_MODEL_NAME`). Así, promover un challenger en el
   registry NO obliga a reconstruir la imagen de serving.
2. **Archivo local** `models/model.joblib` (desarrollo / fallback): el modo de
   siempre, útil en local o si MLflow no está disponible.

El artefacto es siempre el mismo dict de joblib (pipeline + metadatos), así que el
resto de la API no cambia según el origen.
"""

from __future__ import annotations

import os

import joblib
from loguru import logger

from fraud.config import MODELS_DIR

MODEL_FILE = MODELS_DIR / "model.joblib"

# Config del registry (se activa solo si MLFLOW_TRACKING_URI está seteada)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "fraud-detection")
MLFLOW_MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")
# Nombre del archivo joblib tal como lo loguea el DAG de entrenamiento
MLFLOW_ARTIFACT_PATH = os.getenv("MLFLOW_ARTIFACT_PATH", "model_artifact/model.joblib")


class ModelStore:
    def __init__(self) -> None:
        self._artifact: dict | None = None
        self._source: str | None = None

    @property
    def loaded(self) -> bool:
        return self._artifact is not None

    @property
    def source(self) -> str | None:
        """De dónde se cargó el modelo: 'registry' o 'local'."""
        return self._source

    def load(self) -> bool:
        """Intenta el registry de MLflow y, si no, el archivo local."""
        if MLFLOW_TRACKING_URI:
            if self._load_from_registry():
                return True
            logger.warning("No se pudo cargar del registry; intento archivo local.")
        return self._load_from_file()

    def _load_from_registry(self) -> bool:
        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model_uri = f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
            logger.info(f"Descargando artefacto del champion desde {model_uri} ...")
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=f"{model_uri}/{MLFLOW_ARTIFACT_PATH}"
            )
            self._artifact = joblib.load(local_path)
            self._source = "registry"
            logger.success(
                f"Modelo cargado del registry ({MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}, "
                f"v{self._artifact.get('version')})"
            )
            return True
        except Exception as exc:  # noqa: BLE001 - queremos degradar a local ante cualquier fallo
            logger.warning(f"Fallo al cargar del registry MLflow: {exc}")
            return False

    def _load_from_file(self) -> bool:
        if not MODEL_FILE.exists():
            logger.error(f"No existe el modelo local en {MODEL_FILE}")
            return False
        self._artifact = joblib.load(MODEL_FILE)
        self._source = "local"
        logger.success(f"Modelo cargado del archivo local {MODEL_FILE}")
        return True

    def reload(self) -> bool:
        """Vuelve a cargar el champion (p.ej. tras una promoción en el registry)."""
        self._artifact = None
        self._source = None
        return self.load()

    @property
    def pipeline(self):
        return self._require()["pipeline"]

    @property
    def version(self) -> str:
        return self._require()["version"]

    @property
    def feature_order(self) -> list[str]:
        return self._require()["feature_order"]

    @property
    def metrics(self) -> dict:
        return self._require()["metrics"]

    @property
    def info(self) -> dict:
        info = {k: v for k, v in self._require().items() if k != "pipeline"}
        info["source"] = self._source
        return info

    def _require(self) -> dict:
        if self._artifact is None:
            raise RuntimeError(f"Modelo no cargado, se esperaba {MODEL_FILE} o el registry")
        return self._artifact
