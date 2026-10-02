"""Deploy / rollback del modelo (UC6).

Promueve el **challenger** a **champion**: a partir de ahí el ex-challenger pasa a
decidir y el champion anterior se retira (queda disponible para rollback). Dos modos:

- **MLflow** (si `MLFLOW_TRACKING_URI` está seteada): mueve el alias `champion` a la
  versión que hoy tiene el alias `challenger`, y deja la versión saliente como
  `previous` para poder revertir. No se reconstruye ninguna imagen.
- **Local** (fallback / dev): copia `models/challenger.joblib` sobre
  `models/model.joblib`, respaldando el anterior en `models/previous.joblib`.

Tras promover, se recarga el serving (reset del Scorer) para que el núcleo tome el
nuevo champion.
"""

from __future__ import annotations

import os
import shutil

from loguru import logger

from fraud.api.model_loader import (
    CHALLENGER_FILE,
    MLFLOW_CHALLENGER_ALIAS,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MODEL_FILE,
)
from fraud.config import MODELS_DIR

PREVIOUS_FILE = MODELS_DIR / "previous.joblib"


def _mlflow_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def promote() -> dict:
    """Promueve el challenger a champion. Devuelve el resultado de la operación."""
    if _mlflow_enabled():
        return _promote_mlflow()
    return _promote_local()


def rollback() -> dict:
    """Revierte al champion anterior (previous)."""
    if _mlflow_enabled():
        return _rollback_mlflow()
    return _rollback_local()


# ------------------------------------------------------------------ MLflow
def _promote_mlflow() -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    challenger = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_CHALLENGER_ALIAS)
    try:
        champion = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "previous", champion.version)
        prev = champion.version
    except Exception:  # noqa: BLE001 - si no hay champion previo, no hay nada que respaldar
        prev = None

    client.set_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS, challenger.version)
    client.delete_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_CHALLENGER_ALIAS)
    _reload_serving()
    logger.success(f"Promovido challenger v{challenger.version} a champion (previo v{prev}).")
    return {
        "mode": "mlflow",
        "promoted_version": challenger.version,
        "previous_version": prev,
        "status": "deployed",
    }


def _rollback_mlflow() -> dict:
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    prev = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, "previous")
    client.set_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS, prev.version)
    _reload_serving()
    logger.success(f"Rollback: champion vuelve a v{prev.version}.")
    return {"mode": "mlflow", "champion_version": prev.version, "status": "rolled_back"}


# ------------------------------------------------------------------- local
def _promote_local() -> dict:
    if not CHALLENGER_FILE.exists():
        raise FileNotFoundError(
            f"No hay challenger local en {CHALLENGER_FILE}; entrená/activá uno primero."
        )
    if MODEL_FILE.exists():
        shutil.copyfile(MODEL_FILE, PREVIOUS_FILE)
    shutil.copyfile(CHALLENGER_FILE, MODEL_FILE)
    CHALLENGER_FILE.unlink()  # el challenger ya es champion; no queda sombra
    _reload_serving()
    logger.success("Promovido challenger local a champion (previo en previous.joblib).")
    return {
        "mode": "local",
        "status": "deployed",
        "previous_backup": str(PREVIOUS_FILE) if PREVIOUS_FILE.exists() else None,
    }


def _rollback_local() -> dict:
    if not PREVIOUS_FILE.exists():
        raise FileNotFoundError(f"No hay champion previo en {PREVIOUS_FILE} para revertir.")
    shutil.copyfile(PREVIOUS_FILE, MODEL_FILE)
    _reload_serving()
    logger.success("Rollback: champion local restaurado desde previous.joblib.")
    return {"mode": "local", "status": "rolled_back"}


def _reload_serving() -> None:
    """Recarga el Scorer del serving para que tome el nuevo champion."""
    try:
        from fraud.api.scoring import reset_scorer

        reset_scorer()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"No se pudo resetear el Scorer: {exc}")
