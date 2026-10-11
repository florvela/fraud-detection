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
    FEDERATED_FILE,
    MLFLOW_CHALLENGER_ALIAS,
    MLFLOW_FEDERATED_ALIAS,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MODEL_FILE,
)
from fraud.config import MODELS_DIR

PREVIOUS_FILE = MODELS_DIR / "previous.joblib"

# Modelos en sombra que se pueden promover a champion (UC6). Cada uno mapea a su
# alias en el registry de MLflow y a su artefacto local de fallback.
_SOURCES = {
    "challenger": (MLFLOW_CHALLENGER_ALIAS, CHALLENGER_FILE),
    "federated": (MLFLOW_FEDERATED_ALIAS, FEDERATED_FILE),
}


def _resolve_source(source: str) -> tuple[str, object]:
    try:
        return _SOURCES[source]
    except KeyError:
        raise ValueError(f"source inválido: {source!r} (esperado {sorted(_SOURCES)})")


def _mlflow_enabled() -> bool:
    return bool(os.getenv("MLFLOW_TRACKING_URI"))


def promote(source: str = "challenger") -> dict:
    """Promueve a champion el modelo en sombra indicado (challenger | federated)."""
    alias, local_file = _resolve_source(source)
    if _mlflow_enabled():
        return _promote_mlflow(source, alias)
    return _promote_local(source, local_file)


def rollback() -> dict:
    """Revierte al champion anterior (previous)."""
    if _mlflow_enabled():
        return _rollback_mlflow()
    return _rollback_local()


# ------------------------------------------------------------------ MLflow
def _promote_mlflow(source: str, source_alias: str) -> dict:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    candidate = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, source_alias)
    try:
        champion = client.get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
        client.set_registered_model_alias(MLFLOW_MODEL_NAME, "previous", champion.version)
        prev = champion.version
    except Exception:  # noqa: BLE001 - si no hay champion previo, no hay nada que respaldar
        prev = None

    client.set_registered_model_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS, candidate.version)
    # El modelo promovido deja de ser sombra: retiramos su alias de origen.
    client.delete_registered_model_alias(MLFLOW_MODEL_NAME, source_alias)
    _reload_serving()
    logger.success(f"Promovido {source} v{candidate.version} a champion (previo v{prev}).")
    return {
        "mode": "mlflow",
        "source": source,
        "promoted_version": candidate.version,
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
def _promote_local(source: str, source_file) -> dict:
    if not source_file.exists():
        raise FileNotFoundError(
            f"No hay {source} local en {source_file}; entrená/activá uno primero."
        )
    if MODEL_FILE.exists():
        shutil.copyfile(MODEL_FILE, PREVIOUS_FILE)
    shutil.copyfile(source_file, MODEL_FILE)
    source_file.unlink()  # el modelo promovido ya es champion; no queda sombra
    _reload_serving()
    logger.success(f"Promovido {source} local a champion (previo en previous.joblib).")
    return {
        "mode": "local",
        "source": source,
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
    """Recarga el serving para que tome el nuevo champion.

    Resetea el Scorer in-process del borde REST y, si el scoring del champion se
    delega al núcleo gRPC (GRPC_HOST seteada), le pide que recargue su modelo —
    si no, el contenedor gRPC seguiría sirviendo el champion viejo.
    """
    try:
        from fraud.api.scoring import reset_scorer

        reset_scorer()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"No se pudo resetear el Scorer: {exc}")

    try:
        from fraud.api import grpc_client

        version = grpc_client.reload_remote()
        if version is not None:
            logger.success(f"Núcleo gRPC recargado al champion v{version}.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"No se pudo recargar el núcleo gRPC: {exc}")
