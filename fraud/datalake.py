"""Helpers mínimos para el data lake (MinIO/S3) con boto3.

El pipeline usa MinIO no sólo como artifact store de MLflow, sino también como
**data lake**: el raw bajado de HuggingFace y los datos procesados se guardan en el
bucket ``data-lake``, de modo que el entrenamiento lea del lake (reproducible) y no
dependa de un proveedor externo ni del disco local.

Capas (keys dentro del bucket):
    raw/fraud_sample.parquet          <- dump crudo (el ingest lo sube)
    processed/{train,val,test}.parquet <- features + split (features los sube)

Tolerante por diseño: si ``boto3`` no está instalado o ``MLFLOW_S3_ENDPOINT_URL`` no
está seteada (p. ej. dev local sin MinIO), las funciones NO fallan — devuelven False y
el pipeline sigue con los archivos locales de siempre.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

BUCKET = os.getenv("DATA_LAKE_BUCKET", "data-lake")
_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL")  # reutilizamos el endpoint de MinIO (p. ej. http://minio:9000)


def _client():
    """Cliente S3 apuntando a MinIO, o None si no hay data lake configurado/disponible."""
    if not _ENDPOINT:
        return None
    try:
        import boto3

        return boto3.client(
            "s3",
            endpoint_url=_ENDPOINT,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except Exception as exc:  # noqa: BLE001 - sin boto3/credenciales seguimos con disco local
        logger.warning(f"Data lake no disponible ({_ENDPOINT}): {exc}")
        return None


def upload(local_path, key: str) -> bool:
    """Sube un archivo local al data lake como ``s3://{BUCKET}/{key}``. Devuelve True si subió."""
    client = _client()
    if client is None:
        logger.info("Sin data lake; no subo '{}' (sigo con disco local).", key)
        return False
    try:
        client.upload_file(str(local_path), BUCKET, key)
        logger.success("Data lake ← subido s3://{}/{}", BUCKET, key)
        return True
    except Exception as exc:  # noqa: BLE001 - subir al lake es best-effort, no debe tumbar el pipeline
        logger.warning(f"No se pudo subir '{key}' al data lake: {exc}")
        return False


def download(key: str, local_path) -> bool:
    """Baja ``s3://{BUCKET}/{key}`` a ``local_path``. Devuelve True si bajó (si no, se usa el local)."""
    client = _client()
    if client is None:
        return False
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client.download_file(BUCKET, key, str(local_path))
        logger.success("Data lake → descargado s3://{}/{} -> {}", BUCKET, key, local_path)
        return True
    except Exception as exc:  # noqa: BLE001 - si no está en el lake, caemos al archivo local
        logger.info(f"'{key}' no está en el data lake ({exc}); uso el archivo local si existe.")
        return False
