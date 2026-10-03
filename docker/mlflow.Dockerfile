# Servidor de tracking + Model Registry de MLflow.
# Backend store: PostgreSQL. Artifact store: MinIO (S3).
FROM python:3.12-slim

# NOTA: mlflow 2.16.2 importa `FallbackAsyncAdaptedQueuePool`, que SQLAlchemy
# eliminó en 2.0.36. Sin fijarlo, pip trae una SQLAlchemy nueva y el server
# crashea al iniciar ("cannot import name 'FallbackAsyncAdaptedQueuePool'").
RUN pip install --no-cache-dir \
    "mlflow==2.16.2" \
    "sqlalchemy<2.0.36" \
    psycopg2-binary boto3

EXPOSE 5000

# Los parámetros reales (URIs de Postgres y S3/MinIO) se pasan por el compose.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
