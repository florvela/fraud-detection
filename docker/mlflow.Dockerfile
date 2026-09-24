# Servidor de tracking + Model Registry de MLflow.
# Backend store: PostgreSQL. Artifact store: MinIO (S3).
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow==2.16.2" \
    psycopg2-binary boto3

EXPOSE 5000

# Los parámetros reales (URIs de Postgres y S3/MinIO) se pasan por el compose.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
