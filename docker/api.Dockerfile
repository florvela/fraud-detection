# Imagen de serving REST (borde). Delega el scoring en el núcleo gRPC y carga los
# metadatos del modelo del registry de MLflow (no hornea el .joblib en la imagen).
# Se construye desde la raíz del proyecto:
#   docker build -f docker/api.Dockerfile -t fraud-rest .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencias de serving REST + cliente gRPC (para delegar) + MLflow (registry)
RUN uv pip install --system --no-cache \
    fastapi "uvicorn[standard]" pydantic \
    grpcio grpcio-tools \
    scikit-learn xgboost pandas numpy joblib \
    "mlflow==2.16.2" boto3 \
    strawberry-graphql neo4j \
    prometheus-fastapi-instrumentator \
    loguru python-dotenv requests

# Código de la API (el modelo llega por registry/volumen, no se copia)
COPY fraud/ ./fraud/

# Regenera los stubs de protobuf con el runtime de ESTA imagen (evita mismatch de versiones)
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fraud/api/proto/fraud.proto

EXPOSE 8080

CMD ["uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
