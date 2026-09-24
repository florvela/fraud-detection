# Núcleo de scoring gRPC. NO hornea el modelo: lo carga del registry de MLflow
# (alias champion) al arrancar. Si MLFLOW_TRACKING_URI no está, cae al joblib local.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN uv pip install --system --no-cache \
    grpcio grpcio-tools \
    scikit-learn xgboost pandas numpy joblib \
    "mlflow==2.16.2" boto3 loguru python-dotenv

# Solo el código (el modelo llega por registry o por volumen, no se copia acá)
COPY fraud/ ./fraud/

# Regenera los stubs de protobuf con el runtime de ESTA imagen (evita mismatch de versiones)
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. fraud/api/proto/fraud.proto

EXPOSE 50052

CMD ["python", "-m", "fraud.api.grpc_server"]
