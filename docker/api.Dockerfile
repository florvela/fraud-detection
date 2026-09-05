# Imagen de serving: solo sirve el modelo, no lo entrena
# Se construye desde la raíz del proyecto:
#   docker build -f docker/api.Dockerfile -t fraud-api .
#   docker run -p 8080:8080 fraud-api
#   docker run -p 8080:8080 -e API_KEYS="mi-token" fraud-api

# numpy/pandas requieren Python >= 3.12
FROM python:3.12-slim

# uv (gestor rápido): copiamos su binario desde la imagen oficial, sin instalarlo
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# PYTHONPATH=/app hace importable el paquete 'fraud' sin instalarlo
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Solo las dependencias de serving (no jupyter/datasets/matplotlib de dev/train)
RUN uv pip install --system --no-cache \
    fastapi "uvicorn[standard]" pydantic \
    scikit-learn xgboost pandas numpy joblib \
    loguru python-dotenv

# Código de la API + modelo ya entrenado
COPY fraud/ ./fraud/
COPY models/model.joblib ./models/model.joblib

EXPOSE 8080

CMD ["uvicorn", "fraud.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
