# Imagen única para producer + consumer del servicio de streaming (Mini-TP 4).
# Python 3.11 base (según consigna). El TP es autocontenido: el modelo se
# hornea en la imagen desde models/model.joblib (no se monta nada del repo raíz).
FROM python:3.11-slim

# libgomp1 es requerida por xgboost (OpenMP) en runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Todo el TP vive en /app: common.HERE resuelve a /app y el modelo queda en
# /app/models/model.joblib.
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py producer.py consumer.py batch_compare.py ./
COPY models/model.joblib ./models/model.joblib

ENV PYTHONUNBUFFERED=1

# Por defecto levanta el consumer contra Kafka. El producer sobreescribe el
# command en el docker-compose (ver README).
CMD ["python", "consumer.py", "--source", "kafka", "--bootstrap-servers", "redpanda:9092"]
