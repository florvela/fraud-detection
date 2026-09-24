# Imagen única para producer + consumer del servicio de streaming (Mini-TP 4).
# Python 3.11 base (según consigna). El modelo NO se copia: se monta como volumen
# en /app/models/model.joblib.
FROM python:3.11-slim

# libgomp1 es requerida por xgboost (OpenMP) en runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# El código vive en /app/services/streaming para que common.PROJECT_ROOT (= dos
# niveles arriba) resuelva a /app, y así el modelo montado quede en
# /app/models/model.joblib tal como pide la consigna.
WORKDIR /app/services/streaming

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common.py producer.py consumer.py batch_compare.py ./

ENV PYTHONUNBUFFERED=1

# Por defecto levanta el consumer contra Kafka. El producer sobreescribe el
# command en el docker-compose (ver README).
CMD ["python", "consumer.py", "--source", "kafka", "--bootstrap-servers", "redpanda:9092"]
