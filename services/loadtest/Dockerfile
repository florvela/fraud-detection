# Simulador de carga (Locust). Web UI en :8089 para lanzar N clientes y ver
# throughput + latencias p50/p95/p99 en vivo contra el borde REST.
FROM python:3.12-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY locustfile.py .

EXPOSE 8089

# El host objetivo se pasa por env LOCUST_HOST (lo lee Locust). Levanta la web UI.
ENTRYPOINT ["locust"]
CMD ["-f", "/app/locustfile.py"]
