# Imagen para correr el reporte de drift con Evidently.
FROM python:3.12-slim

# Evita prompts y buffering de logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Instalar dependencias primero para aprovechar la cache de capas.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del servicio.
COPY evidently_report.py .

# Los datos y la carpeta de reports se montan como volúmenes desde compose:
#   - /data      -> data/ del repo (reference/current)
#   - /app/reports -> services/monitoring/reports (salida HTML)
VOLUME ["/data", "/app/reports"]

# Por defecto genera el reporte train (reference) vs test (current).
ENTRYPOINT ["python", "evidently_report.py"]
CMD ["run", \
     "--reference", "/data/processed/train.parquet", \
     "--current", "/data/processed/test.parquet", \
     "--output", "/app/reports/drift_report.html"]
