# Interfaz del analista de fraude (Streamlit)
# Se construye desde este directorio (services/ui):
#   docker build -t fraud-ui .
#   docker run -p 8501:8501 -e REST_URL=http://host.docker.internal:8080 fraud-ui

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Primero las dependencias para aprovechar la cache de Docker.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código de la app + datos placeholder (alerts.csv).
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
