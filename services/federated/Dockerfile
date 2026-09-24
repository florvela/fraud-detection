# Imagen única parametrizable por CMD para servidor y clientes federados.
# - fed-server:   CMD ["python", "server.py"]
# - fed-client-*: CMD ["python", "client.py"]  (silo por env FED_SILO)
FROM python:3.12-slim

WORKDIR /app

# Dependencias del sistema mínimas (torch trae sus binarios; sin build tools).
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código del servicio. Los datos (silos) y modelos se montan/comparten por volumen.
COPY . .

# Puerto por defecto del servidor Flower.
EXPOSE 8080

# Por defecto arranca el servidor; los clientes sobreescriben el CMD en compose.
CMD ["python", "server.py"]
