FROM python:3.12-slim

WORKDIR /app

# Poetry para instalar dependencias.
RUN pip install --no-cache-dir poetry

# Copiamos sólo el pyproject primero (aprovecha la cache de Docker).
COPY pyproject.toml ./

# Instalamos en el Python del sistema (sin venv de poetry, más simple en contenedor).
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-root

# Generamos el código gRPC a partir del .proto DENTRO de la imagen.
COPY proto ./proto/
RUN python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/ml_service.proto

# Copiamos el resto (main.py, model.joblib).
COPY . .

CMD ["python", "main.py"]
