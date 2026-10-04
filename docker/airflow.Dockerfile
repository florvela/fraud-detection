# Imagen de Airflow con las dependencias del proyecto de fraude.
# Airflow corre el pipeline reutilizando el paquete `fraud` montado por volumen.
FROM apache/airflow:2.9.3-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CLI de docker + plugin compose v2 (sin daemon): el DAG federado orquesta los
# contenedores Flower del stack vía el socket del host (Docker-outside-of-Docker).
# Arch-aware: funciona en x86_64 (Intel) y aarch64 (Apple Silicon).
ARG DOCKER_VERSION=27.3.1
ARG COMPOSE_VERSION=v2.29.7
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
      x86_64) dk=x86_64; cp=x86_64 ;; \
      aarch64|arm64) dk=aarch64; cp=aarch64 ;; \
      *) echo "arch no soportada: $arch" >&2; exit 1 ;; \
    esac; \
    curl -fsSL "https://download.docker.com/linux/static/stable/${dk}/docker-${DOCKER_VERSION}.tgz" -o /tmp/docker.tgz; \
    tar -xzf /tmp/docker.tgz -C /tmp; \
    install -m 0755 /tmp/docker/docker /usr/local/bin/docker; \
    install -m 0755 -d /usr/local/lib/docker/cli-plugins; \
    curl -fsSL "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${cp}" \
      -o /usr/local/lib/docker/cli-plugins/docker-compose; \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-compose; \
    rm -rf /tmp/docker /tmp/docker.tgz

USER airflow
# Dependencias para correr el pipeline y registrar en MLflow
RUN pip install --no-cache-dir \
    "mlflow==2.16.2" \
    xgboost scikit-learn pandas numpy pyarrow joblib \
    datasets loguru typer python-dotenv boto3
