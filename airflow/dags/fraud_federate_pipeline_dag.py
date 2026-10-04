"""DAG del pipeline FEDERADO: silos -> entrenamiento Flower -> evaluación -> registro.

A diferencia de ``fraud_pipeline`` (entrenamiento centralizado de un XGBoost), este
DAG orquesta el camino de **aprendizaje federado**: Airflow no entrena, sino que
**coordina los contenedores reales** de Flower (un servidor + N bancos). Cada banco
entrena sobre su silo y nunca comparte sus transacciones; solo viajan los pesos del
MLP, que el servidor promedia (FedAvg).

Pasos
-----
1. **partition_silos**: particiona el train en silos no-IID por banco
   (``banco_a.parquet`` / ``banco_b.parquet``), simulando dos regiones.
2. **federated_train**: levanta ``fed-server`` + ``fed-client-a/b`` del compose
   (perfil ``federated``) y espera a que el servidor complete los rounds y guarde
   el modelo global en ``models/mlp_federado.npz``.
3. **evaluate_vs_central**: compara el federado contra el champion XGBoost sobre el
   test y vuelca las métricas a ``models/federated_metrics.json``.
4. **register_federated**: envuelve el MLP en un pipeline sklearn y lo registra en
   MLflow como el alias ``federated`` del modelo ``fraud-detection`` (queda listo
   para correr en sombra y, opcionalmente, promoverse a champion desde la UI).

Orquestación de los contenedores (DooD)
---------------------------------------
Airflow dispara los contenedores Flower vía ``docker compose`` usando el socket de
Docker del host (montado en el compose) + el CLI de docker instalado en la imagen de
Airflow. Se fija ``-p fraud-detection`` para reutilizar la red/imágenes del stack
ya levantado (el proyecto por defecto sería el nombre del directorio).

Se dispara a mano (``schedule=None``): es un entrenamiento pesado y deliberado.
"""

from __future__ import annotations

from datetime import datetime, timedelta
import os

from airflow import DAG
from airflow.operators.bash import BashOperator

# El repo se monta en /opt/airflow/repo (ver docker-compose.yml). Esta ruta se usa
# para los pasos que corren DENTRO del worker de Airflow (partition, register).
REPO = "/opt/airflow/repo"

# Ruta del repo EN EL HOST (inyectada por el compose como HOST_REPO_DIR). Los pasos
# que llaman a `docker compose` hablan con el daemon del host, así que deben usar
# rutas del host; el repo está montado también en esa misma ruta (${PWD}:${PWD}).
HOST_REPO = os.environ.get("HOST_REPO_DIR", REPO)
COMPOSE = f"{HOST_REPO}/docker-compose.yml"
# Nombre del proyecto compose del stack levantado desde el host (dir del repo).
# Fijarlo hace que los contenedores fed-* reutilicen la red/imágenes existentes.
PROJECT = "fraud-detection"

# Prefijo común: compose apuntando al archivo/proyecto/directorio del host.
DC = f"docker compose -p {PROJECT} --project-directory {HOST_REPO} -f {COMPOSE}"

default_args = {
    "owner": "mlops2",
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="fraud_federate_pipeline",
    description="Aprendizaje federado (Flower FedAvg) + registro del modelo global en MLflow",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,  # manual: entrenamiento pesado y deliberado
    catchup=False,
    tags=["fraude", "mlops2", "federado", "flower"],
) as dag:
    # 1. Silos no-IID por banco (Oeste/Este). Corre en el worker de Airflow:
    #    data_partition importa `model` (constantes) con torch diferido -> no necesita torch.
    partition_silos = BashOperator(
        task_id="partition_silos",
        bash_command=(
            f"cd {REPO}/services/federated && "
            f"python data_partition.py --out-dir {REPO}/services/federated/data"
        ),
    )

    # 2. Entrenamiento federado con los contenedores reales (servidor + 2 bancos).
    #    --abort-on-container-exit: cuando el servidor termina sus rounds (y guarda el
    #    npz), compose frena a los clientes. --exit-code-from: el task hereda el código
    #    de salida del servidor.
    federated_train = BashOperator(
        task_id="federated_train",
        bash_command=(
            f"{DC} --profile federated up --no-build "
            f"--abort-on-container-exit --exit-code-from fed-server "
            f"fed-server fed-client-a fed-client-b"
        ),
    )

    # 3. Federado vs champion centralizado sobre el test; vuelca métricas a JSON.
    #    Corre dentro de la imagen federada (tiene torch) con `compose run`.
    evaluate = BashOperator(
        task_id="evaluate_vs_central",
        bash_command=(
            f"{DC} run --rm --no-deps fed-server "
            f"python evaluate_vs_central.py --metrics-out /app/models/federated_metrics.json"
        ),
    )

    # 4. Envuelve el MLP en pipeline sklearn y lo registra en MLflow (alias federated).
    register = BashOperator(
        task_id="register_federated",
        bash_command=f"cd {REPO} && python airflow/scripts/register_federated.py",
    )

    partition_silos >> federated_train >> evaluate >> register
