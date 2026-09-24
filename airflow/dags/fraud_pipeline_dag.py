"""DAG del pipeline de fraude: ingesta -> features -> train -> registro en MLflow.

Reutiliza los mismos módulos del paquete `fraud` que se corren en local
(`python -m fraud.dataset`, etc.), de modo que el pipeline dockerizado y el local
son el MISMO código. El paso final registra el modelo en MLflow y promueve el
champion (ver `airflow/scripts/register_model.py`).

Sirve para el arranque en frío (seed): al levantar el compose se dispara una vez
para dejar un champion disponible, y luego corre programado para reentrenar.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# El repo se monta en /opt/airflow/repo (ver docker-compose.yml)
REPO = "/opt/airflow/repo"

default_args = {
    "owner": "mlops2",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="fraud_pipeline",
    description="Ingesta + features + entrenamiento + registro en MLflow (champion/challenger)",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["fraude", "mlops2", "retrain"],
) as dag:
    # 1. Ingesta: baja una muestra con la distribución real (~1% fraude)
    ingest = BashOperator(
        task_id="ingest",
        bash_command=f"cd {REPO} && python -m fraud.dataset --n-rows 200000",
    )

    # 2. Features + split estratificado train/val/test
    features = BashOperator(
        task_id="features",
        bash_command=f"cd {REPO} && python -m fraud.features",
    )

    # 3. Entrenamiento (rebalanceo solo en train) -> models/model.joblib
    train = BashOperator(
        task_id="train",
        bash_command=f"cd {REPO} && python -m fraud.modeling.train",
    )

    # 4. Registro en MLflow + promoción champion/challenger
    register = BashOperator(
        task_id="register_model",
        bash_command=f"cd {REPO} && python airflow/scripts/register_model.py",
    )

    ingest >> features >> train >> register
