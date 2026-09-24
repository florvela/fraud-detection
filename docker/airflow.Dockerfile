# Imagen de Airflow con las dependencias del proyecto de fraude.
# Airflow corre el pipeline reutilizando el paquete `fraud` montado por volumen.
FROM apache/airflow:2.9.3-python3.12

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
# Dependencias para correr el pipeline y registrar en MLflow
RUN pip install --no-cache-dir \
    "mlflow==2.16.2" \
    xgboost scikit-learn pandas numpy pyarrow joblib \
    datasets loguru typer python-dotenv boto3
