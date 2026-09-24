-- Crea las dos bases que usa el stack: una para Airflow y otra para MLflow.
-- La base `airflow` ya la crea POSTGRES_DB; acá agregamos la de MLflow.
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
