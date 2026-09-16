"""Cliente que compara la latencia de REST, GraphQL y gRPC prediciendo lo mismo.

Hosts configurables por variable de entorno:
  - en tu terminal local (contra los puertos publicados): usa los defaults.
  - dentro de la red de Docker: se pasan por env en docker-compose (nombres de servicio).
"""

import json
import os
import time

import grpc
import requests

import ml_service_pb2
import ml_service_pb2_grpc

# Configuración (defaults para correr desde tu terminal local).
GRPC_HOST = os.getenv("GRPC_HOST", "localhost:50052")
GRAPHQL_URL = os.getenv("GRAPHQL_URL", "http://localhost:8000/graphql")
REST_URL = os.getenv("REST_URL", "http://localhost:8001/predict")
NUM_REQUESTS = int(os.getenv("NUM_REQUESTS", "500"))

# La misma transacción para los tres protocolos.
TX = {
    "amt": 950.75,
    "category": "shopping_net",
    "gender": "F",
    "city_pop": 15000,
    "lat": 40.1,
    "long": -74.5,
    "merch_lat": 41.9,
    "merch_long": -80.2,
    "hour": 2,
    "age": 35,
}

# GraphQL usa camelCase (Strawberry lo convierte automáticamente).
TX_GQL = {
    "amt": 950.75,
    "category": "shopping_net",
    "gender": "F",
    "cityPop": 15000,
    "lat": 40.1,
    "long": -74.5,
    "merchLat": 41.9,
    "merchLong": -80.2,
    "hour": 2,
    "age": 35,
}
GQL_QUERY = """
query Predict($tx: TransactionInput!) {
  predict(transaction: $tx) { isFraud probability modelVersion }
}
"""


def run_grpc_client() -> float:
    start = time.time()
    with grpc.insecure_channel(GRPC_HOST) as channel:
        stub = ml_service_pb2_grpc.MLServiceStub(channel)
        request = ml_service_pb2.Transaction(**TX)
        for _ in range(NUM_REQUESTS):
            stub.Predict(request)
    return time.time() - start


def run_graphql_client() -> float:
    start = time.time()
    session = requests.Session()
    payload = {"query": GQL_QUERY, "variables": {"tx": TX_GQL}}
    for _ in range(NUM_REQUESTS):
        resp = session.post(GRAPHQL_URL, data=json.dumps(payload),
                            headers={"Content-Type": "application/json"})
        resp.raise_for_status()
    return time.time() - start


def run_rest_client() -> float:
    start = time.time()
    session = requests.Session()
    for _ in range(NUM_REQUESTS):
        resp = session.post(REST_URL, data=json.dumps(TX),
                            headers={"Content-Type": "application/json"})
        resp.raise_for_status()
    return time.time() - start


if __name__ == "__main__":
    print(f"Realizando {NUM_REQUESTS} solicitudes a cada servicio...\n")

    grpc_time = run_grpc_client()
    graphql_time = run_graphql_client()
    rest_time = run_rest_client()

    print(f"gRPC    : {grpc_time:.4f} s total | {grpc_time / NUM_REQUESTS * 1000:.4f} ms/llamada")
    print(f"GraphQL : {graphql_time:.4f} s total | {graphql_time / NUM_REQUESTS * 1000:.4f} ms/llamada")
    print(f"REST    : {rest_time:.4f} s total | {rest_time / NUM_REQUESTS * 1000:.4f} ms/llamada")
    print(f"\ngRPC es ~{rest_time / grpc_time:.1f}x más rápido que REST en esta prueba.")
