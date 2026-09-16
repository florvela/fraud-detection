"""Compara la latencia de predecir la MISMA transacción por REST, GraphQL y gRPC.

Antes de correr, levanta los servidores:
    # REST + GraphQL (misma app FastAPI)
    ./.venv/bin/uvicorn fraud.api.main:app --port 8080
    # gRPC (en otra terminal)
    ./.venv/bin/python -m fraud.api.grpc_server
"""

import time

import grpc
import requests

from fraud.api.grpc_server import GRPC_PORT
from fraud.api.proto import fraud_pb2, fraud_pb2_grpc

N = 200  # cantidad de llamadas para promediar

REST_URL = "http://localhost:8080/v1/predict"
GRAPHQL_URL = "http://localhost:8080/graphql"
HEADERS_REST = {"X-API-KEY": "token-secreto-123"}

# La misma transacción, expresada para cada protocolo.
TX_JSON = {
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

TX_GRPC = fraud_pb2.Transaction(**TX_JSON)


def time_calls(fn) -> float:
    """Corre fn() N veces y devuelve el promedio en milisegundos."""
    t = time.perf_counter()
    for _ in range(N):
        fn()
    return (time.perf_counter() - t) / N * 1000


def main() -> None:
    # Un cliente/canal por protocolo, reutilizado en todas las llamadas.
    session = requests.Session()
    channel = grpc.insecure_channel(f"localhost:{GRPC_PORT}")
    stub = fraud_pb2_grpc.FraudScoringStub(channel)

    def call_rest():
        session.post(REST_URL, json=TX_JSON, headers=HEADERS_REST).json()

    def call_graphql():
        session.post(GRAPHQL_URL, json={"query": GQL_QUERY, "variables": {"tx": TX_GQL}}).json()

    def call_grpc():
        stub.Predict(TX_GRPC)

    try:
        # Una llamada de prueba a cada uno para validar que responden bien.
        rest_sample = session.post(REST_URL, json=TX_JSON, headers=HEADERS_REST).json()
        gql_sample = session.post(
            GRAPHQL_URL, json={"query": GQL_QUERY, "variables": {"tx": TX_GQL}}
        ).json()["data"]["predict"]
        grpc_sample = stub.Predict(TX_GRPC)
    except (requests.exceptions.ConnectionError, grpc.RpcError):
        print(
            "[ERROR] Faltan servidores. Levanta en dos terminales:\n"
            "    ./.venv/bin/uvicorn fraud.api.main:app --port 8080\n"
            "    ./.venv/bin/python -m fraud.api.grpc_server"
        )
        channel.close()
        return

    print("Misma predicción por los tres protocolos:")
    print(f"  REST    -> {rest_sample}")
    print(f"  GraphQL -> {gql_sample}")
    print(f"  gRPC    -> is_fraud={grpc_sample.is_fraud} probability={grpc_sample.probability}\n")

    rest_ms = time_calls(call_rest)
    gql_ms = time_calls(call_graphql)
    grpc_ms = time_calls(call_grpc)

    print(f"Latencia media sobre {N} llamadas:")
    print(f"  REST    : {rest_ms:6.3f} ms/llamada")
    print(f"  GraphQL : {gql_ms:6.3f} ms/llamada")
    print(f"  gRPC    : {grpc_ms:6.3f} ms/llamada")
    print(f"\n  gRPC es ~{rest_ms / grpc_ms:.1f}x más rápido que REST en esta prueba local.")

    channel.close()


if __name__ == "__main__":
    main()
