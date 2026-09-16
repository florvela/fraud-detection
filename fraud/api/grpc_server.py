"""Servidor gRPC que sirve el modelo de fraude (unary + server-streaming)."""

from __future__ import annotations

from concurrent import futures

import grpc
import pandas as pd

from fraud.api.model_loader import ModelStore
from fraud.api.proto import fraud_pb2, fraud_pb2_grpc

MODEL_NAME = "fraud-detection"
GRPC_PORT = 50052  # puerto local (evitamos 50051 que ya usa Multipass)

# El modelo se carga UNA sola vez al iniciar el servidor (nunca por request).
store = ModelStore()
store.load()


def _transaction_to_row(tx: fraud_pb2.Transaction) -> dict:
    """Convierte un mensaje Transaction a un dict con las features del modelo."""
    return {
        "amt": tx.amt,
        "category": tx.category,
        "gender": tx.gender,
        "city_pop": tx.city_pop,
        "lat": tx.lat,
        "long": tx.long,
        "merch_lat": tx.merch_lat,
        "merch_long": tx.merch_long,
        "hour": tx.hour,
        "age": tx.age,
    }


def _predict_one(tx: fraud_pb2.Transaction) -> fraud_pb2.Prediction:
    """Arma un DataFrame de una fila (en el orden que espera el modelo) y predice."""
    row = _transaction_to_row(tx)
    X = pd.DataFrame([row], columns=store.feature_order)
    probability = float(store.pipeline.predict_proba(X)[0][1])
    is_fraud = bool(store.pipeline.predict(X)[0])
    return fraud_pb2.Prediction(
        is_fraud=is_fraud,
        probability=round(probability, 4),
        model_version=store.version,
    )


class FraudScoringServicer(fraud_pb2_grpc.FraudScoringServicer):
    def Predict(self, request, context):
        """unary: 1 transacción -> 1 predicción."""
        if not store.loaded:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Modelo no cargado")
            return fraud_pb2.Prediction()
        return _predict_one(request)

    def PredictStream(self, request, context):
        """server streaming: recibe un lote y emite una predicción por transacción."""
        if not store.loaded:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Modelo no cargado")
            return
        for tx in request.items:
            yield _predict_one(tx)

    def GetModelInfo(self, request, context):
        """unary: devuelve sólo name + version (para comparar con REST y GraphQL)."""
        return fraud_pb2.ModelInfo(name=MODEL_NAME, version=store.version)


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    fraud_pb2_grpc.add_FraudScoringServicer_to_server(FraudScoringServicer(), server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")
    server.start()
    print(f"Servidor gRPC escuchando en localhost:{GRPC_PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
