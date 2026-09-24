"""Servicio gRPC: sirve el modelo de fraude (unary + server-streaming)."""

import os
from concurrent import futures

import grpc
import joblib
import pandas as pd

import ml_service_pb2
import ml_service_pb2_grpc

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# El modelo se carga UNA sola vez al iniciar (nunca por request).
ARTIFACT = joblib.load(MODEL_PATH)
PIPELINE = ARTIFACT["pipeline"]
VERSION = ARTIFACT["version"]
FEATURE_ORDER = ARTIFACT["feature_order"]


def _row_from_request(tx) -> dict:
    """Mensaje Transaction -> dict con las features del modelo."""
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


def _predict_one(tx) -> ml_service_pb2.Prediction:
    X = pd.DataFrame([_row_from_request(tx)], columns=FEATURE_ORDER)
    probability = float(PIPELINE.predict_proba(X)[0][1])
    is_fraud = bool(PIPELINE.predict(X)[0])
    return ml_service_pb2.Prediction(
        is_fraud=is_fraud,
        probability=round(probability, 4),
        model_version=VERSION,
    )


class MLServiceServicer(ml_service_pb2_grpc.MLServiceServicer):
    def Predict(self, request, context):
        return _predict_one(request)

    def PredictStream(self, request, context):
        for tx in request.items:
            yield _predict_one(tx)


def serve() -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    ml_service_pb2_grpc.add_MLServiceServicer_to_server(MLServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Servidor gRPC escuchando en el puerto 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
