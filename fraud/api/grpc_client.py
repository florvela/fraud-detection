"""Cliente gRPC del borde REST hacia el núcleo de scoring.

Cuando la env `GRPC_HOST` está seteada, el endpoint REST `/v1/predict` **delega**
el scoring en el núcleo gRPC (un solo motor de inferencia). Si no está seteada, la
API REST puntúa localmente con su propio modelo (modo desarrollo).
"""

from __future__ import annotations

import os

import grpc

from fraud.api.proto import fraud_pb2, fraud_pb2_grpc

GRPC_HOST = os.getenv("GRPC_HOST")
GRPC_PORT = os.getenv("GRPC_PORT", "50052")

_channel: grpc.Channel | None = None
_stub: fraud_pb2_grpc.FraudScoringStub | None = None


def delegates_to_grpc() -> bool:
    """True si el borde REST debe delegar el scoring en el núcleo gRPC."""
    return bool(GRPC_HOST)


def _get_stub() -> fraud_pb2_grpc.FraudScoringStub:
    global _channel, _stub
    if _stub is None:
        _channel = grpc.insecure_channel(f"{GRPC_HOST}:{GRPC_PORT}")
        _stub = fraud_pb2_grpc.FraudScoringStub(_channel)
    return _stub


def predict(row: dict) -> tuple[bool, float, str]:
    """Llama al núcleo gRPC (unary) y devuelve (is_fraud, probability, model_version)."""
    tx = fraud_pb2.Transaction(
        amt=float(row["amt"]),
        category=str(row["category"]),
        gender=str(row["gender"]),
        city_pop=int(row["city_pop"]),
        lat=float(row["lat"]),
        long=float(row["long"]),
        merch_lat=float(row["merch_lat"]),
        merch_long=float(row["merch_long"]),
        hour=int(row["hour"]),
        age=int(row["age"]),
    )
    pred = _get_stub().Predict(tx)
    return bool(pred.is_fraud), float(pred.probability), str(pred.model_version)


def get_model_info() -> tuple[str, str] | None:
    """Pide al núcleo gRPC la info del champion que REALMENTE decide.

    Devuelve (name, version), o None si no hay delegación o la llamada falla."""
    if not delegates_to_grpc():
        return None
    try:
        info = _get_stub().GetModelInfo(fraud_pb2.Empty())
        return str(info.name), str(info.version)
    except grpc.RpcError:
        return None


def reload_remote() -> str | None:
    """Pide al núcleo gRPC que recargue el champion (tras un deploy). Devuelve la
    versión nueva, o None si no hay delegación o la llamada falla."""
    if not delegates_to_grpc():
        return None
    try:
        info = _get_stub().Reload(fraud_pb2.Empty())
        return str(info.version)
    except grpc.RpcError:
        return None
