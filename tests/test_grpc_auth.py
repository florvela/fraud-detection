"""Test de la autenticación del núcleo gRPC (SAIF capa 1).

Levanta un servidor gRPC real en un puerto efímero con el interceptor de API key y
verifica que: sin key (o con key inválida) aborta con UNAUTHENTICATED, y con la key
correcta responde. Cubre tanto el método unary como el streaming.
"""

from concurrent import futures

import grpc
import pytest

pytest.importorskip("grpc")

from fraud.api.grpc_server import ApiKeyInterceptor, FraudScoringServicer, store  # noqa: E402
from fraud.api.proto import fraud_pb2, fraud_pb2_grpc  # noqa: E402

VALID = (("x-api-key", "clave-valida"),)
INVALID = (("x-api-key", "clave-mala"),)

EXAMPLE = fraud_pb2.Transaction(
    amt=120.5, category="grocery_pos", gender="F", city_pop=50000,
    lat=40.1, long=-74.5, merch_lat=40.3, merch_long=-74.2, hour=2, age=35,
)


@pytest.fixture(scope="module")
def grpc_channel():
    if not store.loaded:
        pytest.skip("modelo no cargado en el entorno de test")
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2),
        interceptors=[ApiKeyInterceptor({"clave-valida"})],
    )
    fraud_pb2_grpc.add_FraudScoringServicer_to_server(FraudScoringServicer(), server)
    port = server.add_insecure_port("[::]:0")  # puerto efímero
    server.start()
    channel = grpc.insecure_channel(f"localhost:{port}")
    yield channel
    channel.close()
    server.stop(grace=None)


def _stub(channel):
    return fraud_pb2_grpc.FraudScoringStub(channel)


def test_sin_key_aborta_unauthenticated(grpc_channel):
    with pytest.raises(grpc.RpcError) as exc:
        _stub(grpc_channel).Predict(EXAMPLE)  # sin metadata
    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_key_invalida_aborta_unauthenticated(grpc_channel):
    with pytest.raises(grpc.RpcError) as exc:
        _stub(grpc_channel).Predict(EXAMPLE, metadata=INVALID)
    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED


def test_key_valida_responde(grpc_channel):
    pred = _stub(grpc_channel).Predict(EXAMPLE, metadata=VALID)
    assert isinstance(pred.is_fraud, bool)
    assert 0.0 <= pred.probability <= 1.0


def test_streaming_sin_key_tambien_aborta(grpc_channel):
    batch = fraud_pb2.TransactionBatch(items=[EXAMPLE, EXAMPLE])
    with pytest.raises(grpc.RpcError) as exc:
        list(_stub(grpc_channel).PredictStream(batch))  # sin metadata
    assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED
