"""Demo gRPC del TP3 contra el grpc_service de este TP (no el de prod).

Muestra los dos métodos del contrato `proto/ml_service.proto`:
  1) unary          Predict(Transaction) -> Prediction
  2) server-stream  PredictStream(TransactionBatch) -> stream Prediction

Usa los stubs `ml_service_pb2*` (generados del .proto; en Docker los regenera el
Dockerfile del cliente). Host configurable por env (default: puerto publicado):
    GRPC_HOST=localhost:50052 python demo_grpc.py
"""

import os

import grpc

import ml_service_pb2
import ml_service_pb2_grpc

GRPC_HOST = os.getenv("GRPC_HOST", "localhost:50052")

TX = ml_service_pb2.Transaction(
    amt=950.75, category="shopping_net", gender="F", city_pop=15000,
    lat=40.1, long=-74.5, merch_lat=41.9, merch_long=-80.2, hour=2, age=35,
)


def sep(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def main() -> None:
    channel = grpc.insecure_channel(GRPC_HOST)
    stub = ml_service_pb2_grpc.MLServiceStub(channel)
    try:
        sep("1) unary  (Predict)")
        r = stub.Predict(TX)
        print(f"is_fraud={r.is_fraud} | probability={r.probability} | version={r.model_version}")

        sep("2) streaming  (PredictStream sobre un lote de 3)")
        batch = ml_service_pb2.TransactionBatch(items=[TX, TX, TX])
        for i, p in enumerate(stub.PredictStream(batch), start=1):
            print(f"  item {i}: is_fraud={p.is_fraud} | probability={p.probability}")
    except grpc.RpcError as err:
        print(f"\n[ERROR] No respondió el grpc_service en {GRPC_HOST} ({err.code()}).\n"
              "Levantá el TP: docker compose up --build -d")
    finally:
        channel.close()


if __name__ == "__main__":
    main()
