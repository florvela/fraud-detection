"""Cliente gRPC de prueba: llama al servicio de fraude (unary + streaming)."""

import grpc

from fraud.api.grpc_server import GRPC_PORT
from fraud.api.proto import fraud_pb2, fraud_pb2_grpc


def separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# Una transacción de ejemplo (mismos valores que usamos en el cliente REST).
EXAMPLE = fraud_pb2.Transaction(
    amt=950.75,
    category="shopping_net",
    gender="F",
    city_pop=15000,
    lat=40.1,
    long=-74.5,
    merch_lat=41.9,
    merch_long=-80.2,
    hour=2,
    age=35,
)


def main() -> None:
    # Abrimos UN canal y lo reutilizamos (crear uno por llamada es caro).
    channel = grpc.insecure_channel(f"localhost:{GRPC_PORT}")
    stub = fraud_pb2_grpc.FraudScoringStub(channel)

    try:
        # --- unary: 1 -> 1 ---
        separator("1) unary  (Predict)")
        r = stub.Predict(EXAMPLE)
        print(f"is_fraud={r.is_fraud} | probability={r.probability} | version={r.model_version}")

        # --- server streaming: 1 lote -> N predicciones ---
        separator("2) streaming  (PredictStream sobre un lote de 3)")
        batch = fraud_pb2.TransactionBatch(items=[EXAMPLE, EXAMPLE, EXAMPLE])
        for i, p in enumerate(stub.PredictStream(batch), start=1):
            print(f"  item {i}: is_fraud={p.is_fraud} | probability={p.probability}")

        # --- metadatos ---
        separator("3) metadatos  (GetModelInfo)")
        info = stub.GetModelInfo(fraud_pb2.Empty())
        print(f"name={info.name} | version={info.version}")
    except grpc.RpcError as err:
        print(
            f"\n[ERROR] No se pudo hablar con el servidor gRPC ({err.code()}).\n"
            "Proba levantarlo:\n"
            "    ./.venv/bin/python -m fraud.api.grpc_server"
        )
    finally:
        channel.close()


if __name__ == "__main__":
    main()
