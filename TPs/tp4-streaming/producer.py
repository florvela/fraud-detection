"""Producer del flujo de transacciones (Mini-TP 4).

Genera un flujo de transacciones con las features del modelo y las publica al
topic Kafka `transactions`. A mitad del flujo introduce un cambio de distribución
(sube la media de `amt` y cambia la mezcla de `category`) para poder detectar drift.

Modos (`--source`):
    kafka -> publica al broker con kafka-python.
    sim   -> escribe a una cola local (JSON Lines) para probar sin broker.

Ejemplos:
    ./.venv/bin/python TPs/tp4-streaming/producer.py --source sim --total 400
    ./.venv/bin/python TPs/tp4-streaming/producer.py --source kafka \\
        --bootstrap-servers localhost:9092 --total 1000 --rate 200
"""

from __future__ import annotations

import json
import random
import time

import typer
from loguru import logger

import common

app = typer.Typer(add_completion=False, help="Producer de transacciones para Kafka.")


def _make_kafka_producer(bootstrap_servers: str):
    """Crea un KafkaProducer que serializa cada evento a JSON."""
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=bootstrap_servers.split(","),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        linger_ms=10,
    )


@app.command()
def run(
    source: str = typer.Option("sim", help="Destino del flujo: 'kafka' o 'sim'."),
    total: int = typer.Option(400, help="Cantidad total de eventos a emitir."),
    rate: float = typer.Option(100.0, help="Eventos por segundo (throttling)."),
    drift_at: float = typer.Option(
        0.5, help="Fracción del flujo (0-1) a partir de la cual arranca el drift."
    ),
    bootstrap_servers: str = typer.Option(
        "localhost:9092", help="Broker(s) Kafka (solo modo kafka)."
    ),
    topic: str = typer.Option(common.TOPIC_TRANSACTIONS, help="Topic destino."),
    seed: int = typer.Option(42, help="Semilla para reproducibilidad."),
) -> None:
    """Emite `total` transacciones; la segunda mitad va con distribución cambiada."""
    artifact = common.load_artifact()
    pool = common.build_event_pool(artifact, n=total, seed=seed)
    rng = random.Random(seed)

    drift_start = int(total * drift_at)
    logger.info(
        "Emitiendo {} eventos a '{}' (source={}); drift a partir del evento {}.",
        total, topic, source, drift_start,
    )

    producer = None
    if source == "kafka":
        producer = _make_kafka_producer(bootstrap_servers)
    elif source == "sim":
        common.reset_sim_queue()
    else:
        raise typer.BadParameter("source debe ser 'kafka' o 'sim'.")

    sleep_s = 1.0 / rate if rate > 0 else 0.0
    n_drift = 0
    for i in range(total):
        event = dict(pool[i % len(pool)])
        drifted = i >= drift_start
        if drifted:
            event = common.apply_drift(event, rng)
            n_drift += 1
        # Metadatos útiles para el consumer (no son features del modelo).
        event["event_id"] = i
        event["ts"] = time.time()
        event["drift"] = drifted

        if source == "kafka":
            producer.send(topic, value=event)
        else:
            common.append_sim_event(event)

        if (i + 1) % 100 == 0:
            logger.info("  emitidos {}/{} eventos...", i + 1, total)
        if sleep_s:
            time.sleep(sleep_s)

    if producer is not None:
        producer.flush()
        producer.close()

    logger.success(
        "Flujo completo: {} eventos ({} normales + {} con drift).",
        total, total - n_drift, n_drift,
    )
    if source == "sim":
        logger.info("Cola local escrita en {}", common.SIM_QUEUE_PATH)


if __name__ == "__main__":
    app()
