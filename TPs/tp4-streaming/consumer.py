"""Consumer del flujo de transacciones (Mini-TP 4).

Consume el flujo de transacciones, carga el modelo UNA sola vez y scorea cada
evento online. Calcula métricas por ventana (throughput, latencia p95, tasa de
fraude y un indicador de drift PSI sobre `amt`) y dispara alertas cuando una
métrica cruza su umbral.

Modos (`--source`):
    kafka -> consume del topic `transactions`; publica los eventos puntuados a
             `scored-transactions` y las alertas a `fraud-alerts`.
    sim   -> lee la cola local (JSON Lines) que dejó el producer, sin broker;
             escribe los puntuados/alertas a archivos locales equivalentes.

Ejemplos:
    ./.venv/bin/python TPs/tp4-streaming/consumer.py --source sim --window 100
    ./.venv/bin/python TPs/tp4-streaming/consumer.py --source kafka \\
        --bootstrap-servers localhost:9092
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import typer
from loguru import logger

import common

app = typer.Typer(add_completion=False, help="Consumer + scoring online de transacciones.")

# Topics/archivos de salida (la consigna pide 'scored-transactions' además de alertas).
TOPIC_SCORED = "scored-transactions"
SIM_SCORED_PATH = common.STREAM_DIR / "scored.jsonl"

# Feature numérica que vigilamos para drift.
DRIFT_FEATURE = "amt"


# ---------------------------------------------------------------------------
# Agregador de métricas por ventana
# ---------------------------------------------------------------------------
@dataclass
class Window:
    """Acumula métricas de una ventana de N eventos."""

    window_id: int
    latencies_ms: list[float] = field(default_factory=list)
    amt_values: list[float] = field(default_factory=list)
    fraud_flags: list[int] = field(default_factory=list)
    t_start: float = field(default_factory=time.perf_counter)

    def add(self, amt: float, proba: float, latency_ms: float, fraud_threshold: float) -> None:
        self.latencies_ms.append(latency_ms)
        self.amt_values.append(float(amt))
        self.fraud_flags.append(int(proba >= fraud_threshold))

    def close(self, baseline_amt: np.ndarray | None) -> dict[str, Any]:
        elapsed = max(time.perf_counter() - self.t_start, 1e-9)
        n = len(self.latencies_ms)
        lat = np.asarray(self.latencies_ms)
        psi = (
            common.population_stability_index(baseline_amt, np.asarray(self.amt_values))
            if baseline_amt is not None
            else 0.0
        )
        return {
            "window_id": self.window_id,
            "n": n,
            "throughput_evs": n / elapsed,
            "p95_latency_ms": float(np.percentile(lat, 95)),
            "mean_latency_ms": float(lat.mean()),
            "fraud_rate": float(np.mean(self.fraud_flags)) if n else 0.0,
            "psi_amt": psi,
        }


# ---------------------------------------------------------------------------
# Fuentes de eventos
# ---------------------------------------------------------------------------
def _iter_sim() -> Iterator[dict[str, Any]]:
    """Lee los eventos de la cola local escrita por el producer en modo sim."""
    events = common.read_sim_events()
    if not events:
        raise typer.BadParameter(
            f"La cola local {common.SIM_QUEUE_PATH} está vacía. "
            "Corré primero el producer con --source sim."
        )
    logger.info("Leídos {} eventos de la cola local {}", len(events), common.SIM_QUEUE_PATH)
    yield from events


def _iter_kafka(bootstrap_servers: str, topic: str, timeout_ms: int) -> Iterator[dict[str, Any]]:
    """Consume eventos del topic Kafka hasta agotar/timeout."""
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers.split(","),
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="fraud-stream-consumer",
        consumer_timeout_ms=timeout_ms,
    )
    logger.info("Consumiendo topic '{}' de {}", topic, bootstrap_servers)
    for msg in consumer:
        yield msg.value
    consumer.close()


# ---------------------------------------------------------------------------
# Sinks de salida
# ---------------------------------------------------------------------------
class Sinks:
    """Encapsula a dónde van los eventos puntuados y las alertas según el modo."""

    def __init__(self, source: str, bootstrap_servers: str) -> None:
        self.source = source
        self._producer = None
        if source == "kafka":
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                linger_ms=10,
            )
        else:
            common.ensure_stream_dir()
            SIM_SCORED_PATH.write_text("", encoding="utf-8")
            Path(common.SIM_ALERTS_PATH).write_text("", encoding="utf-8")

    def scored(self, payload: dict[str, Any]) -> None:
        if self._producer is not None:
            self._producer.send(TOPIC_SCORED, value=payload)
        else:
            with SIM_SCORED_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def alert(self, payload: dict[str, Any]) -> None:
        if self._producer is not None:
            self._producer.send(common.TOPIC_ALERTS, value=payload)
        else:
            common.append_sim_alert(payload)

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush()
            self._producer.close()


# ---------------------------------------------------------------------------
# Comando principal
# ---------------------------------------------------------------------------
@app.command()
def run(
    source: str = typer.Option("sim", help="Origen del flujo: 'kafka' o 'sim'."),
    window: int = typer.Option(100, help="Tamaño de ventana en cantidad de eventos."),
    bootstrap_servers: str = typer.Option("localhost:9092", help="Broker(s) Kafka (modo kafka)."),
    topic: str = typer.Option(common.TOPIC_TRANSACTIONS, help="Topic de entrada (modo kafka)."),
    timeout_ms: int = typer.Option(10000, help="Corte por inactividad del consumer Kafka."),
    fraud_threshold: float = typer.Option(0.5, help="Umbral de probabilidad para marcar fraude."),
    psi_threshold: float = typer.Option(0.2, help="Umbral de PSI para alertar drift."),
    fraud_rate_threshold: float = typer.Option(
        0.05, help="Umbral de tasa de fraude por ventana para alertar."
    ),
) -> None:
    """Consume, scorea online y reporta métricas por ventana con alertas."""
    if source not in {"kafka", "sim"}:
        raise typer.BadParameter("source debe ser 'kafka' o 'sim'.")

    artifact = common.load_artifact()  # <-- el modelo se carga UNA sola vez
    sinks = Sinks(source, bootstrap_servers)

    if source == "sim":
        events = _iter_sim()
    else:
        events = _iter_kafka(bootstrap_servers, topic, timeout_ms)

    baseline_amt: np.ndarray | None = None  # se fija con la 1ª ventana
    win = Window(window_id=0)
    n_total = 0
    n_fraud_total = 0
    n_alerts = 0
    t0 = time.perf_counter()

    def close_window(w: Window) -> None:
        nonlocal baseline_amt, n_alerts
        m = w.close(baseline_amt)
        logger.info(
            "[ventana {:>2}] n={:>4} | throughput={:8.1f} ev/s | p95={:6.2f} ms | "
            "fraude={:6.2%} | PSI(amt)={:5.3f}",
            m["window_id"], m["n"], m["throughput_evs"], m["p95_latency_ms"],
            m["fraud_rate"], m["psi_amt"],
        )
        if baseline_amt is None:  # la primera ventana define el baseline de drift
            baseline_amt = np.asarray(w.amt_values)

        # --- Alertas por cruce de umbral ---
        alerts = []
        if m["psi_amt"] > psi_threshold:
            alerts.append(f"DRIFT alto en '{DRIFT_FEATURE}': PSI={m['psi_amt']:.3f} > {psi_threshold}")
        if m["fraud_rate"] > fraud_rate_threshold:
            alerts.append(
                f"TASA DE FRAUDE alta: {m['fraud_rate']:.2%} > {fraud_rate_threshold:.2%}"
            )
        for a in alerts:
            logger.warning("ALERTA (ventana {}): {}", m["window_id"], a)
            sinks.alert({**m, "alert": a, "ts": time.time()})
        n_alerts += len(alerts)

    for event in events:
        # --- inferencia online, cronometrada por evento ---
        t_score = time.perf_counter()
        proba = common.score_one(artifact, event)
        latency_ms = (time.perf_counter() - t_score) * 1000.0

        is_fraud = int(proba >= fraud_threshold)
        n_total += 1
        n_fraud_total += is_fraud

        sinks.scored(
            {
                "event_id": event.get("event_id"),
                "fraud_proba": round(proba, 6),
                "is_fraud_pred": is_fraud,
                "model_version": artifact.get("version"),
                "ts_scored": time.time(),
            }
        )

        win.add(event[DRIFT_FEATURE], proba, latency_ms, fraud_threshold)
        if len(win.latencies_ms) >= window:
            close_window(win)
            win = Window(window_id=win.window_id + 1)

    if win.latencies_ms:  # cerrar ventana parcial final
        close_window(win)

    sinks.close()
    elapsed = time.perf_counter() - t0
    logger.success(
        "Consumer terminó: {} eventos en {:.2f}s ({:.1f} ev/s) | fraudes={} ({:.2%}) | alertas={}",
        n_total, elapsed, (n_total / elapsed) if elapsed else 0.0,
        n_fraud_total, (n_fraud_total / n_total) if n_total else 0.0, n_alerts,
    )


if __name__ == "__main__":
    app()
