"""Comparación batch vs streaming (Mini-TP 4).

Puntúa el MISMO conjunto de eventos de dos formas y compara tiempo/throughput:

    1) Online (streaming): un evento por vez -> `score_one` en un loop.
       Emula el costo por evento del consumer.
    2) Batch: todos los eventos de una sola pasada -> `score_batch`.

Toma los eventos de la cola local que dejó el producer (`--source sim`). Si la
cola está vacía, reconstruye el mismo conjunto con `build_event_pool` + drift.

Ejemplos:
    ./.venv/bin/python services/streaming/batch_compare.py
    ./.venv/bin/python services/streaming/batch_compare.py --total 400 --drift-at 0.5
"""

from __future__ import annotations

import random
import time

import numpy as np
import typer
from loguru import logger

import common

app = typer.Typer(add_completion=False, help="Compara scoring batch vs streaming.")


def _rebuild_events(total: int, drift_at: float, seed: int) -> list[dict]:
    """Reconstruye el mismo conjunto de eventos que emitiría el producer."""
    artifact = common.load_artifact()
    pool = common.build_event_pool(artifact, n=total, seed=seed)
    rng = random.Random(seed)
    drift_start = int(total * drift_at)
    events = []
    for i in range(total):
        event = dict(pool[i % len(pool)])
        if i >= drift_start:
            event = common.apply_drift(event, rng)
        events.append(event)
    return events


@app.command()
def run(
    total: int = typer.Option(400, help="Cantidad de eventos (si hay que reconstruir)."),
    drift_at: float = typer.Option(0.5, help="Fracción del flujo donde arranca el drift."),
    seed: int = typer.Option(42, help="Semilla (debe coincidir con el producer)."),
) -> None:
    """Corre ambos modos sobre el mismo set y muestra la comparación."""
    artifact = common.load_artifact()

    # Mismo conjunto de eventos: preferimos la cola local real del producer.
    events = common.read_sim_events()
    if events:
        logger.info("Usando {} eventos de la cola local {}", len(events), common.SIM_QUEUE_PATH)
    else:
        logger.info("Cola local vacía: reconstruyo {} eventos con seed={}.", total, seed)
        events = _rebuild_events(total, drift_at, seed)

    n = len(events)

    # --- 1) Streaming: un evento por vez ---
    t0 = time.perf_counter()
    stream_probas = [common.score_one(artifact, ev) for ev in events]
    t_stream = time.perf_counter() - t0

    # --- 2) Batch: todo de una sola pasada ---
    t0 = time.perf_counter()
    batch_probas = common.score_batch(artifact, events)
    t_batch = time.perf_counter() - t0

    # Sanity check: ambos caminos dan (prácticamente) las mismas probabilidades.
    max_diff = float(np.max(np.abs(np.asarray(stream_probas) - np.asarray(batch_probas))))

    thr_stream = n / t_stream if t_stream else 0.0
    thr_batch = n / t_batch if t_batch else 0.0
    speedup = t_stream / t_batch if t_batch else float("inf")

    print("\n" + "=" * 64)
    print(f"COMPARACIÓN BATCH vs STREAMING  ({n} eventos)")
    print("=" * 64)
    print(f"{'Modo':<14}{'Tiempo (s)':>14}{'Throughput (ev/s)':>22}")
    print("-" * 64)
    print(f"{'streaming':<14}{t_stream:>14.4f}{thr_stream:>22.1f}")
    print(f"{'batch':<14}{t_batch:>14.4f}{thr_batch:>22.1f}")
    print("-" * 64)
    print(f"Speedup batch vs streaming : {speedup:>8.1f}x")
    print(f"Latencia media por evento  : streaming={t_stream / n * 1000:.3f} ms | "
          f"batch={t_batch / n * 1000:.3f} ms")
    print(f"Máx. diferencia de proba   : {max_diff:.2e}  (deben coincidir)")
    print("=" * 64 + "\n")

    logger.success(
        "Batch es ~{:.0f}x más rápido por evento; streaming gana en latencia de reacción.",
        speedup,
    )


if __name__ == "__main__":
    app()
