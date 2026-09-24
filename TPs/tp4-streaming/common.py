"""Utilidades compartidas del servicio de streaming (Mini-TP 4).

Reúne todo lo común entre `producer.py`, `consumer.py` y `batch_compare.py`:
carga del modelo, inferencia (batch), muestreo de features reales/sintéticas,
la cola local para el modo `sim` (sin broker) y el cálculo de PSI para drift.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from loguru import logger

# --- Rutas del proyecto ---------------------------------------------------
# common.py vive en services/streaming/ -> la raíz del repo son 2 niveles arriba.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_sample.parquet"

# Cola local para el modo `sim` (funciona sin Kafka): un archivo JSON Lines.
STREAM_DIR = Path(__file__).resolve().parent / "data"
SIM_QUEUE_PATH = STREAM_DIR / "stream.jsonl"
SIM_ALERTS_PATH = STREAM_DIR / "alerts.jsonl"

# Nombres de topics Kafka (coinciden con la consigna).
TOPIC_TRANSACTIONS = "transactions"
TOPIC_ALERTS = "fraud-alerts"


# --- Modelo ---------------------------------------------------------------
def load_artifact(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Carga el dict de joblib con el pipeline y metadatos del modelo."""
    import joblib

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. "
            "En Docker el modelo llega por volumen montado en /app/models."
        )
    artifact = joblib.load(model_path)
    logger.info(
        "Modelo cargado: version={} entrenado={}",
        artifact.get("version"),
        artifact.get("trained_at"),
    )
    return artifact


def score_batch(artifact: dict[str, Any], events: Iterable[dict[str, Any]]) -> np.ndarray:
    """Puntúa una lista de eventos en una sola pasada.

    Construye un DataFrame con las columnas exactas de `feature_order` y devuelve
    la probabilidad de fraude (clase positiva) para cada fila.
    """
    feature_order = artifact["feature_order"]
    df = pd.DataFrame(list(events), columns=feature_order)
    proba = artifact["pipeline"].predict_proba(df)[:, 1]
    return proba


def score_one(artifact: dict[str, Any], event: dict[str, Any]) -> float:
    """Puntúa un único evento (inferencia online). Devuelve la probabilidad."""
    return float(score_batch(artifact, [event])[0])


# --- Generación de eventos ------------------------------------------------
def _load_real_pool(data_path: str | Path, n: int, seed: int) -> list[dict[str, Any]] | None:
    """Muestrea features reales del parquet. Devuelve None si no está disponible.

    El parquet trae `trans_date_trans_time` y `dob` en vez de `hour`/`age`,
    así que derivamos esas dos columnas que el modelo espera.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        return None
    try:
        df = pd.read_parquet(data_path)
    except Exception as exc:  # pragma: no cover - depende de pyarrow instalado
        logger.warning("No se pudo leer el parquet ({}), uso datos sintéticos.", exc)
        return None

    df = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    # Derivamos `hour` y `age` si vienen las columnas crudas.
    if "trans_date_trans_time" in df.columns:
        df["hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour
    else:
        df["hour"] = np.random.randint(0, 24, size=len(df))
    if "dob" in df.columns:
        dob = pd.to_datetime(df["dob"])
        ref = pd.to_datetime(df.get("trans_date_trans_time", pd.Timestamp("2020-06-30")))
        df["age"] = ((ref - dob).dt.days // 365).clip(lower=18, upper=95)
    else:
        df["age"] = np.random.randint(18, 90, size=len(df))

    cols = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long",
            "hour", "age", "category", "gender"]
    return df[cols].to_dict("records")


def _synth_event(artifact: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Fabrica un evento plausible usando las categorías válidas del artifact."""
    cats = artifact["categories"]
    return {
        "amt": round(rng.lognormvariate(3.5, 1.0), 2),
        "city_pop": rng.randint(100, 3_000_000),
        "lat": round(rng.uniform(25.0, 49.0), 4),
        "long": round(rng.uniform(-124.0, -67.0), 4),
        "merch_lat": round(rng.uniform(25.0, 49.0), 4),
        "merch_long": round(rng.uniform(-124.0, -67.0), 4),
        "hour": rng.randint(0, 23),
        "age": rng.randint(18, 90),
        "category": rng.choice(cats["category"]),
        "gender": rng.choice(cats["gender"]),
    }


def build_event_pool(
    artifact: dict[str, Any],
    n: int,
    seed: int = 42,
    data_path: str | Path = DEFAULT_DATA_PATH,
) -> list[dict[str, Any]]:
    """Devuelve `n` eventos base (reales si hay parquet, sintéticos si no)."""
    pool = _load_real_pool(data_path, n, seed)
    if pool is not None:
        logger.info("Pool de {} eventos muestreados de datos reales.", len(pool))
        return pool
    rng = random.Random(seed)
    pool = [_synth_event(artifact, rng) for _ in range(n)]
    logger.info("Pool de {} eventos sintéticos (no había parquet).", len(pool))
    return pool


def apply_drift(event: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    """Introduce un cambio de distribución en un evento.

    Sube fuerte la media de `amt` y sesga la mezcla de `category` hacia
    canales online de alto ticket. Es el drift que el consumer debe detectar.
    """
    drifted = dict(event)
    drifted["amt"] = round(drifted["amt"] * rng.uniform(2.5, 4.0) + 200.0, 2)
    drifted["category"] = rng.choice(
        ["shopping_net", "misc_net", "grocery_net", "travel"]
    )
    return drifted


# --- Cola local (modo sim) ------------------------------------------------
def ensure_stream_dir() -> None:
    """Crea el directorio de la cola local si no existe."""
    STREAM_DIR.mkdir(parents=True, exist_ok=True)


def reset_sim_queue() -> None:
    """Vacía la cola local antes de un nuevo flujo en modo sim."""
    ensure_stream_dir()
    SIM_QUEUE_PATH.write_text("", encoding="utf-8")


def append_sim_event(event: dict[str, Any]) -> None:
    """Agrega un evento (una línea JSON) a la cola local."""
    with SIM_QUEUE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_sim_events() -> list[dict[str, Any]]:
    """Lee todos los eventos escritos en la cola local."""
    if not SIM_QUEUE_PATH.exists():
        return []
    events = []
    for line in SIM_QUEUE_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def append_sim_alert(alert: dict[str, Any]) -> None:
    """Guarda una alerta en el archivo local de alertas (equivalente a fraud-alerts)."""
    ensure_stream_dir()
    with SIM_ALERTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(alert, ensure_ascii=False) + "\n")


# --- Drift: PSI -----------------------------------------------------------
def population_stability_index(
    baseline: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Calcula el PSI de una variable numérica contra un baseline.

    PSI < 0.1  -> sin cambios;  0.1–0.25 -> cambio moderado;  > 0.25 -> fuerte.
    Los cortes de los bins se fijan sobre los cuantiles del baseline.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(baseline) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 2:  # baseline casi constante -> no se puede binnear
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    base_counts, _ = np.histogram(baseline, bins=edges)
    curr_counts, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    base_pct = base_counts / base_counts.sum() + eps
    curr_pct = curr_counts / curr_counts.sum() + eps

    psi = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi)
