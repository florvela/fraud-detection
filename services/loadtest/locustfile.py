"""Prueba de carga del borde REST con Locust (simulación de muchos clientes).

Cada "usuario" virtual envía transacciones **reales** del split de test
(`data/processed/test.parquet`) al endpoint `/v1/predict`, igual que lo haría el
sistema de pagos. Locust mide throughput (RPS), latencias (p50/p95/p99) y tasa de
error en vivo en su web UI (http://localhost:8089), y el tráfico queda registrado en
`fraud.db` (datos reales que luego se consultan desde la propia base).

Config por entorno (sin hardcodear):
    LOCUST_HOST     URL del borde REST (p.ej. http://rest:8080). La toma Locust.
    API_KEY         token del header X-API-KEY (default demo: token-secreto-123).
    LOADTEST_DATA   parquet con transacciones a muestrear.
    LOADTEST_SAMPLE cuántas filas cachear en memoria (default 5000).
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

# Contrato de features (igual que el pipeline de datos del repo).
NUMERIC_FEATURES = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "hour", "age"]
CATEGORICAL_FEATURES = ["category", "gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
_INT_COLS = {"city_pop", "hour", "age"}

API_KEY = os.getenv("API_KEY", "token-secreto-123")
DATA_PATH = os.getenv("LOADTEST_DATA", "/app/data/processed/test.parquet")
SAMPLE_SIZE = int(os.getenv("LOADTEST_SAMPLE", "5000"))

# Transacción de respaldo si no hay dataset montado (un fraude claro).
_FALLBACK = [{
    "amt": 925.94, "category": "shopping_net", "gender": "M", "city_pop": 4653,
    "lat": 40.5046, "long": -77.7186, "merch_lat": 41.4437, "merch_long": -78.3918,
    "hour": 12, "age": 22,
}]

# Cache de transacciones (se llena una vez al iniciar).
_TX: list[dict] = []


def _coerce(row: dict) -> dict:
    """Normaliza tipos numpy -> nativos para serializar a JSON."""
    out: dict = {}
    for col in FEATURES:
        val = row[col]
        if col in _INT_COLS:
            out[col] = int(val)
        elif col in CATEGORICAL_FEATURES:
            out[col] = str(val)
        else:
            out[col] = float(val)
    return out


@events.init.add_listener
def _load_transactions(environment, **_kwargs) -> None:
    """Carga y cachea transacciones reales del dataset una sola vez."""
    global _TX
    try:
        import pandas as pd

        df = pd.read_parquet(DATA_PATH, columns=FEATURES)
        take = min(SAMPLE_SIZE, len(df))
        _TX = [_coerce(r) for r in df.sample(n=take, random_state=1).to_dict("records")]
        print(f"[loadtest] cacheadas {len(_TX)} transacciones reales de {DATA_PATH}")
    except Exception as exc:  # noqa: BLE001 - sin dataset, usamos el fallback
        _TX = [_coerce(r) for r in _FALLBACK]
        print(f"[loadtest] no se pudo leer {DATA_PATH} ({exc}); usando fallback")


class FraudUser(HttpUser):
    """Cliente virtual que puntúa transacciones contra el borde REST."""

    # Espera 0.1-0.5s entre requests: simula clientes realistas, no un martillo.
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict(self) -> None:
        row = random.choice(_TX)
        self.client.post(
            "/v1/predict",
            json=row,
            headers={"X-API-KEY": API_KEY},
            name="/v1/predict",
        )

    @task(1)
    def model_info(self) -> None:
        self.client.get(
            "/v1/model-info",
            headers={"X-API-KEY": API_KEY},
            name="/v1/model-info",
        )
