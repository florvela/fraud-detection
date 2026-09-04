"""Ingesta: baja datos de fraude de Hugging Face a data/raw/ (distribución real)"""

from __future__ import annotations

from itertools import islice

import pandas as pd
import typer
from datasets import load_dataset
from loguru import logger

from fraud.config import RAW_DATA_DIR

app = typer.Typer()

HF_DATASET = "pointe77/credit-card-transaction"
TARGET = "is_fraud"

RAW_COLUMNS = [
    "trans_date_trans_time",
    "category",
    "amt",
    "gender",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "dob",
    "is_fraud",
]


def load_raw(full: bool = False, n_rows: int = 200000, seed: int = 42) -> pd.DataFrame:
    """Devuelve un DataFrame con la distribución real (sin balancear).

    full=True baja el dataset completo (~1.85M filas, ~500MB); si no, toma una
    muestra aleatoria de n_rows filas por streaming (shuffle con buffer)
    """
    if full:
        logger.info(f"Descargando '{HF_DATASET}' COMPLETO...")
        ds = load_dataset(HF_DATASET, split="train")
        df = ds.to_pandas()[RAW_COLUMNS].copy()
    else:
        logger.info(f"Muestreando {n_rows:,} filas de '{HF_DATASET}' (streaming)...")
        stream = load_dataset(HF_DATASET, split="train", streaming=True)
        stream = stream.shuffle(seed=seed, buffer_size=10000)
        rows = [{col: r.get(col) for col in RAW_COLUMNS} for r in islice(stream, n_rows)]
        df = pd.DataFrame(rows, columns=RAW_COLUMNS)

    logger.info(f"Filas: {len(df):,} | tasa de fraude real: {df[TARGET].mean():.2%}")
    return df


@app.command()
def main(
    full: bool = typer.Option(False, help="Descargar el dataset completo (~500MB)"),
    n_rows: int = typer.Option(200000, help="Filas a muestrear si no se usa --full"),
    seed: int = typer.Option(42, help="Semilla del muestreo aleatorio"),
) -> None:
    """Descarga los datos y los guarda en data/raw/fraud_sample.parquet"""
    df = load_raw(full=full, n_rows=n_rows, seed=seed)

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DATA_DIR / "fraud_sample.parquet"
    df.to_parquet(output_path, index=False)
    logger.success(f"Guardado: {output_path}")


if __name__ == "__main__":
    app()
