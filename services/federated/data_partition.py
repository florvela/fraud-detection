"""Particionador de datos en silos no-IID para Banco A y Banco B (Typer CLI).

Objetivo
--------
Simular dos bancos que NO comparten sus transacciones. Cada uno recibe un
subconjunto del train con una distribución DISTINTA (no-IID), como ocurriría en
la realidad si operan en regiones geográficas diferentes.

Criterio de partición (no-IID)
------------------------------
El dataset crudo (``data/raw/fraud_sample.parquet``) NO tiene columna de estado
o región, pero sí coordenadas geográficas. Usamos la **longitud** como proxy
geográfico y partimos por su mediana:

* **Banco A = región Oeste**  -> ``long <= mediana``
* **Banco B = región Este**   -> ``long >  mediana``

Esto genera silos con distribuciones espaciales claramente distintas
(``lat``, ``long``, ``merch_lat``, ``merch_long`` cambian de forma marcada entre
silos), que es la esencia de un escenario no-IID: cada banco "ve" otra parte del
mapa. El desbalance (~1% de fraude) se preserva en ambos silos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from model import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

app = typer.Typer(add_completion=False, help="Particiona train en silos no-IID (Banco A / Banco B).")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN = _REPO_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_OUT_DIR = _SERVICE_DIR / "data"

_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def _resumen_silo(nombre: str, df: pd.DataFrame) -> None:
    """Loguea un resumen del silo para documentar el carácter no-IID."""
    logger.info(
        "{}: n={} | fraude={:.3%} | long[min={:.2f}, max={:.2f}, media={:.2f}]",
        nombre, len(df), df[TARGET].mean(), df["long"].min(), df["long"].max(), df["long"].mean(),
    )


@app.command()
def split(
    train_path: Path = typer.Option(DEFAULT_TRAIN, help="Parquet de train a particionar."),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, help="Directorio de salida de los silos."),
    criterio: str = typer.Option(
        "long", help="Proxy geográfico para el split no-IID: 'long' (este/oeste) o 'lat' (norte/sur).",
    ),
) -> None:
    """Genera ``banco_a.parquet`` (Oeste/Sur) y ``banco_b.parquet`` (Este/Norte)."""
    if criterio not in {"long", "lat"}:
        raise typer.BadParameter("criterio debe ser 'long' o 'lat'.")

    logger.info("Leyendo train desde {}", train_path)
    df = pd.read_parquet(train_path, columns=_COLS)

    umbral = float(df[criterio].median())
    logger.info("Partición no-IID por '{}' con umbral (mediana) = {:.4f}", criterio, umbral)

    banco_a = df[df[criterio] <= umbral].reset_index(drop=True)  # Oeste (long) / Sur (lat)
    banco_b = df[df[criterio] > umbral].reset_index(drop=True)   # Este (long) / Norte (lat)

    out_dir.mkdir(parents=True, exist_ok=True)
    path_a = out_dir / "banco_a.parquet"
    path_b = out_dir / "banco_b.parquet"
    banco_a.to_parquet(path_a, index=False)
    banco_b.to_parquet(path_b, index=False)

    _resumen_silo("Banco A", banco_a)
    _resumen_silo("Banco B", banco_b)
    logger.success("Silos escritos en {} y {}", path_a, path_b)


if __name__ == "__main__":
    app()
