"""CLI para generar reportes de drift con Evidently (Mini-TP: Monitoreo).

Compara un dataset *reference* (baseline, típicamente `train.parquet` o la
primera mitad del histórico) contra un dataset *current* (típicamente
`test.parquet` o una partición a la que se le inyectó un shift) y produce:

- Un reporte HTML con **Data Drift** (siempre) y, si están disponibles las
  columnas de predicción y/o target, **Target/Prediction Drift**.
- Un resumen en logs (columnas con drift, share of drifted features).

Versión de Evidently asumida: **0.4.x** (API "clásica" con `Report` +
metric presets). Ver README para el detalle de compatibilidad.

Uso local (desde la raíz del repo):

    ./.venv/bin/python services/monitoring/evidently_report.py \\
        --reference data/processed/train.parquet \\
        --current   data/processed/test.parquet \\
        --output    services/monitoring/reports/drift_report.html

Para probar la detección de drift sin datos ya "driftados", se puede inyectar
un shift sintético sobre el current con `--inject-shift`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
from loguru import logger

# --- Configuración de features del modelo de fraude -----------------------
# Coinciden con el esquema del pipeline de serving (XGBoost).
NUMERICAL_FEATURES = [
    "amt",
    "city_pop",
    "lat",
    "long",
    "merch_lat",
    "merch_long",
    "hour",
    "age",
]
CATEGORICAL_FEATURES = ["category", "gender"]
TARGET = "is_fraud"
# Nombre convencional de la columna de predicción (score/label del modelo).
# Si el dataset la incluye, se activa el análisis de prediction drift.
PREDICTION = "prediction"

# monitoring/ vive en services/monitoring/ -> la raíz del repo son 2 niveles arriba.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent / "reports"

app = typer.Typer(add_completion=False, help="Reportes de drift con Evidently.")


@app.callback()
def _main() -> None:
    """Herramienta de monitoreo de drift para el modelo de fraude.

    El callback fuerza el modo multi-comando de Typer para que el subcomando
    `run` sea explícito (coherente con README y Dockerfile).
    """


def _load_dataset(path: Path) -> pd.DataFrame:
    """Carga un parquet/csv a DataFrame validando su existencia."""
    if not path.exists():
        raise typer.BadParameter(f"No existe el dataset: {path}")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise typer.BadParameter(f"Formato no soportado: {path.suffix}")
    logger.info("Cargado {} -> shape={}", path, df.shape)
    return df


def _inject_shift(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """Inyecta un shift sintético para forzar drift (solo para demo/test).

    Escala `amt` y desplaza `hour` para simular un cambio de comportamiento
    en las transacciones. No modifica el DataFrame original.
    """
    df = df.copy()
    if "amt" in df.columns:
        df["amt"] = df["amt"] * factor
    if "hour" in df.columns:
        df["hour"] = (df["hour"] + 6) % 24
    logger.warning("Shift sintético inyectado en el current (factor={}).", factor)
    return df


def _build_column_mapping(columns: list[str]):
    """Construye el ColumnMapping de Evidently según columnas presentes."""
    from evidently import ColumnMapping

    mapping = ColumnMapping()
    mapping.numerical_features = [c for c in NUMERICAL_FEATURES if c in columns]
    mapping.categorical_features = [c for c in CATEGORICAL_FEATURES if c in columns]
    mapping.target = TARGET if TARGET in columns else None
    mapping.prediction = PREDICTION if PREDICTION in columns else None
    return mapping


def _log_drift_summary(report) -> None:
    """Extrae y loguea el resumen de drift a partir de report.as_dict()."""
    result = report.as_dict()
    for metric in result.get("metrics", []):
        if metric.get("metric") != "DataDriftTable":
            continue
        res = metric.get("result", {})
        n_cols = res.get("number_of_columns")
        n_drifted = res.get("number_of_drifted_columns")
        share = res.get("share_of_drifted_columns")
        dataset_drift = res.get("dataset_drift")
        logger.info("=== Resumen de Data Drift ===")
        logger.info("Columnas analizadas: {}", n_cols)
        logger.info("Columnas con drift: {}", n_drifted)
        logger.info("Share of drifted features: {:.3f}", share or 0.0)
        logger.info("Dataset drift (flag global): {}", dataset_drift)

        drifted = [
            col
            for col, info in res.get("drift_by_columns", {}).items()
            if info.get("drift_detected")
        ]
        if drifted:
            logger.warning("Columnas con drift detectado: {}", ", ".join(drifted))
        else:
            logger.info("No se detectó drift en columnas individuales.")


@app.command()
def run(
    reference: Path = typer.Option(
        PROJECT_ROOT / "data" / "processed" / "train.parquet",
        "--reference",
        "-r",
        help="Dataset baseline (reference).",
    ),
    current: Path = typer.Option(
        PROJECT_ROOT / "data" / "processed" / "test.parquet",
        "--current",
        "-c",
        help="Dataset actual (current) a comparar contra el reference.",
    ),
    output: Path = typer.Option(
        DEFAULT_REPORTS_DIR / "drift_report.html",
        "--output",
        "-o",
        help="Ruta del HTML de salida.",
    ),
    inject_shift: bool = typer.Option(
        False,
        "--inject-shift",
        help="Inyecta un shift sintético en el current (para demostrar drift).",
    ),
    summary_json: Optional[Path] = typer.Option(
        None,
        "--summary-json",
        help="Si se indica, vuelca el resumen (as_dict) a un JSON.",
    ),
) -> None:
    """Genera el reporte de drift comparando reference vs current."""
    # Imports de Evidently dentro de la función: así el CLI se puede importar
    # / compilar aunque la librería no esté instalada en el entorno.
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

    ref_df = _load_dataset(reference)
    cur_df = _load_dataset(current)

    if inject_shift:
        cur_df = _inject_shift(cur_df)

    # Solo comparamos columnas comunes a ambos datasets.
    common_cols = [c for c in ref_df.columns if c in cur_df.columns]
    ref_df = ref_df[common_cols]
    cur_df = cur_df[common_cols]

    column_mapping = _build_column_mapping(common_cols)

    # Data Drift siempre; Target/Prediction Drift si hay target o predicción.
    metrics = [DataDriftPreset()]
    if column_mapping.target is not None or column_mapping.prediction is not None:
        logger.info(
            "Agregando TargetDriftPreset (target={}, prediction={}).",
            column_mapping.target,
            column_mapping.prediction,
        )
        metrics.append(TargetDriftPreset())
    else:
        logger.info("Sin target/prediction: solo se analiza Data Drift.")

    report = Report(metrics=metrics)
    report.run(
        reference_data=ref_df,
        current_data=cur_df,
        column_mapping=column_mapping,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(output))
    logger.success("Reporte HTML guardado en {}", output)

    _log_drift_summary(report)

    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(report.as_dict(), indent=2, default=str, ensure_ascii=False)
        )
        logger.info("Resumen JSON guardado en {}", summary_json)


if __name__ == "__main__":
    app()
