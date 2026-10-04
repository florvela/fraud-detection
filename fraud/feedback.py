"""Feedback: fusiona el train base con las transacciones inyectadas (ground truth).

Paso del DAG que corre ENTRE `fraud.features` y `fraud.modeling.train`. El train base
sale del dataset original (HF -> raw -> processed/train.parquet); las transacciones
inyectadas (seed del analista + revisiones) viven en `fraud.db`. Este módulo lee
ambas fuentes y reescribe `train.parquet` = base + inyectados, de modo que el
reentrenamiento use el dataset ORIGINAL más la ground truth acumulada.

Idempotente por corrida: el paso `features` regenera el train base desde cero antes
de este paso, así que los inyectados se agregan una sola vez por corrida del DAG.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json

import pandas as pd
import typer
from loguru import logger

from fraud.config import PROCESSED_DATA_DIR
from fraud.features import FEATURES, TARGET

app = typer.Typer()


def _load_injected() -> list[dict]:
    """Lee las filas inyectadas desde fraud.db (features crudas + is_fraud).

    Usa el store (fraud/api/__init__.py es liviano). Si la base no existe o está
    vacía, devuelve [] en vez de romper.
    """
    try:
        from fraud.api.store import TransactionStore

        return TransactionStore().labeled_training_rows()
    except Exception as exc:  # noqa: BLE001 - base inexistente/vacía -> 0 inyectados
        logger.warning(f"No se pudieron leer inyectados de fraud.db ({exc}); asumo 0.")
        return []


@app.command()
def main() -> None:
    """Fusiona train base + inyectados y reescribe data/processed/train.parquet."""
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    if not train_path.exists():
        logger.error(
            f"No existe {train_path}; corré `python -m fraud.features` primero."
        )
        raise typer.Exit(code=1)

    base = pd.read_parquet(train_path)
    n_base = len(base)

    injected_rows = _load_injected()
    n_injected = len(injected_rows)

    if n_injected:
        injected = pd.DataFrame(injected_rows)
        # Aseguramos columnas FEATURES + TARGET y dtypes que matcheen el train base.
        for col in FEATURES + [TARGET]:
            if col not in injected.columns:
                injected[col] = None
        injected = injected[FEATURES + [TARGET]]
        for col in FEATURES + [TARGET]:
            try:
                injected[col] = injected[col].astype(base[col].dtype)
            except (ValueError, TypeError) as exc:
                logger.warning(f"No pude coercionar '{col}' al dtype del base: {exc}")
        aug = pd.concat([base, injected], ignore_index=True)
    else:
        aug = base

    n_total = len(aug)
    aug.to_parquet(train_path, index=False)

    meta = {
        "n_base": n_base,
        "n_injected": n_injected,
        "n_total": n_total,
        "injected_at": datetime.now(UTC).isoformat(),
    }
    (PROCESSED_DATA_DIR / "train_meta.json").write_text(json.dumps(meta, indent=2))

    logger.success(f"merge_feedback: base={n_base} injected={n_injected} total={n_total}")


if __name__ == "__main__":
    app()
