"""Métricas PROPIAS del MLP federado global sobre el test (Typer CLI).

Este es el paso de evaluación que vive DENTRO del DAG de entrenamiento federado:
calcula las métricas del modelo global (``models/mlp_federado.npz``) sobre
``data/processed/test.parquet`` y las vuelca a un JSON que consume el registro en
MLflow. **No compara contra ningún otro modelo**: un entrenamiento se autoevalúa con
sus propias métricas y no debe depender de que exista un champion centralizado (en
una producción *federated-first* podría no existir).

La comparación federado-vs-centralizado (el benchmark del TP) es una
responsabilidad aparte y vive en ``evaluate_vs_central.py``, que se corre a demanda.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
    load_preprocessor,
    set_weights,
)

app = typer.Typer(add_completion=False, help="Métricas propias del MLP federado (sin comparación).")

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # /app en Docker
_SERVICE_DIR = _HERE.parent
DEFAULT_TEST = _REPO_ROOT / "data" / "processed" / "test.parquet"
DEFAULT_FED_WEIGHTS = _SERVICE_DIR / "models" / "mlp_federado.npz"
_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _cargar_pesos_npz(path: Path) -> list[np.ndarray]:
    data = np.load(path)
    return [data[k] for k in data.files]


def _probs_federado(df: pd.DataFrame, fed_weights: Path) -> np.ndarray:
    """Probabilidades de fraude del MLP federado global."""
    import torch

    pre = load_preprocessor()
    X = pre.transform(df[_FEATURES])
    model = build_model(pre.n_features)
    set_weights(model, _cargar_pesos_npz(fed_weights))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
        return torch.sigmoid(logits).numpy().ravel()


def _metricas(y_true: np.ndarray, probs: np.ndarray, umbral: float) -> dict:
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

    pred = (probs >= umbral).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "recall_fraude": float(recall_score(y_true, pred, zero_division=0)),
    }


@app.command()
def evaluate(
    test_path: Path = typer.Option(DEFAULT_TEST, help="Parquet de test."),
    fed_weights: Path = typer.Option(DEFAULT_FED_WEIGHTS, help="Pesos del MLP federado (.npz)."),
    umbral: float = typer.Option(0.5, help="Umbral de decisión para el recall."),
    metrics_out: Path = typer.Option(
        None, help="Ruta del JSON de métricas (lo consume el registro en MLflow del DAG)."
    ),
) -> None:
    """Evalúa el federado sobre el test y, si se indica, vuelca sus métricas a JSON."""
    if not fed_weights.exists():
        raise typer.BadParameter(f"No existe {fed_weights}; corré primero el entrenamiento federado.")

    logger.info("Leyendo test desde {}", test_path)
    df = pd.read_parquet(test_path)
    y = df[TARGET].to_numpy(dtype=int)
    logger.info("Test n={} | fraude={:.3%}", len(y), y.mean())

    metrics = _metricas(y, _probs_federado(df, fed_weights), umbral)

    print("\n" + "=" * 60)
    print(f"{'MLP federado (FedAvg)':<28}{'PR-AUC':>10}{'ROC-AUC':>10}{'Recall':>10}")
    print("-" * 60)
    print(f"{'':<28}{metrics['pr_auc']:>10.4f}{metrics['roc_auc']:>10.4f}{metrics['recall_fraude']:>10.4f}")
    print("=" * 60 + "\n")

    if metrics_out is not None:
        payload = {**metrics, "n_test": len(y)}
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(payload, indent=2))
        logger.success("Métricas del federado volcadas a {}", metrics_out)


if __name__ == "__main__":
    app()
