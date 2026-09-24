"""Compara el MLP federado global contra el champion XGBoost centralizado (Typer CLI).

Evalúa ambos modelos sobre ``data/processed/test.parquet`` y responde la
pregunta clave del TP: **¿cuánto del PR-AUC del modelo centralizado (que ve
todos los datos) recupera el modelo federado (que nunca los comparte)?**

Métricas: PR-AUC (average precision), ROC-AUC y recall de la clase fraude a un
umbral fijo (0.5 para el MLP; para XGBoost se usa su probabilidad de clase 1).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import typer
from loguru import logger

from model import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    build_model,
    load_preprocessor,
    set_weights,
)

app = typer.Typer(add_completion=False, help="Federado vs centralizado (champion XGBoost).")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_TEST = _REPO_ROOT / "data" / "processed" / "test.parquet"
DEFAULT_CHAMPION = _REPO_ROOT / "models" / "model.joblib"
DEFAULT_FED_WEIGHTS = _SERVICE_DIR / "models" / "mlp_federado.npz"
_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _cargar_pesos_npz(path: Path) -> list[np.ndarray]:
    """Lee el .npz con los pesos del modelo global en orden."""
    data = np.load(path)
    return [data[k] for k in data.files]


def _metricas(y_true: np.ndarray, probs: np.ndarray, umbral: float = 0.5) -> dict:
    """Calcula PR-AUC, ROC-AUC y recall de fraude a un umbral."""
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score

    pred = (probs >= umbral).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, probs)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "recall_fraude": float(recall_score(y_true, pred, zero_division=0)),
    }


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


def _probs_champion(df: pd.DataFrame, champion_path: Path) -> np.ndarray:
    """Probabilidades de fraude del champion XGBoost.

    ``model.joblib`` es un dict con el ``pipeline`` sklearn (imputación +
    one-hot + XGBoost) y metadata (``feature_order``). Soporta tanto ese
    formato como un estimador/pipeline plano.
    """
    modelo = joblib.load(champion_path)
    if isinstance(modelo, dict):
        pipeline = modelo["pipeline"]
        orden = modelo.get("feature_order", _FEATURES)
    else:
        pipeline, orden = modelo, _FEATURES
    X = df[orden]
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    # Fallback por si es un Booster sin predict_proba.
    return np.asarray(pipeline.predict(X)).ravel()


@app.command()
def compare(
    test_path: Path = typer.Option(DEFAULT_TEST, help="Parquet de test."),
    fed_weights: Path = typer.Option(DEFAULT_FED_WEIGHTS, help="Pesos del MLP federado (.npz)."),
    champion_path: Path = typer.Option(DEFAULT_CHAMPION, help="Champion XGBoost (.joblib)."),
    umbral: float = typer.Option(0.5, help="Umbral de decisión para el recall."),
) -> None:
    """Evalúa ambos modelos y muestra la tabla comparativa."""
    logger.info("Leyendo test desde {}", test_path)
    df = pd.read_parquet(test_path)
    y = df[TARGET].to_numpy(dtype=int)
    logger.info("Test n={} | fraude={:.3%}", len(y), y.mean())

    resultados: dict[str, dict] = {}

    logger.info("Evaluando champion XGBoost (centralizado)...")
    resultados["XGBoost (centralizado)"] = _metricas(y, _probs_champion(df, champion_path), umbral)

    if fed_weights.exists():
        logger.info("Evaluando MLP federado (FedAvg, sin compartir datos)...")
        resultados["MLP federado (FedAvg)"] = _metricas(y, _probs_federado(df, fed_weights), umbral)
    else:
        logger.warning("No existe {}; corré primero server.py + clientes.", fed_weights)

    # --- Tabla ---
    print("\n" + "=" * 78)
    print(f"{'Modelo':<28}{'PR-AUC':>12}{'ROC-AUC':>12}{'Recall fraude':>16}")
    print("-" * 78)
    for nombre, m in resultados.items():
        print(f"{nombre:<28}{m['pr_auc']:>12.4f}{m['roc_auc']:>12.4f}{m['recall_fraude']:>16.4f}")
    print("=" * 78)

    # --- Cuánto del PR-AUC centralizado recupera el federado ---
    if "MLP federado (FedAvg)" in resultados:
        pr_central = resultados["XGBoost (centralizado)"]["pr_auc"]
        pr_fed = resultados["MLP federado (FedAvg)"]["pr_auc"]
        recuperado = (pr_fed / pr_central * 100.0) if pr_central > 0 else 0.0
        print(
            f"\nEl MLP federado recupera el {recuperado:.1f}% del PR-AUC del modelo "
            f"centralizado\nSIN que los bancos compartan sus transacciones "
            f"(PR-AUC {pr_fed:.4f} vs {pr_central:.4f})."
        )
    print()


if __name__ == "__main__":
    app()
