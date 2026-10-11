from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import typer
from xgboost import XGBClassifier

from fraud.config import MODELS_DIR, PROCESSED_DATA_DIR
from fraud.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

app = typer.Typer()

# Versión por defecto del modelo "bueno" (balanceado, muchos árboles). El CLI permite
# sobreescribirla para versionar otras configuraciones (p.ej. el champion malo de la demo).
DEFAULT_VERSION = "1.0.0"


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"{name}.parquet")
    return df[FEATURES], df[TARGET]


def build_pipeline(
    scale_pos_weight: float,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.1,
) -> Pipeline:
    """One-Hot para categóricas + XGBoost.

    Los hiperparámetros se exponen para poder entrenar tanto el modelo productivo
    (muchos árboles, balanceado) como variantes deliberadamente débiles (pocas
    iteraciones) desde el MISMO código — sin artefactos "mágicos" fuera del pipeline.
    """
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("prep", preprocessor), ("clf", model)])


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
    }
    logger.info(f"[{name}] ROC-AUC={metrics['roc_auc']:.4f} | PR-AUC={metrics['pr_auc']:.4f}")
    return metrics


@app.command()
def main(
    n_estimators: int = typer.Option(
        300, envvar="TRAIN_N_ESTIMATORS", help="Nº de árboles (pocos = modelo débil)."
    ),
    max_depth: int = typer.Option(6, envvar="TRAIN_MAX_DEPTH", help="Profundidad máxima."),
    learning_rate: float = typer.Option(0.1, envvar="TRAIN_LEARNING_RATE", help="Learning rate."),
    balance: bool = typer.Option(
        True,
        "--balance/--no-balance",
        envvar="TRAIN_BALANCE",
        help="Manejar el desbalance con scale_pos_weight (--no-balance lo desactiva).",
    ),
    version: str = typer.Option(
        None, envvar="MODEL_VERSION", help="Versión del artefacto (default: derivada de los params)."
    ),
    output: Path = typer.Option(
        None, envvar="MODEL_OUTPUT", help="Ruta de salida (default: models/model.joblib)."
    ),
) -> None:
    """Lee data/processed/, entrena y guarda el artefacto del modelo.

    Con los defaults produce el modelo productivo (`models/model.joblib`, v1.0.0,
    balanceado, 300 árboles). Variando los hiperparámetros se genera cualquier otra
    configuración (p.ej. el champion malo: `--no-balance --n-estimators 5`).
    """
    out_path = output or (MODELS_DIR / "model.joblib")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # scale_pos_weight se calcula SOLO con train (nunca tocamos val/test)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = (n_neg / max(n_pos, 1)) if balance else 1.0
    logger.info(
        f"n_estimators={n_estimators} max_depth={max_depth} balance={balance} "
        f"scale_pos_weight={scale_pos_weight:.1f} (neg={n_neg}, pos={n_pos})"
    )

    pipeline = build_pipeline(scale_pos_weight, n_estimators, max_depth, learning_rate)
    logger.info("Entrenando XGBoost...")
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_val, y_val, "val")
    test_metrics = evaluate(pipeline, X_test, y_test, "test")

    print("\n=== Test (distribución real) ===")
    print(classification_report(y_test, pipeline.predict(X_test), target_names=["legit", "fraud"]))
    print("Matriz de confusión [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, pipeline.predict(X_test)))

    # Versión derivada de la config si no se pasó una explícita: así el artefacto
    # "se autodocumenta" (un champion malo queda etiquetado como tal).
    if version is None:
        version = DEFAULT_VERSION if (balance and n_estimators >= 100) else (
            f"0.1.0-{n_estimators}est-{'bal' if balance else 'nobal'}"
        )

    categories = {c: sorted(X_train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "version": version,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": FEATURES,
        "categories": categories,
        "target": TARGET,
        "metrics": test_metrics,
        "hyperparams": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "balance": balance,
            "scale_pos_weight": round(scale_pos_weight, 2),
        },
        "trained_at": datetime.now(UTC).isoformat(),
    }
    joblib.dump(artifact, out_path)
    logger.success(f"Modelo guardado en {out_path} (v{version})")


if __name__ == "__main__":
    app()
