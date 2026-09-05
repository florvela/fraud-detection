from __future__ import annotations

from datetime import datetime, timezone

import joblib
import pandas as pd
import typer
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from fraud.config import MODELS_DIR, PROCESSED_DATA_DIR
from fraud.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

app = typer.Typer()

MODEL_VERSION = "1.0.0"


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"{name}.parquet")
    return df[FEATURES], df[TARGET]


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """One-Hot para categóricas + XGBoost"""
    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)],
        remainder="passthrough",
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
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
def main() -> None:
    """Lee data/processed/, entrena y guarda models/model.joblib"""
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    # scale_pos_weight se calcula SOLO con train (nunca tocamos val/test)
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)
    logger.info(f"scale_pos_weight={scale_pos_weight:.1f} (neg={n_neg}, pos={n_pos})")

    pipeline = build_pipeline(scale_pos_weight)
    logger.info("Entrenando XGBoost...")
    pipeline.fit(X_train, y_train)

    evaluate(pipeline, X_val, y_val, "val")
    test_metrics = evaluate(pipeline, X_test, y_test, "test")

    print("\n=== Test (distribución real) ===")
    print(classification_report(y_test, pipeline.predict(X_test), target_names=["legit", "fraud"]))
    print("Matriz de confusión [ [TN FP] [FN TP] ]:")
    print(confusion_matrix(y_test, pipeline.predict(X_test)))

    categories = {c: sorted(X_train[c].dropna().unique().tolist()) for c in CATEGORICAL_FEATURES}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "version": MODEL_VERSION,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_order": FEATURES,
        "categories": categories,
        "target": TARGET,
        "metrics": test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, MODELS_DIR / "model.joblib")
    logger.success(f"Modelo guardado en models/model.joblib (v{MODEL_VERSION})")


if __name__ == "__main__":
    app()
