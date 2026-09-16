"""Esquema GraphQL (Strawberry) que expone los metadatos del modelo"""

from __future__ import annotations

import pandas as pd
import strawberry

from fraud.api.lineage import get_lineage
from fraud.api.model_loader import ModelStore

MODEL_NAME = "fraud-detection"

_store = ModelStore()
_store.load()


@strawberry.type
class Metrics:
    roc_auc: float
    pr_auc: float


@strawberry.type
class LineageNode:
    name: str
    kind: str


@strawberry.type
class Model:
    name: str
    version: str
    metrics: Metrics

    @strawberry.field
    def lineage(self) -> list[LineageNode]:
        return [LineageNode(name=r["name"], kind=r["kind"]) for r in get_lineage(self.name)]


def _current_model() -> Model:
    m = _store.metrics
    return Model(
        name=MODEL_NAME,
        version=_store.version,
        metrics=Metrics(roc_auc=m["roc_auc"], pr_auc=m["pr_auc"]),
    )


# --- predicción vía GraphQL (para comparar con REST y gRPC) ---


@strawberry.input
class TransactionInput:
    """Entrada de una transacción (mismos campos que el modelo espera)."""

    amt: float
    category: str
    gender: str
    city_pop: int
    lat: float
    long: float
    merch_lat: float
    merch_long: float
    hour: int
    age: int


@strawberry.type
class PredictionType:
    is_fraud: bool
    probability: float
    model_version: str


def _predict(tx: TransactionInput) -> PredictionType:
    # DataFrame de una fila, respetando el orden de columnas del modelo.
    row = {name: getattr(tx, name) for name in _store.feature_order}
    X = pd.DataFrame([row], columns=_store.feature_order)
    probability = float(_store.pipeline.predict_proba(X)[0][1])
    is_fraud = bool(_store.pipeline.predict(X)[0])
    return PredictionType(
        is_fraud=is_fraud,
        probability=round(probability, 4),
        model_version=_store.version,
    )


@strawberry.type
class Query:
    @strawberry.field
    def model(self) -> Model:
        return _current_model()

    @strawberry.field
    def predict(self, transaction: TransactionInput) -> PredictionType:
        return _predict(transaction)


schema = strawberry.Schema(query=Query)
