"""Esquema GraphQL (Strawberry) que expone los metadatos del modelo"""

from __future__ import annotations

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


@strawberry.type
class Query:
    @strawberry.field
    def model(self) -> Model:
        return _current_model()


schema = strawberry.Schema(query=Query)
