"""Servicio GraphQL (Strawberry): sirve el modelo de fraude en /graphql.

Cubre el TP2 (GraphQL + Neo4j) de forma autocontenida:
- `predict(transaction)`  -> scoring vía GraphQL (para comparar con REST y gRPC).
- `model { name version metrics lineage }` -> metadatos del modelo + **linaje**
  dato→feature→modelo consultado en Neo4j (ver `lineage.py` y `docs/03_neo4j.md`).
  GraphQL luce acá porque el cliente pide exactamente los campos que quiere en una
  sola query (sin over/under-fetching).
"""

import os

import joblib
import pandas as pd
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

from lineage import get_lineage, seed

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
MODEL_NAME = "fraud-detection"

# El modelo se carga UNA sola vez al iniciar.
ARTIFACT = joblib.load(MODEL_PATH)
PIPELINE = ARTIFACT["pipeline"]
VERSION = ARTIFACT["version"]
FEATURE_ORDER = ARTIFACT["feature_order"]
METRICS = ARTIFACT.get("metrics", {"roc_auc": 0.0, "pr_auc": 0.0})


@strawberry.input
class TransactionInput:
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
class Prediction:
    is_fraud: bool
    probability: float
    model_version: str


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


@strawberry.type
class Query:
    @strawberry.field
    def predict(self, transaction: TransactionInput) -> Prediction:
        row = {name: getattr(transaction, name) for name in FEATURE_ORDER}
        X = pd.DataFrame([row], columns=FEATURE_ORDER)
        probability = float(PIPELINE.predict_proba(X)[0][1])
        is_fraud = bool(PIPELINE.predict(X)[0])
        return Prediction(
            is_fraud=is_fraud,
            probability=round(probability, 4),
            model_version=VERSION,
        )

    @strawberry.field
    def model(self) -> Model:
        return Model(
            name=MODEL_NAME,
            version=VERSION,
            metrics=Metrics(roc_auc=METRICS["roc_auc"], pr_auc=METRICS["pr_auc"]),
        )


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
def read_root():
    return {"message": "GraphQL ML Service is running. Access /graphql"}


@app.on_event("startup")
def _seed_lineage_on_startup() -> None:
    """Siembra el grafo de linaje en Neo4j si está configurado (best-effort)."""
    if os.getenv("NEO4J_URI"):
        try:
            seed(model_name=MODEL_NAME)
        except Exception:  # noqa: BLE001 - Neo4j puede no estar listo; linaje degrada a []
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
