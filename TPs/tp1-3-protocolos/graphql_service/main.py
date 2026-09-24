"""Servicio GraphQL (Strawberry): sirve el modelo de fraude en /graphql."""

import os

import joblib
import pandas as pd
import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# El modelo se carga UNA sola vez al iniciar.
ARTIFACT = joblib.load(MODEL_PATH)
PIPELINE = ARTIFACT["pipeline"]
VERSION = ARTIFACT["version"]
FEATURE_ORDER = ARTIFACT["feature_order"]


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


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema)

app = FastAPI()
app.include_router(graphql_app, prefix="/graphql")


@app.get("/")
def read_root():
    return {"message": "GraphQL ML Service is running. Access /graphql"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
