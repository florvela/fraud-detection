"""Servicio REST (FastAPI): sirve el modelo de fraude en POST /predict."""

import os

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")

# El modelo se carga UNA sola vez al iniciar.
ARTIFACT = joblib.load(MODEL_PATH)
PIPELINE = ARTIFACT["pipeline"]
VERSION = ARTIFACT["version"]
FEATURE_ORDER = ARTIFACT["feature_order"]


class Transaction(BaseModel):
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


app = FastAPI()


@app.post("/predict")
def predict(tx: Transaction):
    row = {name: getattr(tx, name) for name in FEATURE_ORDER}
    X = pd.DataFrame([row], columns=FEATURE_ORDER)
    probability = float(PIPELINE.predict_proba(X)[0][1])
    is_fraud = bool(PIPELINE.predict(X)[0])
    return {
        "is_fraud": is_fraud,
        "probability": round(probability, 4),
        "model_version": VERSION,
    }


@app.get("/")
def read_root():
    return {"message": "REST ML Service is running. Post to /predict"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
