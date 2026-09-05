"""Tests de la API con el TestClient de FastAPI (app en memoria)"""

from fastapi.testclient import TestClient

from fraud.api.main import app

client = TestClient(app)

HEADERS_OK = {"X-API-KEY": "token-secreto-123"}

VALID_TX = {
    "amt": 120.5,
    "category": "grocery_pos",
    "gender": "F",
    "city_pop": 50000,
    "lat": 40.1,
    "long": -74.5,
    "merch_lat": 40.3,
    "merch_long": -74.2,
    "hour": 2,
    "age": 35,
}


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_predict_valid_case():
    resp = client.post("/v1/predict", json=VALID_TX, headers=HEADERS_OK)
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["is_fraud"], bool)
    assert 0.0 <= body["probability"] <= 1.0
    assert "model_version" in body


def test_predict_invalid_type_returns_422():
    tx = {**VALID_TX, "amt": "no_es_un_numero"}
    assert client.post("/v1/predict", json=tx, headers=HEADERS_OK).status_code == 422


def test_predict_invalid_category_returns_422():
    tx = {**VALID_TX, "category": "cripto"}
    assert client.post("/v1/predict", json=tx, headers=HEADERS_OK).status_code == 422


def test_predict_hour_out_of_range_returns_422():
    tx = {**VALID_TX, "hour": 99}
    assert client.post("/v1/predict", json=tx, headers=HEADERS_OK).status_code == 422


def test_predict_missing_field_returns_422():
    tx = {k: v for k, v in VALID_TX.items() if k != "age"}
    assert client.post("/v1/predict", json=tx, headers=HEADERS_OK).status_code == 422


def test_predict_non_positive_amount_returns_422():
    tx = {**VALID_TX, "amt": -5.0}
    assert client.post("/v1/predict", json=tx, headers=HEADERS_OK).status_code == 422


def test_predict_extra_field_returns_422():
    tx = {**VALID_TX, "columna_extra": 1}
    assert client.post("/v1/predict", json=tx, headers=HEADERS_OK).status_code == 422


def test_predict_invalid_token_returns_403():
    resp = client.post("/v1/predict", json=VALID_TX, headers={"X-API-KEY": "token-equivocado"})
    assert resp.status_code == 403


def test_predict_no_token_returns_403():
    assert client.post("/v1/predict", json=VALID_TX).status_code == 403


def test_health_is_public_without_token():
    assert client.get("/health").status_code == 200
