"""Tests del servicio GraphQL del TP2 (metadatos + linaje + predict).

Standalone: se corre desde este directorio o con la carpeta en el PYTHONPATH.
    cd TPs/tp1-3-protocolos/graphql_service && python -m pytest test_graphql_service.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient  # noqa: E402

from main import app  # noqa: E402

client = TestClient(app)


def _gql(query: str):
    return client.post("/graphql", json={"query": query})


def test_model_query_returns_metadata():
    resp = _gql("{ model { name version metrics { rocAuc prAuc } } }")
    assert resp.status_code == 200
    model = resp.json()["data"]["model"]
    assert model["name"]
    assert model["version"]
    assert 0.0 <= model["metrics"]["rocAuc"] <= 1.0
    assert 0.0 <= model["metrics"]["prAuc"] <= 1.0


def test_minimal_query_omits_metrics():
    resp = _gql("{ model { name version } }")
    model = resp.json()["data"]["model"]
    assert "metrics" not in model


def test_lineage_field_is_list():
    resp = _gql("{ model { lineage { name kind } } }")
    assert resp.status_code == 200
    lineage = resp.json()["data"]["model"]["lineage"]
    assert isinstance(lineage, list)


def test_predict_via_graphql():
    query = """
    {
      predict(transaction: {
        amt: 120.5, category: "grocery_pos", gender: "F", cityPop: 50000,
        lat: 40.1, long: -74.5, merchLat: 40.3, merchLong: -74.2, hour: 2, age: 35
      }) { isFraud probability modelVersion }
    }
    """
    resp = _gql(query)
    assert resp.status_code == 200
    pred = resp.json()["data"]["predict"]
    assert isinstance(pred["isFraud"], bool)
    assert 0.0 <= pred["probability"] <= 1.0
