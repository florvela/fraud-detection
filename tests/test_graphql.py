"""Tests de la API GraphQL con el TestClient de FastAPI"""

from fastapi.testclient import TestClient

from fraud.api.main import app

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
