"""Tests del flujo champion/challenger: scoring en sombra, revisión, evaluación y deploy."""

from fastapi.testclient import TestClient

from fraud.api.main import app

client = TestClient(app)
HEADERS_OK = {"X-API-KEY": "token-secreto-123"}

VALID_TX = {
    "amt": 980.50,
    "category": "shopping_net",
    "gender": "M",
    "city_pop": 12000,
    "lat": 40.71,
    "long": -74.00,
    "merch_lat": 34.05,
    "merch_long": -118.24,
    "hour": 3,
    "age": 27,
}


def _predict() -> dict:
    resp = client.post("/v1/predict", json=VALID_TX, headers=HEADERS_OK)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_predict_records_tx_and_runs_challenger_in_shadow():
    body = _predict()
    # El champion decide (campos de siempre) ...
    assert isinstance(body["is_fraud"], bool)
    assert "model_version" in body
    # ... y ahora además hay trazabilidad y decisión.
    assert body["transaction_id"].startswith("tx-")
    assert body["decision"] in {"approve", "review"}
    assert body["status"] in {"OK", "PENDING"}
    # El challenger corrió en sombra (conftest activó uno).
    assert body["challenger"] is not None
    assert set(body["challenger"]) == {"is_fraud", "probability", "model_version"}


def test_review_flow_approve_sets_label():
    tx_id = _predict()["transaction_id"]

    # Con umbral 0.0, la tx quedó retenida y aparece en la cola de revisión.
    reviews = client.get("/v1/reviews?status=PENDING", headers=HEADERS_OK).json()
    assert any(item["transaction_id"] == tx_id for item in reviews["items"])

    # Detalle de la tx.
    detail = client.get(f"/v1/transactions/{tx_id}", headers=HEADERS_OK)
    assert detail.status_code == 200
    assert detail.json()["status"] == "PENDING"

    # El analista la aprueba (legítima -> label 0).
    dec = client.post(
        f"/v1/transactions/{tx_id}/decision",
        json={"decision": "approve"},
        headers=HEADERS_OK,
    )
    assert dec.status_code == 200

    resolved = client.get(f"/v1/transactions/{tx_id}", headers=HEADERS_OK).json()
    assert resolved["status"] == "APPROVED"
    assert resolved["label"]["label"] == 0
    assert resolved["label"]["source"] == "review"


def test_report_fraud_post_mortem():
    tx_id = _predict()["transaction_id"]
    resp = client.post(f"/v1/transactions/{tx_id}/report-fraud", headers=HEADERS_OK)
    assert resp.status_code == 201
    tx = client.get(f"/v1/transactions/{tx_id}", headers=HEADERS_OK).json()
    assert tx["label"]["label"] == 1
    assert tx["label"]["source"] == "post_mortem"


def test_report_fraud_unknown_tx_returns_404():
    resp = client.post("/v1/transactions/tx-noexiste/report-fraud", headers=HEADERS_OK)
    assert resp.status_code == 404


def test_get_unknown_transaction_returns_404():
    assert client.get("/v1/transactions/tx-noexiste", headers=HEADERS_OK).status_code == 404


def test_decision_requires_auth():
    assert client.post(
        "/v1/transactions/tx-x/decision", json={"decision": "approve"}
    ).status_code == 403


def test_evaluate_challenger_vs_champion():
    # Generamos ground truth: una tx fraude (reject) y otra legítima (approve).
    fraud_id = _predict()["transaction_id"]
    client.post(
        f"/v1/transactions/{fraud_id}/decision", json={"decision": "reject"}, headers=HEADERS_OK
    )
    legit_id = _predict()["transaction_id"]
    client.post(
        f"/v1/transactions/{legit_id}/decision", json={"decision": "approve"}, headers=HEADERS_OK
    )

    report = client.post("/v1/mlops/evaluate", headers=HEADERS_OK)
    assert report.status_code == 200
    data = report.json()
    assert data["n_with_ground_truth"] >= 2
    assert data["champion"] is not None
    assert data["challenger"] is not None
    for key in ("precision", "recall", "f1", "fraud_amount_captured"):
        assert key in data["champion"]
    assert "challenger_better" in data


def test_zdeploy_promote_local():
    """Último test: promueve el challenger local a champion (muta models/)."""
    resp = client.post("/v1/mlops/deploy?action=promote", headers=HEADERS_OK)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["mode"] == "local"
    assert body["status"] == "deployed"
