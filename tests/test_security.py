"""Tests de los controles de seguridad SAIF del serving REST.

Cubre las capas de la clase 7: autenticación por API key, autorización por rol
(RBAC), hash de integridad del modelo y model card de gobernanza.
"""

from fastapi.testclient import TestClient

from fraud.api.main import app

client = TestClient(app)

ADMIN = {"X-API-KEY": "token-secreto-123"}      # rol admin (default, hace todo)
ANALYST = {"X-API-KEY": "analyst-key"}          # rol analyst (revisión)
MLOPS = {"X-API-KEY": "mlops-key"}              # rol mlops (modelos)

VALID_TX = {
    "amt": 120.5, "category": "grocery_pos", "gender": "F", "city_pop": 50000,
    "lat": 40.1, "long": -74.5, "merch_lat": 40.3, "merch_long": -74.2,
    "hour": 2, "age": 35,
}


# --------------------------------------------------------------------------- #
# Autenticación (Capa 1)
# --------------------------------------------------------------------------- #
def test_predict_sin_token_da_403():
    assert client.post("/v1/predict", json=VALID_TX).status_code == 403


def test_predict_token_invalido_da_403():
    resp = client.post("/v1/predict", json=VALID_TX, headers={"X-API-KEY": "no-existe"})
    assert resp.status_code == 403


# --------------------------------------------------------------------------- #
# Autorización por rol / RBAC (Capa 1)
# --------------------------------------------------------------------------- #
def test_analista_puede_predecir_y_ver_reviews():
    assert client.post("/v1/predict", json=VALID_TX, headers=ANALYST).status_code == 200
    assert client.get("/v1/reviews", headers=ANALYST).status_code == 200


def test_analista_NO_puede_mlops():
    # Un analista no puede entrenar/evaluar/deployar: 403.
    assert client.post("/v1/mlops/evaluate", headers=ANALYST).status_code == 403
    assert client.post("/v1/mlops/reload", headers=ANALYST).status_code == 403


def test_mlops_puede_mlops():
    assert client.post("/v1/mlops/reload", headers=MLOPS).status_code == 200


def test_mlops_tambien_puede_predecir():
    # mlops es un rol de pipeline; también puede usar el scoring.
    assert client.post("/v1/predict", json=VALID_TX, headers=MLOPS).status_code == 200


def test_admin_puede_todo():
    assert client.post("/v1/predict", json=VALID_TX, headers=ADMIN).status_code == 200
    assert client.post("/v1/mlops/reload", headers=ADMIN).status_code == 200
    assert client.get("/v1/reviews", headers=ADMIN).status_code == 200


# --------------------------------------------------------------------------- #
# Integridad (Capa 7) y gobernanza (Capa 8)
# --------------------------------------------------------------------------- #
def test_health_expone_hash_de_integridad():
    body = client.get("/health").json()
    assert "model_sha256" in body
    # Con el modelo cargado, el hash es un sha256 hex (64 chars).
    if body["status"] == "ok":
        assert body["model_sha256"] is None or len(body["model_sha256"]) == 64


def test_model_card_tiene_version_y_controles():
    resp = client.get("/v1/model-card", headers=ADMIN)
    assert resp.status_code == 200
    card = resp.json()
    assert card["model_name"] == "fraud-detection"
    assert "version" in card
    assert "sha256" in card
    assert isinstance(card["security_controls"], list) and card["security_controls"]


def test_model_card_requiere_auth():
    assert client.get("/v1/model-card").status_code == 403


def test_audit_endpoint_devuelve_lista():
    resp = client.get("/v1/audit", headers=ADMIN)
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body and isinstance(body["entries"], list)


def test_audit_requiere_auth():
    assert client.get("/v1/audit").status_code == 403
