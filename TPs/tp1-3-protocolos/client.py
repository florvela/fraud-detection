"""Cliente de prueba de la API: dispara un caso válido y uno inválido"""

import json

import requests

BASE_URL = "http://localhost:8080"
HEADERS = {"X-API-KEY": "token-secreto-123"}
HEADERS_BAD = {"X-API-KEY": "token-equivocado"}


def separator(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def check_valid_case() -> None:
    separator("1) caso valido  (esperamos 200)")
    transaction = {
        "amt": 950.75,
        "category": "shopping_net",
        "gender": "F",
        "city_pop": 15000,
        "lat": 40.1,
        "long": -74.5,
        "merch_lat": 41.9,
        "merch_long": -80.2,
        "hour": 2,
        "age": 35,
    }
    print("Enviando:", transaction)
    resp = requests.post(f"{BASE_URL}/v1/predict", json=transaction, headers=HEADERS)
    print("Status code:", resp.status_code)
    print("Respuesta  :", resp.json())


def check_invalid_case() -> None:
    separator("2) caso invalido  (esperamos 422)")
    invalid = {
        "amt": "no_es_un_numero",
        "category": "cripto",
        "gender": "F",
        "city_pop": 15000,
        "lat": 40.1,
        "long": -74.5,
        "merch_lat": 41.9,
        "merch_long": -80.2,
        "hour": 99,
    }
    print("Enviando:", invalid)
    resp = requests.post(f"{BASE_URL}/v1/predict", json=invalid, headers=HEADERS)
    print("Status code:", resp.status_code)
    print("Detalle de validación:")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


def check_invalid_token() -> None:
    separator("3) token invalido  (esperamos 403)")
    transaction = {
        "amt": 42.0,
        "category": "grocery_pos",
        "gender": "M",
        "city_pop": 800000,
        "lat": 40.7,
        "long": -73.9,
        "merch_lat": 40.7,
        "merch_long": -73.9,
        "hour": 14,
        "age": 50,
    }
    print("Enviando con token equivocado:", HEADERS_BAD)
    resp = requests.post(f"{BASE_URL}/v1/predict", json=transaction, headers=HEADERS_BAD)
    print("Status code:", resp.status_code)
    print("Respuesta  :", resp.json())


def check_health() -> None:
    separator("4) health check")
    resp = requests.get(f"{BASE_URL}/health")
    print("Status code:", resp.status_code)
    print("Respuesta  :", resp.json())


def main() -> None:
    try:
        check_valid_case()
        check_invalid_case()
        check_invalid_token()
        check_health()
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] No se pudo conectar a la API.\n"
            "Proba correr:\n"
            "    ./.venv/bin/uvicorn fraud.api.main:app --reload --port 8080"
        )


if __name__ == "__main__":
    main()
