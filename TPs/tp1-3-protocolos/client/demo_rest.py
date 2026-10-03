"""Demo REST del TP1 contra el rest_service de este TP (no el sistema de prod).

Muestra los casos que documenta `docs/01_REST.md`: 200 válido, 422 inválido,
403 token equivocado, /health y /v1/model-info.

Host configurable por env (default: puerto publicado del compose del TP):
    REST_BASE=http://localhost:8001 python demo_rest.py
"""

import json
import os

import requests

REST_BASE = os.getenv("REST_BASE", "http://localhost:8001")
API_KEY = os.getenv("API_KEY", "token-secreto-123")

TX = {
    "amt": 950.75, "category": "shopping_net", "gender": "F", "city_pop": 15000,
    "lat": 40.1, "long": -74.5, "merch_lat": 41.9, "merch_long": -80.2,
    "hour": 2, "age": 35,
}


def sep(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def main() -> None:
    s = requests.Session()
    try:
        sep("1) caso válido  (esperamos 200)")
        r = s.post(f"{REST_BASE}/v1/predict", json=TX, headers={"X-API-KEY": API_KEY})
        print(r.status_code, r.json())

        sep("2) caso inválido  (esperamos 422)")
        bad = {**TX, "amt": "no_es_numero", "category": "cripto"}
        r = s.post(f"{REST_BASE}/v1/predict", json=bad, headers={"X-API-KEY": API_KEY})
        print(r.status_code)
        print(json.dumps(r.json(), indent=2, ensure_ascii=False)[:500])

        sep("3) token inválido  (esperamos 403)")
        r = s.post(f"{REST_BASE}/v1/predict", json=TX, headers={"X-API-KEY": "mal"})
        print(r.status_code, r.json())

        sep("4) health + model-info (públicos)")
        print("health     :", s.get(f"{REST_BASE}/health").json())
        info = s.get(f"{REST_BASE}/v1/model-info").json()
        print("model-info :", {k: info[k] for k in ("name", "version", "metrics")})
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] No respondió el rest_service en {REST_BASE}.\n"
              "Levantá el TP: docker compose up --build -d")


if __name__ == "__main__":
    main()
