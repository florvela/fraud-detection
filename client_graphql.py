"""Cliente GraphQL de prueba: consulta los metadatos del modelo"""

import json

import requests

GRAPHQL_URL = "http://localhost:8080/graphql"

QUERY_MIN = "{ model { name version } }"
QUERY_FULL = "{ model { name version metrics { rocAuc prAuc } } }"


def run(query: str, title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("Query:", query)
    resp = requests.post(GRAPHQL_URL, json={"query": query})
    print("Status code:", resp.status_code)
    print("Respuesta  :", json.dumps(resp.json(), indent=2, ensure_ascii=False))


def main() -> None:
    try:
        run(QUERY_MIN, "1) query mínima (solo name + version)")
        run(QUERY_FULL, "2) query full (con metrics)")
    except requests.exceptions.ConnectionError:
        print(
            "\n[ERROR] No se pudo conectar a la API\n"
            "Proba correr:\n"
            "    ./.venv/bin/uvicorn fraud.api.main:app --reload --port 8080"
        )


if __name__ == "__main__":
    main()
