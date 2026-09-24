"""Compara REST vs GraphQL para armar la misma vista: name + version del modelo"""

import requests

REST_URL = "http://localhost:8080/v1/model-info"
GRAPHQL_URL = "http://localhost:8080/graphql"
GRAPHQL_QUERY = "{ model { name version } }"


def measure_rest() -> tuple[int, int, dict]:
    resp = requests.get(REST_URL)
    data = resp.json()
    view = {"name": data["name"], "version": data["version"]}
    return 1, len(resp.content), view


def measure_graphql() -> tuple[int, int, dict]:
    resp = requests.post(GRAPHQL_URL, json={"query": GRAPHQL_QUERY})
    view = resp.json()["data"]["model"]
    return 1, len(resp.content), view


def main() -> None:
    try:
        rest_calls, rest_bytes, rest_view = measure_rest()
        gql_calls, gql_bytes, gql_view = measure_graphql()
    except requests.exceptions.ConnectionError:
        print("[ERROR] Levanta la API: ./.venv/bin/uvicorn fraud.api.main:app --port 8080")
        return

    print("Vista objetivo: name + version del modelo\n")
    print(f"REST    -> {rest_calls} llamada  | {rest_bytes:>4} bytes | vista: {rest_view}")
    print(f"GraphQL -> {gql_calls} llamada  | {gql_bytes:>4} bytes | vista: {gql_view}")
    print(f"\nREST transfiere {rest_bytes / gql_bytes:.1f}x más bytes (over-fetching: "
          f"devuelve todo el metadata aunque solo pidamos 2 campos)")


if __name__ == "__main__":
    main()
