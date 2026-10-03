"""Demo GraphQL del TP2 contra el graphql_service de este TP (no el de prod).

Cubre `docs/02_GraphQL.md` de forma autocontenida:
  1) query de metadatos del modelo (name, version, metrics y linaje Neo4j),
  2) comparación REST vs GraphQL para armar la MISMA vista (name + version):
     cuántas llamadas y cuántos bytes transfiere cada uno (over-fetching de REST).

Hosts configurables por env (defaults: puertos publicados del compose del TP):
    GRAPHQL_URL=http://localhost:8000/graphql REST_BASE=http://localhost:8001 python demo_graphql.py
"""

import json
import os

import requests

GRAPHQL_URL = os.getenv("GRAPHQL_URL", "http://localhost:8000/graphql")
REST_BASE = os.getenv("REST_BASE", "http://localhost:8001")

QUERY_FULL = """
{ model { name version metrics { rocAuc prAuc } lineage { name kind } } }
"""
QUERY_MIN = "{ model { name version } }"


def sep(title: str) -> None:
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)


def main() -> None:
    s = requests.Session()
    try:
        sep("1) metadatos del modelo por GraphQL (pide sólo lo necesario)")
        resp = s.post(GRAPHQL_URL, json={"query": QUERY_FULL})
        print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

        sep("2) misma vista (name + version): REST vs GraphQL")
        # REST: GET /v1/model-info devuelve TODO el metadata (over-fetching).
        rest_resp = s.get(f"{REST_BASE}/v1/model-info")
        rest_data = rest_resp.json()
        rest_view = {"name": rest_data["name"], "version": rest_data["version"]}
        rest_bytes = len(rest_resp.content)

        # GraphQL: pide exactamente name + version.
        gql_resp = s.post(GRAPHQL_URL, json={"query": QUERY_MIN})
        gql_view = gql_resp.json()["data"]["model"]
        gql_bytes = len(gql_resp.content)

        print(f"REST    -> 1 llamada | {rest_bytes:>5} bytes | vista: {rest_view}")
        print(f"GraphQL -> 1 llamada | {gql_bytes:>5} bytes | vista: {gql_view}")
        if gql_bytes:
            print(f"\nREST transfiere ~{rest_bytes / gql_bytes:.1f}x más bytes para la misma "
                  "vista (over-fetching); GraphQL trae sólo lo pedido.")
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] No respondieron los servicios ({GRAPHQL_URL} / {REST_BASE}).\n"
              "Levantá el TP: docker compose up --build -d")


if __name__ == "__main__":
    main()
