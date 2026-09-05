"""Linaje del modelo en Neo4j: siembra el grafo dato-feature-modelo y lo consulta"""

from __future__ import annotations

import os

from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "testpass")


def _driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def seed(model_name: str = "fraud-detection") -> None:
    with _driver() as driver, driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        session.run(
            """
            CREATE (d:Artifact {name:$dataset, kind:'dataset'})
            CREATE (f:Artifact {name:$features, kind:'feature'})
            CREATE (m:Model {name:$model, kind:'model'})
            CREATE (d)-[:DERIVES]->(f)-[:DERIVES]->(m)
            """,
            dataset="credit-card-transaction",
            features="features(amt,hour,age,category,...)",
            model=model_name,
        )


def get_lineage(model_name: str) -> list[dict]:
    query = """
        MATCH path=(a)-[:DERIVES*]->(m:Model {name:$name})
        RETURN a.name AS name, a.kind AS kind
        ORDER BY length(path) DESC
    """
    try:
        with _driver() as driver, driver.session() as session:
            return [dict(record) for record in session.run(query, name=model_name)]
    except Exception:
        return []
