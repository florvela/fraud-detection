# TPs 1-3: REST vs GraphQL vs gRPC (ml_services_comparison)

Comparación de **REST vs GraphQL vs gRPC** sirviendo el **mismo modelo de fraude**,
cada protocolo en su propio contenedor Docker (cada servicio trae su copia de
`model.joblib`). Cubre los tres mini-TPs:

- **TP1 (REST)** — FastAPI con validación, API-key y versionado (`rest_service/`).
- **TP2 (GraphQL)** — Strawberry expone metadatos del modelo y el **linaje
  dato→feature→modelo** leído de **Neo4j** (`graphql_service/` + `lineage.py`).
- **TP3 (gRPC)** — contrato `.proto` tipado con unary + server-streaming (`grpc_service/`).

Basado en `clase2` (GraphQL + Neo4j) y `clase3/Practica/gRPC_GraphQL_REST.ipynb`.

## Estructura

```
TPs/tp1-3-protocolos/
├── docker-compose.yml
├── grpc_service/      # gRPC  (puerto contenedor 50051 -> host 50052)
│   ├── Dockerfile     #   genera los stubs del .proto durante el build
│   ├── main.py
│   ├── model.joblib
│   └── proto/ml_service.proto
├── graphql_service/   # GraphQL (puerto 8000)
│   ├── main.py        #   predict + model{name,version,metrics,lineage}
│   └── lineage.py     #   TP2: linaje dato->feature->modelo leído de Neo4j
├── rest_service/      # REST    (puerto 8001)
├── client/            # cliente que mide latencia contra los 3
│   ├── client.py      #   comparación de latencia REST/GraphQL/gRPC
│   ├── demo_rest.py   #   TP1: 200/422/403 + health + model-info
│   ├── demo_graphql.py#   TP2: metadatos + comparación REST vs GraphQL
│   ├── demo_grpc.py   #   TP3: unary + streaming
│   └── proto/ml_service.proto
└── (Neo4j)            # servicio del compose para el linaje del TP2 (7474/7687)
```

> Nota: cada servicio tiene su copia de `model.joblib` porque en microservicios
> los contenedores son independientes (no comparten sistema de archivos).

## Puertos

| Servicio | Host | Contenedor |
|----------|------|------------|
| gRPC     | 50052 | 50051 (el 50051 del host lo usa Multipass) |
| GraphQL  | 8000  | 8000 |
| REST     | 8001  | 8001 |
| Neo4j (browser) | 7474 | 7474 |
| Neo4j (bolt)    | 7687 | 7687 |

## Cómo correrlo

```bash
cd TPs/tp1-3-protocolos

# 1. Construir y levantar los 3 servidores + Neo4j (para el linaje del TP2)
docker compose up --build -d

# 2. Ver que están arriba
docker compose ps

# 3. Correr el cliente comparador (mide latencia REST vs GraphQL vs gRPC)...
#    a) dentro de la red de Docker (se conecta por nombre de servicio)
docker compose run --rm client
```

```bash
#    b) o desde tu terminal local, con un venv propio del cliente
cd client
python3 -m venv .venv
./.venv/bin/pip install grpcio grpcio-tools requests
./.venv/bin/python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. ./proto/ml_service.proto
./.venv/bin/python client.py
```

## Probar a mano

```bash
# REST
curl -s localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"amt":950.75,"category":"shopping_net","gender":"F","city_pop":15000,"lat":40.1,"long":-74.5,"merch_lat":41.9,"merch_long":-80.2,"hour":2,"age":35}'
```

**GraphQL** — abrí **http://localhost:8000/graphql** (GraphiQL) y pegá alguna de estas queries:

```graphql
# a) metadatos del modelo + linaje dato->feature->modelo (Neo4j, TP2)
{ model { name version metrics { rocAuc prAuc } lineage { name kind } } }
```

```graphql
# b) scoring por GraphQL (misma transacción que REST/gRPC)
query {
  predict(transaction: {
    amt: 950.75, category: "shopping_net", gender: "F", cityPop: 15000,
    lat: 40.1, long: -74.5, merchLat: 41.9, merchLong: -80.2, hour: 2, age: 35
  }) { isFraud probability modelVersion }
}
```

O por `curl`:

```bash
curl -s localhost:8000/graphql -H "Content-Type: application/json" \
  -d '{"query":"{ model { name version metrics { rocAuc prAuc } lineage { name kind } } }"}'
```

> El campo `lineage` sale de **Neo4j**; el navegador del grafo está en
> **http://localhost:7474** (usuario `neo4j` / pass `testpass`). Si Neo4j está
> caído, `lineage` degrada a `[]` y el resto sigue andando. Detalle en
> [`docs/03_neo4j.md`](docs/03_neo4j.md).

## Limpieza

```bash
docker compose down
```

## Nota: este TP es autocontenido (no depende del repo raíz)

Todo lo necesario para correr la comparación vive en esta carpeta: cada servicio
trae su `model.joblib` y su `Dockerfile`, y el `client/` mide la latencia contra
los tres contenedores. Con sólo `TPs/tp1-3-protocolos/` alcanza.

> Los clientes que apuntaban al **sistema integrado de producción** del repo
> (`fraud.api` en `localhost:8080`/`50052`, con API-key, `/v1/model-info`, etc.)
> ya **no** viven acá: son parte de prod y se movieron a
> [`fraud/api/clients/`](../../fraud/api/clients/) en la raíz del repo
> (`make compare` los usa). Este TP y ese sistema son dos cosas separadas.
