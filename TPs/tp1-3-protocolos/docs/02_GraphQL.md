# 02 - GraphQL

La misma API expone los **metadatos del modelo** por GraphQL (Strawberry) en `/graphql`, con GraphiQL habilitado. GraphQL deja pedir *solo los campos necesarios*, a diferencia de REST que devuelve todo.

## Esquema (SDL)

```graphql
type Metrics { rocAuc: Float!  prAuc: Float! }
type LineageNode { name: String!  kind: String! }
type Model {
  name: String!
  version: String!
  metrics: Metrics!
  lineage: [LineageNode!]!     # ver 03_neo4j.md
}
type Query { model: Model! }
```

## Probar en local

```bash
./.venv/bin/uvicorn fraud.api.main:app --reload --port 8080
```

GraphiQL en el navegador: **[http://localhost:8080/graphql](http://localhost:8080/graphql)**

```graphql
{ model { name version metrics { rocAuc prAuc } } }
```

Cliente Python: `./.venv/bin/python client_graphql.py`

curl:

```bash
curl -X POST http://localhost:8080/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ model { name version } }"}'
```



## Probar en Docker

GraphQL viaja dentro de la misma imagen que REST:

```bash
docker build -f docker/api.Dockerfile -t fraud-api .
docker run -p 8080:8080 fraud-api
# GraphiQL en http://localhost:8080/graphql
```



## REST vs GraphQL

Para armar la **misma vista** (`name` + `version`):


|                             | Llamadas | Bytes |
| --------------------------- | -------- | ----- |
| REST (`GET /v1/model-info`) | 1        | 654   |
| GraphQL (`POST /graphql`)   | 1        | 68    |


Misma cantidad de llamadas, pero REST transfiere ~10× más bytes: devuelve todo el metadata aunque solo pidamos 2 campos (**over-fetching**). GraphQL trae exactamente lo pedido. Reproducir:

```bash
./.venv/bin/python compare_rest_graphql.py
```



## Tests

```bash
./.venv/bin/python -m pytest tests/test_graphql.py -v
```

