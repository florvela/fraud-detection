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

## Probar (Docker, autocontenido)

El `graphql_service/` de este TP se publica en el puerto **8000**:

```bash
cd TPs/tp1-3-protocolos
docker compose up --build -d graphql_service neo4j
```

GraphiQL en el navegador: **[http://localhost:8000/graphql](http://localhost:8000/graphql)**

```graphql
{ model { name version metrics { rocAuc prAuc } lineage { name kind } } }
```

curl:

```bash
curl -X POST http://localhost:8000/graphql \
  -H "Content-Type: application/json" \
  -d '{"query":"{ model { name version } }"}'
```

Cliente Python (metadatos + comparación REST vs GraphQL):

```bash
docker compose run --rm client python demo_graphql.py
```



## REST vs GraphQL

Para armar la **misma vista** (`name` + `version`), ambos hacen **1 llamada**,
pero REST (`GET /v1/model-info`, puerto 8001) devuelve **todo** el metadata del
modelo aunque sólo pidamos 2 campos (**over-fetching**), mientras que GraphQL
(`POST /graphql`, puerto 8000) trae exactamente lo pedido. En esta prueba REST
transfiere ~10× más bytes. Reproducir con el cliente del TP:

```bash
docker compose run --rm client python demo_graphql.py
```



## Tests

```bash
# desde graphql_service/, con sus deps instaladas:
cd graphql_service && python -m pytest test_graphql_service.py -v
```

