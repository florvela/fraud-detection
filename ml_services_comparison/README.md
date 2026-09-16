# ml_services_comparison

Comparación de **REST vs GraphQL vs gRPC** sirviendo el **mismo modelo de fraude**
(`models/model.joblib` del repo), cada protocolo en su propio contenedor Docker.

Basado en `clase3/Practica/gRPC_GraphQL_REST.ipynb` y `mini_tp3_actividad.ipynb`.

## Estructura

```
ml_services_comparison/
├── docker-compose.yml
├── grpc_service/      # gRPC  (puerto contenedor 50051 -> host 50052)
│   ├── Dockerfile     #   genera los stubs del .proto durante el build
│   ├── main.py
│   ├── model.joblib
│   └── proto/ml_service.proto
├── graphql_service/   # GraphQL (puerto 8000)
├── rest_service/      # REST    (puerto 8001)
└── client/            # cliente que mide latencia contra los 3
    └── proto/ml_service.proto
```

> Nota: cada servicio tiene su copia de `model.joblib` porque en microservicios
> los contenedores son independientes (no comparten sistema de archivos).

## Puertos

| Servicio | Host | Contenedor |
|----------|------|------------|
| gRPC     | 50052 | 50051 (el 50051 del host lo usa Multipass) |
| GraphQL  | 8000  | 8000 |
| REST     | 8001  | 8001 |

## Cómo correrlo

```bash
cd ml_services_comparison

# 1. Construir y levantar los 3 servidores
docker compose up --build -d

# 2. Ver que están arriba
docker compose ps

# 3. Correr el cliente comparador...
#    a) dentro de la red de Docker (se conecta por nombre de servicio):
docker compose run --rm client
#    b) o desde tu terminal local (usa el .venv del repo, apunta a los puertos publicados):
cd client && ../../.venv/bin/python client.py
```

## Probar a mano

```bash
# REST
curl -s localhost:8001/predict -H "Content-Type: application/json" \
  -d '{"amt":950.75,"category":"shopping_net","gender":"F","city_pop":15000,"lat":40.1,"long":-74.5,"merch_lat":41.9,"merch_long":-80.2,"hour":2,"age":35}'

# GraphQL (abrir en el navegador: http://localhost:8000/graphql)
```

## Limpieza

```bash
docker compose down
```
