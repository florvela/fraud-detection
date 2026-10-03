# Clientes del sistema integrado (prod)

Scripts de prueba/benchmark que apuntan al **sistema de fraude en producción**
de este repo (`fraud.api`): REST + GraphQL en `localhost:8080` y gRPC en
`localhost:50052`. No tienen nada que ver con los TPs académicos de `TPs/`
(esos son autocontenidos y usan sus propios contenedores).

| Script | Qué prueba |
|---|---|
| `client_rest.py` | REST `/v1/predict`: casos 200 / 422 / 403 + `/health` |
| `client_graphql.py` | Query de metadatos del modelo por GraphQL |
| `client_grpc.py` | gRPC unary + streaming + `GetModelInfo` |
| `compare_rest_graphql.py` | REST vs GraphQL para la misma vista (over-fetching) |
| `compare_protocols.py` | Latencia + bytes por REST vs GraphQL vs gRPC (lo usa `make compare`) |

Requieren el stack de prod levantado y el `.venv` del repo (el paquete `fraud`
está instalado editable). Ejemplos:

```bash
# REST + GraphQL (misma app FastAPI) y, para gRPC, el servidor en otra terminal
./.venv/bin/uvicorn fraud.api.main:app --port 8080
./.venv/bin/python -m fraud.api.grpc_server

./.venv/bin/python fraud/api/clients/compare_protocols.py   # o: make compare
```
