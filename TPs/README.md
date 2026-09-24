# TPs

Trabajos prácticos standalone de la materia (no forman parte del sistema
integrador de producción — ver el `docker-compose.yml` de la raíz del repo
para eso). Cada carpeta es autocontenida: tiene su propio `docker-compose.yml`
y no depende del compose raíz.

| Carpeta | TP | Qué es | Cómo levantarlo |
|---|---|---|---|
| [`tp1-3-protocolos/`](tp1-3-protocolos/) | TP1 (REST) · TP2 (GraphQL + Neo4j) · TP3 (gRPC) | Comparación REST vs GraphQL vs gRPC sirviendo el mismo modelo de fraude, cada protocolo en su propio contenedor, más docs de cada protocolo en `docs/` | `cd TPs/tp1-3-protocolos && docker compose up --build -d` |
| [`tp4-streaming/`](tp4-streaming/) | TP4 (streaming) | Scoring de fraude en tiempo real (Redpanda/Kafka) con detección de drift (PSI) y alertas, más comparación batch vs streaming | `cd TPs/tp4-streaming && docker compose up --build -d` |

Detalle de cada uno en su propio README.
