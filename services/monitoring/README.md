# Servicio de Monitoreo y Detección de Drift

Capa de observabilidad del sistema de detección de fraude. Cubre dos ejes:

1. **Drift de datos / predicciones** con **Evidently** (reporte HTML offline).
2. **Métricas operativas** (latencia p95, throughput, tasa de error) con
   **Prometheus + Grafana** (dashboards en vivo).

Todo lo de esta capa vive dentro de `services/monitoring/`. Este servicio **no**
implementa las métricas de serving: consume los `/metrics` que exponen los
servicios REST/gRPC (ver [Puntos de integración](#puntos-de-integración)).

---

## Estructura

```
services/monitoring/
├── evidently_report.py                      # CLI (typer) que genera el reporte de drift
├── requirements.txt                         # evidently, pandas, numpy, loguru, typer, pyarrow
├── Dockerfile                               # imagen python:3.12-slim para correr el reporte
├── reports/                                 # salida HTML de Evidently
├── prometheus/
│   └── prometheus.yml                       # scrape de rest:8080 y grpc:9100 cada 15s
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml       # datasource -> http://prometheus:9090
    │   └── dashboards/dashboard.yml         # provider que carga los JSON de dashboards
    └── dashboards/fraud.json                # dashboard: p95, throughput, tasa de error
```

---

## 1. Reporte de drift con Evidently

`evidently_report.py` compara un dataset **reference** (baseline) contra un
dataset **current** y produce:

- Un **HTML** en `reports/` con **Data Drift** (siempre) y, si el dataset
  incluye columnas de `target` (`is_fraud`) y/o `prediction`, además
  **Target/Prediction Drift**.
- Un **resumen en logs** (loguru): columnas con drift, cantidad de columnas
  driftadas y *share of drifted features*.

Features usadas para el mapping (coinciden con el modelo XGBoost de serving):

- Numéricas: `amt`, `city_pop`, `lat`, `long`, `merch_lat`, `merch_long`,
  `hour`, `age`.
- Categóricas: `category`, `gender`.
- Target: `is_fraud`. Predicción (opcional): columna `prediction`.

### Versión de Evidently asumida

Se asume **Evidently 0.4.x** (fijado `evidently==0.4.40` en
`requirements.txt`), que usa la API "clásica":

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=ref, current_data=cur, column_mapping=mapping)
report.save_html("reports/drift_report.html")
report.as_dict()  # resumen para logs
```

> Nota de compatibilidad: en Evidently **0.6+/0.7** la API cambió a
> `from evidently import Report` y `from evidently.presets import DataDriftPreset`,
> con un objeto `Dataset`/`DataDefinition` en lugar de `ColumnMapping`. Si se
> actualiza la librería, hay que adaptar los imports y el `run()`. Este servicio
> se mantiene en 0.4.x por estabilidad de la API de presets.

### Correr localmente

Desde la raíz del repo, con el venv del proyecto:

```bash
# Reference = train, Current = test
./.venv/bin/python services/monitoring/evidently_report.py run \
  --reference data/processed/train.parquet \
  --current   data/processed/test.parquet \
  --output    services/monitoring/reports/drift_report.html

# Forzar drift para una demo (shift sintético sobre amt y hour del current)
./.venv/bin/python services/monitoring/evidently_report.py run \
  --reference data/processed/train.parquet \
  --current   data/processed/test.parquet \
  --inject-shift \
  --output    services/monitoring/reports/drift_report_shift.html

# Además volcar el resumen crudo (as_dict) a JSON
./.venv/bin/python services/monitoring/evidently_report.py run \
  --summary-json services/monitoring/reports/summary.json
```

Ver todas las opciones: `./.venv/bin/python services/monitoring/evidently_report.py run --help`.

### Correr en Docker

```bash
# Build (desde services/monitoring/)
docker build -t fraud-monitoring services/monitoring

# Run: se montan los datos (read-only) y la carpeta de reports (salida)
docker run --rm \
  -v "$PWD/data:/data:ro" \
  -v "$PWD/services/monitoring/reports:/app/reports" \
  fraud-monitoring \
  run --reference /data/processed/train.parquet \
      --current   /data/processed/test.parquet \
      --output    /app/reports/drift_report.html
```

El HTML queda en `services/monitoring/reports/` y se abre en el navegador.

---

## 2. Métricas operativas: Prometheus + Grafana

### Prometheus

`prometheus/prometheus.yml` scrapea cada **15s** los endpoints `/metrics` de:

- `rest:8080`  (job `fraud-rest`)  — servicio de serving REST/FastAPI.
- `grpc:9100`  (job `fraud-grpc`)  — exporter de métricas del servicio gRPC.
- `localhost:9090` (job `prometheus`) — auto-scrape para verificar salud.

Prometheus queda accesible en `http://localhost:9090`.

### Grafana

Al arrancar, Grafana auto-provisiona:

- **Datasource** Prometheus apuntando a `http://prometheus:9090`
  (`grafana/provisioning/datasources/prometheus.yml`).
- **Dashboard** "Fraud Detection - Serving Metrics"
  (`grafana/dashboards/fraud.json`, cargado vía
  `grafana/provisioning/dashboards/dashboard.yml`).

Grafana queda en `http://localhost:3000` (usuario/clave por defecto
`admin`/`admin` salvo que se configure otra cosa por variable de entorno).

### Paneles del dashboard

Las consultas asumen los nombres de métrica por defecto de
`prometheus-fastapi-instrumentator`:

| Panel | Métrica / PromQL |
| --- | --- |
| **Latencia p95 (s)** | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))` |
| **Throughput (req/s)** | `sum(rate(http_requests_total[1m])) by (service)` |
| **Tasa de error** | `sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) / clamp_min(sum(rate(http_requests_total[5m])) by (service), 1)` |

Si el exporter usa otros nombres de métrica, ajustar las expresiones en
`fraud.json`.

---

## Puntos de integración

> **Responsabilidad de los servicios de serving, no de esta capa.**

Para que Prometheus pueda scrapear, los servicios deben exponer `/metrics`:

- **REST (FastAPI):** agregar
  [`prometheus-fastapi-instrumentator`](https://github.com/trallnag/prometheus-fastapi-instrumentator):

  ```python
  from prometheus_fastapi_instrumentator import Instrumentator
  Instrumentator().instrument(app).expose(app)  # publica /metrics en :8080
  ```

- **gRPC:** no expone HTTP nativamente. Se asume un exporter/sidecar que publica
  métricas Prometheus en `:9100` (p.ej. `py-grpc-prometheus` + un pequeño HTTP
  server con `prometheus_client.start_http_server(9100)`).

---

## Fragmento docker-compose

Servicios a integrar en el `docker-compose.yml` **raíz** (esta capa no lo
modifica; queda como referencia para quien integre). Rutas relativas a la raíz
del repo.

```yaml
services:
  prometheus:
    image: prom/prometheus
    container_name: fraud-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./services/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
    # depends_on: [rest, grpc]   # los targets del scrape

  grafana:
    image: grafana/grafana
    container_name: fraud-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_USER=admin
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./services/monitoring/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./services/monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
    depends_on:
      - prometheus

  # Opcional: job batch que genera el reporte de drift y termina.
  evidently:
    build: ./services/monitoring
    container_name: fraud-evidently
    volumes:
      - ./data:/data:ro
      - ./services/monitoring/reports:/app/reports
    command:
      - run
      - "--reference=/data/processed/train.parquet"
      - "--current=/data/processed/test.parquet"
      - "--output=/app/reports/drift_report.html"
```

**Puertos:** Prometheus `9090`, Grafana `3000`. El servicio `evidently` es un
job batch (corre y termina; no expone puertos).
