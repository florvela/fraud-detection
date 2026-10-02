# Interfaz del analista de fraude (Streamlit)

UI web para el analista de fraude. Es la cara humana del sistema champion/challenger:
consume la **API REST** (FastAPI) para scorear transacciones, resolver la cola de
revisión (generando los *labels reales* que alimentan el reentrenamiento) y operar
el ciclo de vida del modelo (evaluar / desplegar challenger).

## Qué hace

La app (`app.py`) tiene cuatro vistas, organizadas en pestañas:

- **Scoring manual**: formulario con las features de una transacción. Hace
  `POST /v1/predict` (header `X-API-KEY`) y muestra la **decisión del champion**
  (aprobada / en revisión) y, si hay un **challenger** activo, su predicción en
  sombra (que no decide).
- **Cola de revisión** (UC2/UC3): lista las transacciones retenidas con
  `GET /v1/reviews?status=PENDING`; por cada una, el analista **Aprueba**
  (legítima → `POST …/decision {approve}`), **Rechaza** (fraude →
  `{reject}`) o **Denuncia** a posteriori (`POST …/report-fraud`). Cada acción
  persiste el label real en el store del servidor por `transaction_id`.
- **MLOps** (UC5/UC6): **Evaluar** challenger vs champion
  (`POST /v1/mlops/evaluate`) contra la ground truth acumulada, y **Deploy** /
  **Rollback** (`POST /v1/mlops/deploy?action=promote|rollback`).
- **Modelo**: muestra `GET /v1/model-info` (nombre, versión y métricas).

La barra lateral hace un chequeo de salud (`GET /health`) y muestra la config
activa. Si la REST no está disponible, la UI muestra un mensaje claro en vez de
un stacktrace. Ya no usa archivos CSV: todo el estado (transacciones y labels)
vive en el store del servidor.

## Variables de entorno

| Variable   | Default                 | Descripción                                  |
|------------|-------------------------|----------------------------------------------|
| `REST_URL` | `http://rest:8080`      | URL base de la API REST.                     |
| `API_KEY`  | `token-secreto-123`     | Token enviado en el header `X-API-KEY`.      |

## Cómo correr en local

Desde este directorio (`services/ui/`):

```bash
# Instalar dependencias (por ejemplo en un venv del proyecto)
./.venv/bin/pip install -r requirements.txt

# Levantar la app apuntando a la REST local (ajustá la URL a tu entorno)
REST_URL=http://localhost:8080 API_KEY=token-secreto-123 \
  ./.venv/bin/streamlit run app.py
```

Si tenés `streamlit` en el PATH, alcanza con:

```bash
REST_URL=http://localhost:8080 streamlit run app.py
```

La UI queda disponible en http://localhost:8501

## Cómo correr en Docker

```bash
# Construir la imagen desde este directorio
docker build -t fraud-ui .

# Correr apuntando a una REST accesible desde el contenedor
docker run -p 8501:8501 \
  -e REST_URL=http://host.docker.internal:8080 \
  -e API_KEY=token-secreto-123 \
  fraud-ui
```

## Fragmento docker-compose

Para integrar la UI en el `docker-compose` raíz junto al resto de servicios
(asume que el servicio REST se llama `rest` y expone el puerto `8080`):

```yaml
  ui:
    build: ./services/ui
    ports:
      - "8501:8501"
    depends_on:
      - rest
    environment:
      REST_URL: http://rest:8080
      API_KEY: token-secreto-123
```

La UI queda en http://localhost:8501
