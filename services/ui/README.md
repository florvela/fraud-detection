# Interfaz del analista de fraude (Streamlit)

UI web para el analista de fraude del sistema de detección de fraude con tarjeta.
Es la cara humana del pipeline: consume la **API REST** (FastAPI) para scorear
transacciones y consultar el modelo, y permite **etiquetar** cada alerta como
fraude o no, generando los *labels reales* que luego alimentan el reentrenamiento.

## Qué hace

La app (`app.py`) tiene tres vistas, organizadas en pestañas:

- **Scoring manual**: formulario con las features de una transacción. Al enviar,
  hace `POST /v1/predict` a la REST (con el header `X-API-KEY`) y muestra la
  probabilidad de fraude y el veredicto del modelo.
- **Bandeja de alertas**: tabla de transacciones sospechosas leídas de
  `data/alerts.csv` (placeholder). Para cada alerta hay botones
  **Confirmar fraude** / **Descartar**, que escriben el label en
  `data/labels.csv` con el formato `transaction_id,label,analyst_ts`
  (`label`: `1` = fraude, `0` = descartada).
- **Modelo**: muestra el resultado de `GET /v1/model-info` (nombre, versión y
  métricas del modelo servido).

La barra lateral hace un chequeo de salud (`GET /health`) y muestra la config
activa. Si la REST no está disponible, la UI muestra un mensaje claro en vez de
un stacktrace.

### Nota sobre el sistema real

- Las alertas de `data/alerts.csv` son un **placeholder**. En el sistema real
  llegarían del topic `fraud-alerts` del broker de streaming.
- Los labels de `data/labels.csv` son un **placeholder** del *store etiquetado*.
  En producción, estos veredictos del analista alimentan el store de labels
  usado para reentrenar el modelo con feedback humano.

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
