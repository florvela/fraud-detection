# 01 - API REST

La API sirve el modelo de fraude por REST con FastAPI.

## Endpoints


| Método | Ruta             | Auth        | Qué hace                                 |
| ------ | ---------------- | ----------- | ---------------------------------------- |
| `GET`  | `/health`        | pública     | estado del servicio + versión del modelo |
| `POST` | `/v1/predict`    | `X-API-KEY` | predice fraude para una transacción      |
| `GET`  | `/v1/model-info` | pública     | metadatos completos del modelo           |
| `GET`  | `/docs`          | pública     | Swagger UI                               |




## Contrato de entrada (`Transaction`)


| Campo                      | Tipo  | Validación                          |
| -------------------------- | ----- | ----------------------------------- |
| `amt`                      | float | `> 0`                               |
| `category`                 | enum  | una de las 14 categorías del modelo |
| `gender`                   | enum  | `F` / `M`                           |
| `city_pop`                 | int   | `>= 0`                              |
| `lat` / `long`             | float | rango de coordenadas                |
| `merch_lat` / `merch_long` | float | rango de coordenadas                |
| `hour`                     | int   | `0–23`                              |
| `age`                      | int   | `0–120`                             |


Campos extra ====> rechazados (`422`). Respuesta: `is_fraud`, `probability`, `model_version`.

Códigos: 

- `200` válido 
- `422` datos inválidos 
- `403` token ausente/incorrecto.



## Seguridad (API Key)

El token se lee de la env `API_KEYS` (varios separados por coma). Sin definirla se usa el token de desarrollo `token-secreto-123`. Nunca se hardcodea en el código.

## Probar en local

```bash
./.venv/bin/uvicorn fraud.api.main:app --reload --port 8080
```

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/v1/predict \
  -H "X-API-KEY: token-secreto-123" -H "Content-Type: application/json" \
  -d '{"amt":950.75,"category":"shopping_net","gender":"F","city_pop":15000,
       "lat":40.1,"long":-74.5,"merch_lat":41.9,"merch_long":-80.2,"hour":2,"age":35}'
```

Cliente de ejemplo: `./.venv/bin/python client.py` (casos 200 / 422 / 403)

## Probar en Docker

```bash
docker build -f docker/api.Dockerfile -t fraud-api .
docker run -p 8080:8080 fraud-api
docker run -p 8080:8080 -e API_KEYS="mi-token" fraud-api   # token propio
```

La imagen solo sirve el modelo: requiere `models/model.joblib` ya generado

## Tests

```bash
./.venv/bin/python -m pytest tests/test_api.py -v
```

