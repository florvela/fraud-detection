# fraud-detection

![](https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter)

MLOPs project for fraud detection

Detección de fraude con tarjeta de crédito con un modelo XGBoost.

Dataset: `pointe77/credit-card-transaction`
(Hugging Face).

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Pipeline (datos → modelo)

```bash
# 1. Ingesta: baja una muestra a data/raw/ (distribución real ~1% fraude)
./.venv/bin/python -m fraud.dataset               # o --full para el dataset completo

# 2. Features + split estratificado train/val/test -> data/processed/
./.venv/bin/python -m fraud.features

# 3. Entrenamiento (rebalanceo solo en train) -> models/model.joblib
./.venv/bin/python -m fraud.modeling.train
```

Exploración y evaluación visual en `notebooks/01_explore_data.ipynb` y
`notebooks/02_evaluate_model.ipynb`.

## Correr la API en local

```bash
./.venv/bin/uvicorn fraud.api.main:app --reload --port 8080
```

- Docs (Swagger): **[http://localhost:8080/docs](http://localhost:8080/docs)**
- `GET /health` (público) · `POST /v1/predict` (requiere header `X-API-KEY`)
- El token se lee de la env `API_KEYS` (default: `token-secreto-123`)



## Correr la API en Docker

```bash
docker build -f docker/api.Dockerfile -t fraud-api .
docker run -p 8080:8080 -e API_KEYS="token-secreto-123" fraud-api
```

La imagen solo **sirve** el modelo: hay que tener `models/model.joblib` generado
(paso 3 del pipeline) antes de construirla.

## Tests

Probar con pytest:

```bash
./.venv/bin/python -m pytest -v
```

**Cliente de ejemplo** (con la API corriendo en otra terminal):

```bash
./.venv/bin/python client.py     # probamos caso válido (200), inválido (422) y token inválido (403)
```

**curl** (con la API corriendo):

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/v1/predict \
  -H "X-API-KEY: token-secreto-123" -H "Content-Type: application/json" \
  -d '{"amt":950.75,"category":"shopping_net","gender":"F","city_pop":15000,
       "lat":40.1,"long":-74.5,"merch_lat":41.9,"merch_long":-80.2,"hour":2,"age":35}'
# {"is_fraud":true,"probability":0.9563,"model_version":"1.0.0"}
```

Respuestas: `200` válido · `422` datos inválidos · `403` token ausente/incorrecto.

## GraphQL (metadatos del modelo)

La misma API expone los metadatos del modelo por GraphQL en `/graphql` (con
GraphiQL habilitado). Con la API corriendo:

```bash
./.venv/bin/python client_graphql.py      # cliente Python de ejemplo
```

Query de ejemplo (pegar en GraphiQL: [http://localhost:8080/graphql](http://localhost:8080/graphql)):

```graphql
{ model { name version metrics { rocAuc prAuc } } }
```



## REST vs GraphQL

Para armar la **misma vista** (`name` + `version` del modelo):


|                             | Llamadas | Bytes |
| --------------------------- | -------- | ----- |
| REST (`GET /v1/model-info`) | 1        | 654   |
| GraphQL (`POST /graphql`)   | 1        | 68    |


Misma cantidad de llamadas, pero **REST transfiere ~10× más bytes**: el endpoint
REST devuelve *todo* el metadata (features, categorías, dataset, `trained_at`…) aunque solo pidamos 2 campos.. **over-fetching**. GraphQL devuelve exactamente lo
que la query pide. 

Reproducilo:

```bash
./.venv/bin/python compare_rest_graphql.py
```



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         fraud and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── fraud   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes fraud a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

---

