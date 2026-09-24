# Servicio de Aprendizaje Federado (Flower + MLP tabular)

Capa de **Aprendizaje Federado** del TP de MLOps II (detección de fraude con
tarjeta). Entrena un **MLP tabular** de forma **federada** entre dos "bancos"
(Banco A y Banco B) que **no comparten sus transacciones**: solo se agregan los
**pesos** del modelo con la estrategia **FedAvg** de [Flower](https://flower.ai).

---

## ¿Por qué un MLP y no el XGBoost champion?

El modelo de scoring principal del repo (`models/model.joblib`) es un
**XGBoost**. FedAvg (Federated Averaging) funciona **promediando los tensores de
pesos** de un modelo paramétrico entre clientes. Un ensamble de árboles de
gradient boosting **no tiene un vector de pesos homogéneo** que se pueda
promediar: cada cliente aprendería árboles con estructuras (splits, hojas)
distintas, y "promediar árboles" no está definido.

Por eso el camino federado usa un **MLP tabular paramétrico** (entrada → 32 → 16
→ 1). Sus pesos **sí** son tensores del mismo shape en todos los bancos, así que
FedAvg puede promediarlos round a round. El XGBoost sigue siendo el **champion
centralizado** y lo usamos como **línea de base** para medir cuánto rinde el
federado sin compartir datos.

---

## ¿Qué son los silos no-IID?

En federado cada cliente tiene su propio dataset local (**silo**). Si los silos
tuvieran la misma distribución serían **IID** y el problema sería casi como
entrenar centralizado. En la realidad **no** lo son: cada banco opera en otra
región, con otros clientes y otros patrones de gasto → **no-IID**.

**Criterio de partición usado.** El dataset crudo no tiene columna de
estado/región, pero sí coordenadas. Usamos la **longitud (`long`)** como proxy
geográfico y partimos por su mediana:

| Silo    | Región | Condición        |
|---------|--------|------------------|
| Banco A | Oeste  | `long <= mediana` |
| Banco B | Este   | `long >  mediana` |

Esto genera silos con distribuciones espaciales claramente distintas
(`lat`, `long`, `merch_lat`, `merch_long` cambian de forma marcada entre silos),
manteniendo el desbalance real (~1% de fraude) en ambos. El criterio es
configurable (`--criterio long|lat`).

---

## Componentes

| Archivo                  | Rol                                                              |
|--------------------------|-----------------------------------------------------------------|
| `data_partition.py`      | CLI Typer: parte el train en `banco_a.parquet` / `banco_b.parquet` (no-IID). |
| `model.py`               | MLP (PyTorch) + preprocesamiento fijo (one-hot + estandarización). |
| `client.py`              | `flwr.client.NumPyClient` de un banco (entrena local, rebalanceo por `pos_weight`). |
| `server.py`              | Servidor Flower con FedAvg; guarda el modelo global (`models/mlp_federado.npz`). |
| `evaluate_vs_central.py` | CLI Typer: compara MLP federado vs champion XGBoost sobre el test. |

**Preprocesamiento consistente.** Las categóricas se codifican one-hot con un
**vocabulario fijo** y las numéricas se estandarizan con medias/desvíos
**acordados** (guardados en `preprocess_spec.json`). Estas constantes se tratan
como **metadata pública del esquema** (no son transacciones); garantizan que
todos los bancos produzcan vectores de features idénticos, requisito para que
FedAvg promedie pesos compatibles. **El rebalanceo del desbalance se aplica solo
en train** (`pos_weight = n_neg / n_pos` en `BCEWithLogitsLoss`); la evaluación
usa la distribución real.

---

## Cómo se corre en local (servidor + 2 clientes)

```bash
cd services/federated
pip install -r requirements.txt

# 1) Generar los silos no-IID (Banco A / Banco B) y el spec de preprocesamiento.
python data_partition.py

# 2) Levantar el servidor (en una terminal). N rounds configurable.
python server.py --rounds 5

# 3) Levantar los dos bancos (en dos terminales aparte).
python client.py --silo data/banco_a.parquet
python client.py --silo data/banco_b.parquet

# 4) Al terminar los rounds, comparar federado vs champion XGBoost.
python evaluate_vs_central.py
```

Los clientes también aceptan configuración por variables de entorno:
`FED_SILO` (parquet del banco), `FED_SERVER` (`host:port`).
El servidor: `FED_ROUNDS`, `FED_SERVER`.

Salida esperada de la comparación (una tabla como esta):

```
==============================================================================
Modelo                            PR-AUC     ROC-AUC   Recall fraude
------------------------------------------------------------------------------
XGBoost (centralizado)            0.xxxx      0.xxxx          0.xxxx
MLP federado (FedAvg)             0.xxxx      0.xxxx          0.xxxx
==============================================================================

El MLP federado recupera el NN.N% del PR-AUC del modelo centralizado
SIN que los bancos compartan sus transacciones.
```

---

## Cómo se corre en Docker

Un **único Dockerfile** parametrizado por `CMD` sirve para servidor y clientes.

```bash
cd services/federated
docker build -t fraud-federated .
```

Ver el fragmento de `docker-compose` más abajo para orquestar los tres
servicios (`fed-server`, `fed-client-a`, `fed-client-b`).

---

## Fragmento docker-compose

```yaml
services:
  fed-server:
    build: ./services/federated
    container_name: fed-server
    command: ["python", "server.py"]
    environment:
      FED_SERVER: "0.0.0.0:8080"
      FED_ROUNDS: "5"
    volumes:
      # Comparte datos/modelos del repo con el contenedor.
      - ./data:/app/repo_data:ro
      - ./services/federated/models:/app/models
    ports:
      - "8080:8080"

  fed-client-a:
    build: ./services/federated
    container_name: fed-client-a
    command: ["python", "client.py"]
    depends_on:
      - fed-server
    environment:
      FED_SERVER: "fed-server:8080"
      FED_SILO: "/app/data/banco_a.parquet"
    volumes:
      - ./services/federated/data:/app/data:ro

  fed-client-b:
    build: ./services/federated
    container_name: fed-client-b
    command: ["python", "client.py"]
    depends_on:
      - fed-server
    environment:
      FED_SERVER: "fed-server:8080"
      FED_SILO: "/app/data/banco_b.parquet"
    volumes:
      - ./services/federated/data:/app/data:ro
```

> Nota: correr `python data_partition.py` antes de levantar el compose para que
> existan `data/banco_a.parquet` y `data/banco_b.parquet`. Los clientes usan
> `depends_on: fed-server`; cada uno apunta a su silo por la env `FED_SILO`.
