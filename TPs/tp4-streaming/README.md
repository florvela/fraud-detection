# Servicio de Streaming - Detección de fraude online (Mini-TP 4)

Flujo de transacciones scoreadas en tiempo real contra el modelo XGBoost del
repo (`models/model.joblib`), con **métricas por ventana**, **detección de drift**
y **alertas** por umbral. Incluye una comparación **batch vs streaming**.

## Componentes

| Archivo | Rol |
|---|---|
| `producer.py` | Emite transacciones (JSON) con las features del modelo. **Introduce un shift de distribución a mitad del stream** (sube `amt` ~3x y sesga `category` a canales online de alto ticket) para poder detectar drift. |
| `consumer.py` | Consume el flujo, **carga el modelo UNA sola vez**, scorea cada evento **online** y calcula métricas por ventana (throughput, latencia p95, tasa de fraude, PSI de `amt`). Dispara alertas al cruzar umbrales. |
| `batch_compare.py` | Puntúa el **mismo** conjunto de eventos de una sola pasada (batch) y compara tiempo/throughput contra el scoreo online. |
| `common.py` | Utilidades compartidas: carga del modelo, inferencia batch/online, generación de eventos, cola local (modo `sim`) y cálculo de PSI. |

Dos modos por `--source`:
- **`sim`** (equivale a *in-memory*): cola local en `data/stream.jsonl` (JSON Lines), **sin broker**. Ideal para probar sin Docker.
- **`kafka`**: publica/consume de un broker Kafka/Redpanda. Topics: entrada `transactions`, puntuados `scored-transactions`, alertas `fraud-alerts`.

El **contrato del modelo**: `pipeline.predict_proba(X)[:,1]` sobre las columnas
`feature_order` = numéricas `[amt, city_pop, lat, long, merch_lat, merch_long, hour, age]`
+ categóricas `[category, gender]`.

---

## Cómo correr

### 1) Modo `sim` (in-memory, sin Docker)

Requiere `models/model.joblib` real en el repo. Creá un venv descartable **dentro
del servicio** e instalá su `requirements.txt`:

```bash
cd TPs/tp4-streaming
python3.11 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt

# 1. Producir el flujo (600 eventos, drift a partir de la mitad)
./.venv/bin/python producer.py --source sim --total 600 --drift-at 0.5

# 2. Consumir + scorear online + métricas por ventana + alertas
./.venv/bin/python consumer.py --source sim --window 100

# 3. Comparar batch vs streaming sobre el mismo conjunto
./.venv/bin/python batch_compare.py
```

Salidas del modo `sim`: eventos puntuados en `data/scored.jsonl` y alertas en
`data/alerts.jsonl`.

### 2) Modo `kafka` con Redpanda (Docker)

Este TP es **autocontenido**: tiene su propio `docker-compose.yml` (ya no
depende del compose raíz del repo).

```bash
cd TPs/tp4-streaming

# Levanta Redpanda + consumer + producer (el modelo se monta desde ../../models)
docker compose up -d --build

# o paso a paso:
docker compose up -d redpanda
docker compose up stream-consumer   # queda escuchando 'transactions'
docker compose run --rm stream-producer  # emite el flujo hacia Kafka

docker compose down                 # limpieza
```

O corriendo los scripts contra un broker ya levantado:

```bash
./.venv/bin/python producer.py --source kafka --bootstrap-servers localhost:9092 --total 1000
./.venv/bin/python consumer.py --source kafka --bootstrap-servers localhost:9092
```

---

## Métricas y alertas

Por cada ventana de `--window` eventos el consumer reporta:
- **throughput** (ev/s),
- **latencia p95** de scoreo por evento (ms),
- **tasa de fraude** de la ventana,
- **PSI de `amt`** contra la primera ventana (baseline de referencia).

Se dispara **alerta** (log `WARNING` + publicación a `fraud-alerts` / `alerts.jsonl`) cuando:
- `PSI(amt) > 0.2` (drift de entradas alto), o
- `tasa_de_fraude > 5%` en la ventana.

Ejemplo real de una corrida `sim` (600 eventos, ventanas de 100, drift desde el evento 300):

```
[ventana  0] n= 100 | throughput=   746.6 ev/s | p95=1.37 ms | fraude=1.00% | PSI(amt)=0.000
[ventana  1] n= 100 | throughput=   718.6 ev/s | p95=1.30 ms | fraude=0.00% | PSI(amt)=0.403  -> ALERTA drift
[ventana  3] n= 100 | throughput=   817.2 ev/s | p95=1.30 ms | fraude=4.00% | PSI(amt)=12.434 -> ALERTA drift
...
Consumer terminó: 600 eventos en 0.75s (798.7 ev/s) | fraudes=13 (2.17%) | alertas=5
```

`batch_compare.py` sobre el mismo set: batch ~277k ev/s vs streaming ~840 ev/s
(**~330x** más rápido por evento), con **diferencia de probabilidades = 0** (mismo modelo).

---

## Reflexión: streaming vs batch

El **batch** es órdenes de magnitud más eficiente por evento (una sola llamada
vectorizada al pipeline), ideal para reprocesar históricos o scorear millones de
transacciones offline donde la latencia individual no importa. El **streaming** cambia
throughput bruto por **reacción inmediata**: cada transacción se puntúa apenas llega,
habilitando bloquear un fraude en el momento y **observar la salud del modelo en vivo**.
El **drift** lo detectamos con PSI sobre `amt` por ventana: al introducir el shift a
mitad del stream, el PSI saltó de ~0 a >12, muy por encima del umbral de 0.2.
**Al dispararse la alerta** haría: (1) notificar/registrar el incidente, (2) inspeccionar
la ventana sospechosa y validar si es drift real o un problema de datos aguas arriba,
(3) subir el umbral de decisión o pasar a revisión manual mientras dure, y
(4) disparar reentrenamiento/rollback del modelo si el drift persiste.

---

## docker-compose.yml de este TP

Este TP ya **no** se levanta desde el compose raíz del repo (salió de
producción junto con el resto de los TPs). Vive en
[`TPs/tp4-streaming/docker-compose.yml`](docker-compose.yml): Redpanda +
`stream-consumer` + `stream-producer`. El modelo y los datos se montan como
volumen de solo lectura desde la raíz del repo (`../../models`, `../../data`),
**no** se copian dentro de la imagen.
