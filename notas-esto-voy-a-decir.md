# Guion para grabar (15 min) — acciones en `[ ]`, lo que digo en `>`

## 0. Antes de grabar (setup)

```bash
cd /Users/prisci/Documents/MIA/repo/fraud-detection
docker compose --profile full down -v --remove-orphans
rm -f data/fraud.db
SEED_BAD_CHAMPION=1 docker compose --profile full up -d --build
curl -s localhost:8080/health
```

| Qué | URL | Credenciales |
| --- | --- | --- |
| **UI del analista** | http://localhost:8501 | — |
| **Airflow** | http://localhost:8081 | `admin` / `admin` |
| **MLflow** | http://localhost:5001 | — |
| **MinIO** | http://localhost:9001 | `minioadmin` / `minioadmin` |
| **Grafana** | http://localhost:3000 | `admin` / `admin` |
| **Locust** | http://localhost:8089 | — |

---

# PARTE A — El analista

## ACTO 1 — El champion deja pasar un fraude

[UI http://localhost:8501 → **Scoring manual** → botón **🟢** → **Predecir**]

> Cargo una transacción, el modelo la aprueba... pero en realidad es un fraude. Es un
> champion malo a propósito: entrenado con pocas iteraciones y sin balanceo, se le escapan.

## ACTO 2 — Cola de revisión

[**Scoring manual** → botón **🔴** → **Predecir**]
[pestaña **Cola de revisión** → **Rechazar**]

> Esta otra sí la retiene y la manda a revisión. El analista decide: rechazar = fraude.
> Esa decisión se vuelve ground truth para entrenar.

## ACTO 3 — Denuncia post-mortem

[pestaña **Transacciones realizadas** → una aprobada → **Ver** → **Denunciar**]

> El otro camino: una que aprobamos resultó ser fraude. La denuncio después. También
> queda como ground truth.

---

# PARTE B — Ciclo de vida del modelo

## ACTO 4 — Reentrenar: corro los dos pipelines

[pestaña **Modelo** → mostrar champion `0.1.0-pocas-iteraciones` + el hash]
[Airflow http://localhost:8081 → **DAGs** → `fraud_pipeline` → **▶ Trigger**]
[cuando termina → `fraud_federate_pipeline` → **▶ Trigger**]

> Corro dos pipelines.
>
> El de fraude tiene **5 pasos**: *ingest* baja 200.000 transacciones y las sube al
> **data lake (MinIO)**, *features* lee del lake, arma los datos y el split (y los vuelve
> a subir), *merge_feedback* suma lo que etiquetamos recién, *train* entrena el modelo
> bueno (300 árboles + balanceo) y *register* lo sube a MLflow como challenger.
>
> El federado tiene **4 pasos**: parte los datos en dos bancos, los entrena juntos **sin
> compartir datos** (solo viajan los pesos), lo evalúa y lo registra como federado.
> Airflow no entrena: coordina.

[UI → **MLOps** → **🔄 Recargar modelos**]
[**Scoring manual** → **🟢** → **Predecir**]

> Ahora, con los dos modelos nuevos corriendo en sombra: el champion sigue aprobando el
> fraude (~11 %), pero el challenger y el federado lo atrapan (~99 %).

## ACTO 5 — MLflow (registro de modelos)

[MLflow http://localhost:5001 → **Models** → `fraud-detection`]

> Acá viven las versiones y los aliases: champion, challenger y federated. Cada
> entrenamiento queda trazado con sus métricas.

## ACTO 6 — MinIO (data lake)

[MinIO http://localhost:9001 → bucket `data-lake` → `raw/` y `processed/`]
[bucket `mlflow` → `1/<hash>/artifacts/model_artifact/`]

> MinIO es mi data lake tipo S3, con dos usos:
>
> - bucket **`data-lake`**: los **datos**. `raw/` = el dump crudo que bajó el ingest,
> `processed/` = train/val/test. El pipeline lee de acá, no de HuggingFace ni de disco.
> - bucket **`mlflow`**: los **modelos**. La ruta es
> `mlflow / experimento / run / artifacts / model_artifact / model.joblib`; ese hash largo
> es el **ID del run** de MLflow (uno por entrenamiento) y el alias champion/challenger
> apunta a ese archivo.

## ACTO 7 — Evaluar y desplegar

[UI → **MLOps** → **🌱 Sembrar 30 fraudes**]
[**MLOps** → **Evaluar sombra vs champion**]
[**MLOps** → **Deploy challenger → champion**]

> Siembro fraudes reales para tener con qué comparar. Al evaluar, el challenger atrapa
> mucho más fraude que el champion, así que lo promuevo. Mueve el alias en MLflow y
> recarga solo, sin reiniciar nada.

## ACTO 8 — Cerrar el loop (feedback → retrain)

[Airflow → `fraud_pipeline` → **▶ Trigger**]
[task **merge_feedback** → **Logs** → ver `injected=30`]

> El paso *merge_feedback* suma lo que el analista etiquetó. Lo que etiquetamos vuelve
> como datos del próximo modelo: el ciclo se cierra solo.

---

# PARTE C — Observabilidad

## ACTO 9 — Métricas en vivo + carga

[`docker compose --profile loadtest up -d`]
[Locust http://localhost:8089 → **50 users** → **Start**]
[Grafana http://localhost:3000 → dashboard **Fraud Detection**]

> Genero carga con clientes reales y en Grafana veo latencia p95, throughput y errores
> moverse en vivo. Así muestro que el sistema aguanta carga.

## ACTO 10 — Drift

[abrir `services/monitoring/reports/drift_report.html`]

> Evidently compara los datos con los que se entrenó contra los nuevos y detecta si la
> distribución se corrió: la señal de que hay que reentrenar.

---

# PARTE D — Aprendizaje federado

## ACTO 11 — Federado vs centralizado (el punchline)

[UI → **MLOps** → **📊 Comparar federado vs centralizado**]

> El federado recupera cerca del 43 % del rendimiento del modelo centralizado **sin que
> ningún banco comparta una sola transacción**. Ese es el precio de la privacidad, medido.

## ACTO 12 — Federado en sombra + deploy

[**MLOps** → **Evaluar sombra vs champion**]
[**MLOps** → **Deploy federado → champion**]

> El federado corre en sombra igual que el challenger y se puede promover a champion con
> un clic, por el mismo circuito.

## ACTO 13 (opcional) — Riesgo del federado: envenenamiento

[`docker compose run --rm fed-server python demo_poisoning.py --rounds 5 --sample 20000`]

> Un banco malicioso que voltea sus etiquetas corrompe el modelo global. Con las defensas
> —clipping, mediana y ruido— se recupera. Muestra 3 escenarios: limpio, atacado y defendido.

---

# PARTE E — Seguridad (desde la UI)

## ACTO 14 — Roles (RBAC)

[sidebar → **Autenticarse como** → **Analista**]
[pestaña **Seguridad** → **Probar acción MLOps**] → ⛔ **403**
[sidebar → **MLOps** → mismo botón] → ✅

> Tres roles: el analista revisa pero no puede desplegar modelos; MLOps sí. Sin API key
> válida, ni se entra.

## ACTO 15 — Integridad + auditoría

[pestaña **Seguridad** → **Hash SHA-256**]
[pestaña **Seguridad** → **Log de auditoría** → **🔄 Actualizar**]

> El hash detecta si alguien reemplazó el modelo. Y todo queda auditado: qué rol hizo qué
> y cuándo, con el token enmascarado.

## ACTO 16 — Lo que menciono (no es pantalla)

> Rate limiting (se ve en Locust como respuestas 429), validación con Pydantic (el
> formulario solo deja valores válidos), secretos en `.env` fuera de git, y el scoring
> corriendo detrás de gRPC que también pide API key.

---

## Cierre

[`docker compose --profile full down -v --remove-orphans`]

> (Bajar siempre con `--profile full`, si no quedan contenedores colgados.)
