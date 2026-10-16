"""Interfaz del analista de fraude (Streamlit).

Cara humana del sistema champion/challenger. Consume la API REST (FastAPI):

- **Scoring manual**: POST /v1/predict — muestra la decisión del champion
  (aprobada / en revisión) y, si hay challenger activo, su predicción en sombra.
- **Cola de revisión** (UC2): GET /v1/reviews?status=PENDING; el analista resuelve
  con /decision (aprobar = legítima, rechazar = fraude).
- **Transacciones realizadas**: histórico de ya procesadas (OK/APPROVED/REJECTED)
  vía GET /v1/reviews?status=... + detalle por /v1/transactions/{id}; las que se
  dejaron pasar (OK/APPROVED) se pueden denunciar a posteriori con /report-fraud (UC3).
- **MLOps** (UC5/UC6): compara challenger vs champion (/v1/mlops/evaluate) y
  despliega / revierte (/v1/mlops/deploy).
- **Modelo**: GET /v1/model-info.

Variables de entorno:
    REST_URL  URL base de la API REST (default: http://rest:8080)
    API_KEY   Token para el header X-API-KEY (default: token-secreto-123)
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

REST_URL = os.getenv("REST_URL", "http://rest:8080").rstrip("/")
API_KEY = os.getenv("API_KEY", "token-secreto-123")           # rol admin (hace todo)
HTTP_TIMEOUT = 5

# Tokens por rol (SAIF capa 1). Permiten demostrar el RBAC desde la UI: se elige el rol
# en el sidebar y todas las llamadas usan ese token. Defaults alineados con el compose.
ANALYST_KEY = os.getenv("ANALYST_API_KEY", "analyst-key")
MLOPS_KEY = os.getenv("MLOPS_API_KEY", "mlops-key")
ROLE_TOKENS = {
    "Admin (todo)": API_KEY,
    "Analista (revisión)": ANALYST_KEY,
    "MLOps (modelos)": MLOPS_KEY,
}

# Fallback si la API todavía no está disponible o el modelo no expone categorías.
# Los valores reales se leen del modelo servido (ver `_catalogos`).
CATEGORIES_FALLBACK = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
GENDERS_FALLBACK = ["F", "M"]

# Presets para los botones de "Scoring manual" (precargan el formulario con un clic).
PRESET_ESCAPA = {  # fraude REAL que el champion MALO DEJA PASAR (~11 %, APROBADA)
    "amt": 925.94, "category": "shopping_net", "gender": "M", "city_pop": 4653,
    "lat": 40.5046, "long": -77.7186, "merch_lat": 41.4437, "merch_long": -78.3918,
    "hour": 12, "age": 22,
}
PRESET_REVISION = {  # tx que el champion MALO RETIENE (~92 %, PENDING -> cola de revisión)
    "amt": 800.0, "category": "grocery_pos", "gender": "F", "city_pop": 3500,
    "lat": 41.44, "long": -78.39, "merch_lat": 34.05, "merch_long": -118.24,
    "hour": 2, "age": 55,
}
# key de st.session_state por cada campo del formulario de scoring.
_SM_KEYS = {campo: f"sm_{campo}" for campo in PRESET_ESCAPA}


# --------------------------------------------------------------------------- #
# Helpers de la API REST
# --------------------------------------------------------------------------- #

def _current_key() -> str:
    """Token activo según el rol elegido en el sidebar (default: admin)."""
    import streamlit as st

    return st.session_state.get("api_key", API_KEY)


def _headers() -> dict:
    return {"X-API-KEY": _current_key()}


def predecir(transaccion: dict) -> dict:
    resp = requests.post(
        f"{REST_URL}/v1/predict", json=transaccion, headers=_headers(), timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def listar_reviews(status: str = "PENDING") -> list[dict]:
    resp = requests.get(
        f"{REST_URL}/v1/reviews", params={"status": status}, headers=_headers(), timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def listar_por_estados(estados: list[str]) -> list[dict]:
    """Junta transacciones de varios estados (el endpoint filtra de a uno)."""
    items: list[dict] = []
    for estado in estados:
        items.extend(listar_reviews(estado))
    items.sort(key=lambda t: t.get("ts", ""), reverse=True)
    return items


def obtener_tx(tx_id: str) -> dict:
    resp = requests.get(
        f"{REST_URL}/v1/transactions/{tx_id}", headers=_headers(), timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def decidir_tx(tx_id: str, decision: str) -> None:
    resp = requests.post(
        f"{REST_URL}/v1/transactions/{tx_id}/decision",
        json={"decision": decision},
        headers=_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()


def denunciar_tx(tx_id: str) -> None:
    resp = requests.post(
        f"{REST_URL}/v1/transactions/{tx_id}/report-fraud", headers=_headers(), timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()


def evaluar() -> dict:
    resp = requests.post(f"{REST_URL}/v1/mlops/evaluate", headers=_headers(), timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def comparar_federado() -> dict:
    # Benchmark offline sobre el test (inferencia de ambos modelos): puede tardar unos
    # segundos con ~40k filas, por eso un timeout generoso.
    resp = requests.post(
        f"{REST_URL}/v1/mlops/compare-federated", headers=_headers(), timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def deploy(action: str, source: str = "challenger") -> dict:
    resp = requests.post(
        f"{REST_URL}/v1/mlops/deploy",
        params={"action": action, "source": source},
        headers=_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def recargar_modelos() -> dict:
    resp = requests.post(f"{REST_URL}/v1/mlops/reload", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def sembrar_fraudes() -> dict:
    resp = requests.post(f"{REST_URL}/v1/mlops/seed-frauds", headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def obtener_model_info() -> dict:
    resp = requests.get(f"{REST_URL}/v1/model-info", headers=_headers(), timeout=HTTP_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def obtener_auditoria(limit: int = 50) -> list[dict]:
    """Últimos eventos del log de auditoría (SAIF capa 8)."""
    resp = requests.get(
        f"{REST_URL}/v1/audit", params={"limit": limit}, headers=_headers(), timeout=HTTP_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json().get("entries", [])


def _catalogos() -> tuple[list[str], list[str]]:
    """Categorías y géneros válidos, leídos del modelo servido (con fallback).

    El artefacto del modelo expone `categories` (vocabulario real con el que se
    entrenó); así los selectores de la UI no quedan hardcodeados y acompañan al modelo.
    """
    try:
        info = obtener_model_info()
        cats = (info.get("categories") or {})
        categories = sorted(cats.get("category") or []) or CATEGORIES_FALLBACK
        genders = sorted(cats.get("gender") or []) or GENDERS_FALLBACK
        return categories, genders
    except requests.RequestException:
        return CATEGORIES_FALLBACK, GENDERS_FALLBACK


# --------------------------------------------------------------------------- #
# Vistas
# --------------------------------------------------------------------------- #

def _aplicar_preset(preset: dict, categories: list[str], genders: list[str]) -> None:
    """Carga un preset en el formulario (vía session_state). Clampa category/gender a
    las opciones disponibles del modelo por si su vocabulario difiere."""
    for campo, valor in preset.items():
        if campo == "category" and valor not in categories:
            valor = categories[0]
        if campo == "gender" and valor not in genders:
            valor = genders[0]
        st.session_state[_SM_KEYS[campo]] = valor


def vista_scoring_manual() -> None:
    st.header("Scoring manual")
    st.caption("Cargá una transacción: el champion decide y el challenger (si hay) corre en sombra. "
               "Usá los botones para precargar un caso de ejemplo.")

    categories, genders = _catalogos()

    # Arranca precargado con el caso "fraude que se escapa" (money shot).
    if _SM_KEYS["amt"] not in st.session_state:
        _aplicar_preset(PRESET_ESCAPA, categories, genders)

    # Botones de preset: cambian los valores del formulario de un clic (fuera del form).
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🟢 Fraude que el champion DEJA PASAR", use_container_width=True,
                     help="Precarga un fraude real que el champion malo aprueba (~11 %). "
                          "El challenger bueno lo atrapa — ideal para el contraste."):
            _aplicar_preset(PRESET_ESCAPA, categories, genders)
            st.rerun()
    with b2:
        if st.button("🔴 Fraude que el champion RETIENE (→ revisión)", use_container_width=True,
                     help="Precarga una tx que el champion malo manda a PENDING (~92 %): "
                          "después aparece en la Cola de revisión del analista."):
            _aplicar_preset(PRESET_REVISION, categories, genders)
            st.rerun()

    with st.form("form_scoring"):
        col1, col2 = st.columns(2)
        # Los widgets leen/escriben su valor en session_state vía `key`, así los
        # botones de preset de arriba pueden cambiarlos.
        with col1:
            amt = st.number_input("Monto (amt)", min_value=0.01, step=1.0, key=_SM_KEYS["amt"])
            category = st.selectbox("Rubro (category)", categories, key=_SM_KEYS["category"])
            gender = st.selectbox("Género (gender)", genders, key=_SM_KEYS["gender"])
            city_pop = st.number_input("Población ciudad (city_pop)", min_value=0, step=1000, key=_SM_KEYS["city_pop"])
            hour = st.slider("Hora del día (hour)", 0, 23, key=_SM_KEYS["hour"])
        with col2:
            age = st.slider("Edad del titular (age)", 0, 120, key=_SM_KEYS["age"])
            lat = st.number_input("Latitud titular (lat)", -90.0, 90.0, key=_SM_KEYS["lat"])
            long = st.number_input("Longitud titular (long)", -180.0, 180.0, key=_SM_KEYS["long"])
            merch_lat = st.number_input("Latitud comercio (merch_lat)", -90.0, 90.0, key=_SM_KEYS["merch_lat"])
            merch_long = st.number_input("Longitud comercio (merch_long)", -180.0, 180.0, key=_SM_KEYS["merch_long"])
        enviado = st.form_submit_button("Predecir")

    if not enviado:
        return

    transaccion = {
        "amt": amt, "category": category, "gender": gender, "city_pop": int(city_pop),
        "lat": lat, "long": long, "merch_lat": merch_lat, "merch_long": merch_long,
        "hour": int(hour), "age": int(age),
    }
    try:
        r = predecir(transaccion)
    except requests.RequestException as err:
        st.error(f"No se pudo contactar la API REST en {REST_URL}. Detalle: {err}")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Prob. fraude (champion)", f"{r.get('probability', 0.0):.2%}")
    estado = r.get("status")
    if estado == "PENDING":
        col_b.warning("EN REVISIÓN (retenida)")
    else:
        col_b.success("APROBADA")
    col_c.caption(f"tx: `{r.get('transaction_id', '?')}`\n\nchampion: {r.get('model_version', '?')}")

    challenger = r.get("challenger")
    if challenger:
        st.info(
            f"Challenger en sombra (no decide): "
            f"prob {challenger['probability']:.2%} · fraude={challenger['is_fraud']} · "
            f"v{challenger['model_version']}"
        )
    else:
        st.caption("No hay challenger activo en este momento.")

    federated = r.get("federated")
    if federated:
        st.info(
            f"Federado en sombra (MLP Flower, no decide): "
            f"prob {federated['probability']:.2%} · fraude={federated['is_fraud']} · "
            f"v{federated['model_version']}"
        )
    else:
        st.caption("No hay modelo federado activo en este momento.")


def _features_resumen(tx: dict) -> str:
    f = tx.get("features", {})
    return f"monto {f.get('amt', '?')} · {f.get('category', '?')} · hora {f.get('hour', '?')}"


def vista_cola_revision() -> None:
    st.header("Cola de revisión")
    st.caption("Transacciones retenidas (PENDING). Resolvé: aprobar = legítima · rechazar = fraude.")

    try:
        pendientes = listar_reviews("PENDING")
    except requests.RequestException as err:
        st.error(f"No se pudo contactar la API REST en {REST_URL}. Detalle: {err}")
        return

    if not pendientes:
        st.info("No hay transacciones pendientes de revisión.")
        return

    # Tabla resumen.
    filas = []
    for tx in pendientes:
        champ = tx.get("champion") or {}
        chal = tx.get("challenger") or {}
        fed = tx.get("federated") or {}
        filas.append({
            "transaction_id": tx["transaction_id"],
            "monto": (tx.get("features") or {}).get("amt"),
            "prob_champion": champ.get("probability"),
            "prob_challenger": chal.get("probability"),
            "prob_federado": fed.get("probability"),
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    st.subheader("Resolver")
    for tx in pendientes:
        tx_id = tx["transaction_id"]
        col_info, col_ok, col_no = st.columns([5, 1, 1])
        with col_info:
            st.write(f"**{tx_id}** — {_features_resumen(tx)}")
        with col_ok:
            if st.button("Aprobar", key=f"ok_{tx_id}"):
                decidir_tx(tx_id, "approve")
                st.success(f"{tx_id} aprobada (legítima).")
                st.rerun()
        with col_no:
            if st.button("Rechazar", key=f"no_{tx_id}"):
                decidir_tx(tx_id, "reject")
                st.error(f"{tx_id} rechazada (fraude).")
                st.rerun()


def vista_transacciones() -> None:
    st.header("Transacciones realizadas")
    st.caption(
        "Histórico de transacciones ya procesadas. Las que se dejaron pasar "
        "(aprobadas / auto-aprobadas) pueden denunciarse como fraude a posteriori "
        "(post-mortem)."
    )

    try:
        # OK = auto-aprobada por el champion · APPROVED/REJECTED = resueltas por el analista
        realizadas = listar_por_estados(["OK", "APPROVED", "REJECTED"])
    except requests.RequestException as err:
        st.error(f"No se pudo contactar la API REST en {REST_URL}. Detalle: {err}")
        return

    if not realizadas:
        st.info("Todavía no hay transacciones procesadas.")
        return

    # Tabla resumen.
    filas = []
    for tx in realizadas:
        champ = tx.get("champion") or {}
        filas.append({
            "transaction_id": tx["transaction_id"],
            "estado": tx.get("status"),
            "decisión": tx.get("decision") or "-",
            "monto": (tx.get("features") or {}).get("amt"),
            "prob_champion": champ.get("probability"),
        })
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)

    st.subheader("Detalle y acciones")
    for tx in realizadas:
        tx_id = tx["transaction_id"]
        estado = tx.get("status")
        with st.expander(f"{tx_id} — {estado} — {_features_resumen(tx)}"):
            try:
                detalle = obtener_tx(tx_id)
            except requests.RequestException as err:
                st.error(f"No se pudo traer el detalle. Detalle: {err}")
                continue
            st.json(detalle)

            label = detalle.get("label")
            ya_denunciada = bool(label) and label.get("source") == "post_mortem"

            # Post-mortem sólo tiene sentido en transacciones que se dejaron pasar.
            if estado in ("OK", "APPROVED"):
                if ya_denunciada:
                    st.warning("Ya denunciada como fraude (post-mortem).")
                elif st.button("Denunciar (post-mortem)", key=f"pm_{tx_id}"):
                    denunciar_tx(tx_id)
                    st.warning(f"{tx_id} denunciada como fraude (post-mortem).")
                    st.rerun()
            elif estado == "REJECTED":
                st.caption("Rechazada (ya marcada como fraude): no aplica post-mortem.")


def vista_mlops() -> None:
    st.header("MLOps — champion / challenger / federado")

    if st.button("🔄 Recargar modelos (tras retrain)"):
        try:
            r = recargar_modelos()
            ch = "sí" if r.get("has_challenger") else "no"
            fed = "sí" if r.get("has_federated") else "no"
            st.success(
                f"Recargado. Champion: {r.get('champion_version')} · "
                f"challenger activo: {ch} · federado activo: {fed}"
            )
        except requests.RequestException as err:
            st.error(f"No se pudo recargar: {err}")

    st.divider()

    if st.button("🌱 Sembrar 30 fraudes (post-mortem)"):
        try:
            r = sembrar_fraudes()
        except requests.RequestException as err:
            st.error(f"No se pudo sembrar: {err}")
        else:
            msg = (
                f"Sembrados: {r.get('seeded', 0)} (fuente: {r.get('source', '?')}) · "
                f"con challenger: {r.get('with_challenger', 0)} · "
                f"con federado: {r.get('with_federated', 0)}. {r.get('note', '')}"
            )
            if r.get("seeded", 0) > 0 and r.get("with_challenger", 0) == 0 and r.get("with_federated", 0) == 0:
                st.warning(msg)
            else:
                st.success(msg)

    st.divider()

    if st.button("Evaluar sombra vs champion"):
        try:
            rep = evaluar()
        except requests.RequestException as err:
            st.error(f"No se pudo evaluar. Detalle: {err}")
            return
        st.write(f"Transacciones con ground truth: **{rep.get('n_with_ground_truth', 0)}**")
        comp = {k: rep[k] for k in ("champion", "challenger", "federated") if rep.get(k)}
        if comp:
            # Tabla: mostramos "n/d" en vez de None (p. ej. PR-AUC sin ambas clases).
            tabla = pd.DataFrame(comp).T
            st.dataframe(tabla.where(pd.notnull(tabla), "n/d"), use_container_width=True)

            # Gráfico comparativo: SOLO métricas con valor real en todos los modelos
            # (así PR-AUC no aparece como una barra falsa en 0 cuando viene None).
            METRICAS = ["recall", "precision", "f1", "pr_auc"]
            metricas_ok = [
                m for m in METRICAS if all(datos.get(m) is not None for datos in comp.values())
            ]
            if metricas_ok:
                chart = {
                    modelo: {m: float(datos[m]) for m in metricas_ok}
                    for modelo, datos in comp.items()
                }
                st.bar_chart(pd.DataFrame(chart))  # index = métricas, series = modelos

            if "pr_auc" not in metricas_ok:
                st.caption("ℹ️ **PR-AUC = n/d**: la ground truth tiene **una sola clase** "
                           "(sembraste solo fraudes). Para que se calcule, resolvé en la "
                           "**Cola de revisión** alguna transacción como **Aprobar** (legítima, "
                           "label 0) además de los fraudes — ahí PR-AUC aparece con valor.")
            st.caption("Criterio para elegir: priorizamos **recall de fraude** (atrapar más "
                       "fraude); si empatan, desempata el **PR-AUC**. El cartel de abajo ya "
                       "aplica ese criterio.")
        if rep.get("challenger_better"):
            st.success(f"El challenger mejora al champion ({rep.get('reason', '')}).")
        else:
            st.info(f"El challenger NO mejora al champion ({rep.get('reason', '')}).")
        if rep.get("federated") is not None:
            if rep.get("federated_better"):
                st.success(f"El federado mejora al champion ({rep.get('reason_federated', '')}).")
            else:
                st.info(f"El federado NO mejora al champion ({rep.get('reason_federated', '')}).")

    st.divider()
    st.subheader("Benchmark federado vs centralizado")
    if st.button("📊 Comparar federado vs centralizado"):
        try:
            rep = comparar_federado()
        except requests.HTTPError as err:
            code = err.response.status_code if err.response is not None else "?"
            if code == 409:
                st.warning("No hay modelo federado activo todavía: corré el DAG "
                           "`fraud_federate_pipeline` y luego 🔄 Recargar modelos.")
            else:
                st.error(f"No se pudo comparar (HTTP {code}): {err}")
        except requests.RequestException as err:
            st.error(f"No se pudo comparar: {err}")
        else:
            st.write(f"Test: **{rep.get('n_test', 0)}** transacciones · "
                     f"fraude {rep.get('fraude_test', 0):.3%}")
            tabla = {
                "centralizado (champion)": rep.get("centralizado", {}),
                "federado (FedAvg)": rep.get("federado", {}),
            }
            st.dataframe(pd.DataFrame(tabla).T, use_container_width=True)
            pct = rep.get("pct_pr_auc_recuperado")
            if pct is not None:
                st.success(
                    f"El MLP federado recupera el **{pct}%** del PR-AUC del modelo "
                    f"centralizado, sin que los bancos compartan sus transacciones."
                )

    st.divider()
    st.caption("Promové el modelo en sombra que prefieras a champion (o revertí al previo).")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Deploy challenger → champion"):
            try:
                st.success(f"Deploy: {deploy('promote', source='challenger')}")
            except requests.RequestException as err:
                st.error(f"Deploy falló: {err}")
    with col2:
        if st.button("Deploy federado → champion"):
            try:
                st.success(f"Deploy: {deploy('promote', source='federated')}")
            except requests.RequestException as err:
                st.error(f"Deploy falló: {err}")
    with col3:
        if st.button("Rollback al champion previo"):
            try:
                st.warning(f"Rollback: {deploy('rollback')}")
            except requests.RequestException as err:
                st.error(f"Rollback falló: {err}")


def vista_modelo() -> None:
    st.header("Modelo")
    st.caption("Champion servido actualmente por la API REST.")
    try:
        info = obtener_model_info()
    except requests.RequestException as err:
        st.error(f"No se pudo contactar la API REST en {REST_URL}. Detalle: {err}")
        return

    store_version = info.get("version")
    serving_version = info.get("serving_champion_version")

    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre", str(info.get("name", "desconocido")))
    col2.metric("Versión (store REST)", str(store_version if store_version is not None else "desconocida"))
    col3.metric(
        "Champion que decide (gRPC)",
        str(serving_version) if serving_version is not None else "n/d",
    )

    if serving_version is not None and str(serving_version) != str(store_version):
        st.warning(
            f"REST y el núcleo gRPC reportan versiones distintas: "
            f"store REST = {store_version} · champion que decide (gRPC) = {serving_version}."
        )

    metricas = info.get("metrics")
    if metricas:
        st.subheader("Métricas")
        st.dataframe(pd.DataFrame([metricas]).T.rename(columns={0: "valor"}), use_container_width=True)

    # --- Integridad (SAIF capa 7) ---
    st.subheader("Seguridad del modelo")
    sha = info.get("sha256")
    if sha:
        st.caption("Hash SHA-256 de integridad del artefacto servido (verifica que no fue manipulado):")
        st.code(sha, language=None)

    with st.expander("Respuesta completa (raw)"):
        st.json(info)


def vista_seguridad() -> None:
    st.header("Seguridad (framework SAIF)")
    st.caption("Controles de seguridad del sistema, visibles desde la UI. El rol activo "
               "se elige en el panel lateral (izquierda).")

    # --- Capa 1: rol activo + prueba de RBAC ---
    st.subheader("Autenticación y roles (RBAC)")
    rol = st.session_state.get("rol_label", "Admin (todo)")
    st.write(f"Rol activo ahora mismo: **{rol}**. Cambialo en el panel lateral y probá "
             "una acción de MLOps: con rol *Analista* el sistema la rechaza.")
    if st.button("Probar acción MLOps con el rol actual (evaluar)"):
        try:
            evaluar()
            st.success("✅ Permitido: el rol actual puede ejecutar acciones de MLOps.")
        except requests.HTTPError as err:
            code = err.response.status_code if err.response is not None else "?"
            if code == 403:
                st.error("⛔ 403 — el rol actual NO está autorizado para MLOps (así debe ser "
                         "para el analista). Cambiá a MLOps o Admin en el panel lateral.")
            else:
                st.warning(f"HTTP {code}: {err}")
        except requests.RequestException as err:
            st.error(f"No se pudo contactar la API: {err}")

    st.divider()

    # --- Integridad (hash) ---
    st.subheader("Integridad del modelo (hash SHA-256)")
    try:
        info = obtener_model_info()
        sha = info.get("sha256")
        if sha:
            st.caption("Hash del artefacto servido (verifica que no fue manipulado):")
            st.code(sha, language=None)
    except requests.RequestException as err:
        st.caption(f"No disponible: {err}")

    st.divider()

    # --- Log de auditoría ---
    st.subheader("Log de auditoría")
    st.caption("Cada acción (predicción, revisión, deploy, etc.) queda registrada: quién "
               "(rol + token enmascarado), qué y cuándo.")
    if st.button("🔄 Actualizar auditoría"):
        st.rerun()
    try:
        eventos = obtener_auditoria(limit=50)
        if eventos:
            st.dataframe(pd.DataFrame(eventos), use_container_width=True, hide_index=True)
        else:
            st.info("Todavía no hay eventos auditados. Hacé una predicción o una acción de MLOps.")
    except requests.HTTPError as err:
        code = err.response.status_code if err.response is not None else "?"
        st.warning(f"No se pudo leer la auditoría (HTTP {code}) — ¿rol sin permiso?")
    except requests.RequestException as err:
        st.error(f"No se pudo leer la auditoría: {err}")

    st.divider()

    # --- Capas que se muestran en otro lado ---
    st.subheader("Otros controles")
    st.markdown(
        "- **Validación (Pydantic):** el formulario de *Scoring manual* sólo deja "
        "elegir categorías/valores válidos — una entrada inválida se rechaza con `422`.\n"
        "- **Rate limiting:** se ve mejor con la **UI de Locust** "
        "(`http://localhost:8089`): al generar muchas peticiones aparecen respuestas `429`.\n"
        "- **gRPC:** el scoring corre detrás del núcleo gRPC, que también exige "
        "API key (ver pestaña *Modelo* → \"Champion que decide (gRPC)\").\n"
        "- **Secretos:** las claves viven en `.env` (gitignored), no en el repo; "
        "esto se verifica en los archivos, no en la UI.\n"
        "- **Entrenamiento federado:** defensas contra envenenamiento "
        "(clipping + agregación robusta + privacidad diferencial) — ver el script "
        "`demo_poisoning.py`."
    )


# --------------------------------------------------------------------------- #
# Layout principal
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Analista de Fraude", page_icon="🔎", layout="wide")
    st.title("Interfaz del analista de fraude")

    with st.sidebar:
        st.subheader("Rol (seguridad)")
        rol_label = st.selectbox(
            "Autenticarse como", list(ROLE_TOKENS.keys()),
            help="Elegí el rol con el que la UI se autentica. El analista no puede MLOps.",
        )
        st.session_state["api_key"] = ROLE_TOKENS[rol_label]
        st.session_state["rol_label"] = rol_label
        key = ROLE_TOKENS[rol_label]
        st.caption(f"Token activo: `{'*' * max(0, len(key) - 4)}{key[-4:]}`")

        st.subheader("Configuración")
        st.write(f"REST_URL: `{REST_URL}`")
        try:
            salud = requests.get(f"{REST_URL}/health", timeout=HTTP_TIMEOUT)
            if salud.ok:
                st.success("API REST: disponible")
            else:
                st.warning(f"API REST respondió {salud.status_code}")
        except requests.RequestException:
            st.error("API REST: no disponible")

    tabs = st.tabs(
        ["Scoring manual", "Cola de revisión", "Transacciones realizadas",
         "MLOps", "Modelo", "Seguridad"]
    )
    with tabs[0]:
        vista_scoring_manual()
    with tabs[1]:
        vista_cola_revision()
    with tabs[2]:
        vista_transacciones()
    with tabs[3]:
        vista_mlops()
    with tabs[4]:
        vista_modelo()
    with tabs[5]:
        vista_seguridad()


if __name__ == "__main__":
    main()
