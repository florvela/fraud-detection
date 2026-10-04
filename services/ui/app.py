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
API_KEY = os.getenv("API_KEY", "token-secreto-123")
HTTP_TIMEOUT = 5

CATEGORIES = [
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos",
    "personal_care", "shopping_net", "shopping_pos", "travel",
]
GENDERS = ["F", "M"]


# --------------------------------------------------------------------------- #
# Helpers de la API REST
# --------------------------------------------------------------------------- #

def _headers() -> dict:
    return {"X-API-KEY": API_KEY}


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


def deploy(action: str) -> dict:
    resp = requests.post(
        f"{REST_URL}/v1/mlops/deploy", params={"action": action}, headers=_headers(), timeout=HTTP_TIMEOUT
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


# --------------------------------------------------------------------------- #
# Vistas
# --------------------------------------------------------------------------- #

def vista_scoring_manual() -> None:
    st.header("Scoring manual")
    st.caption("Cargá una transacción: el champion decide y el challenger (si hay) corre en sombra.")

    with st.form("form_scoring"):
        col1, col2 = st.columns(2)
        # Defaults = un fraude REAL que el champion malo (sin balanceo) DEJA PASAR
        # (~11%) pero el challenger bueno ATRAPA (~89%). Ideal para la demo.
        with col1:
            amt = st.number_input("Monto (amt)", min_value=0.01, value=925.94, step=1.0)
            category = st.selectbox("Rubro (category)", CATEGORIES, index=CATEGORIES.index("shopping_net"))
            gender = st.selectbox("Género (gender)", GENDERS, index=GENDERS.index("M"))
            city_pop = st.number_input("Población ciudad (city_pop)", min_value=0, value=4653, step=1000)
            hour = st.slider("Hora del día (hour)", 0, 23, 12)
        with col2:
            age = st.slider("Edad del titular (age)", 0, 120, 22)
            lat = st.number_input("Latitud titular (lat)", -90.0, 90.0, value=40.5046)
            long = st.number_input("Longitud titular (long)", -180.0, 180.0, value=-77.7186)
            merch_lat = st.number_input("Latitud comercio (merch_lat)", -90.0, 90.0, value=41.4437)
            merch_long = st.number_input("Longitud comercio (merch_long)", -180.0, 180.0, value=-78.3918)
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
        filas.append({
            "transaction_id": tx["transaction_id"],
            "monto": (tx.get("features") or {}).get("amt"),
            "prob_champion": champ.get("probability"),
            "prob_challenger": chal.get("probability"),
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
    st.header("MLOps — challenger / champion")
    st.caption("Evaluá el challenger contra la ground truth y, si mejora, desplegalo.")

    st.info("Tras correr el DAG (retrain), apretá **Recargar modelos** para que el "
            "serving tome el nuevo challenger sin reiniciar nada.")
    if st.button("🔄 Recargar modelos (tras retrain)"):
        try:
            r = recargar_modelos()
            ch = "sí" if r.get("has_challenger") else "no"
            st.success(f"Recargado. Champion: {r.get('champion_version')} · challenger activo: {ch}")
        except requests.RequestException as err:
            st.error(f"No se pudo recargar: {err}")

    st.divider()

    st.caption("Sembrá fraudes post-mortem para poblar la ground truth. Corré esto "
               "DESPUÉS de **Recargar modelos** (con el challenger activo) para que se "
               "registren las predicciones de ambos modelos y la evaluación tenga sentido.")
    if st.button("🌱 Sembrar 30 fraudes (post-mortem)"):
        try:
            r = sembrar_fraudes()
        except requests.RequestException as err:
            st.error(f"No se pudo sembrar: {err}")
        else:
            msg = (f"Sembrados: {r.get('seeded', 0)} · "
                   f"con challenger: {r.get('with_challenger', 0)}. {r.get('note', '')}")
            if r.get("seeded", 0) > 0 and r.get("with_challenger", 0) == 0:
                st.warning(msg)
            else:
                st.success(msg)

    st.divider()

    if st.button("Evaluar challenger vs champion"):
        try:
            rep = evaluar()
        except requests.RequestException as err:
            st.error(f"No se pudo evaluar. Detalle: {err}")
            return
        st.write(f"Transacciones con ground truth: **{rep.get('n_with_ground_truth', 0)}**")
        comp = {k: rep[k] for k in ("champion", "challenger") if rep.get(k)}
        if comp:
            st.dataframe(pd.DataFrame(comp).T, use_container_width=True)
        if rep.get("challenger_better"):
            st.success(f"El challenger mejora al champion ({rep.get('reason', '')}).")
        else:
            st.info(f"El challenger NO mejora al champion ({rep.get('reason', '')}).")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Deploy challenger → champion"):
            try:
                st.success(f"Deploy: {deploy('promote')}")
            except requests.RequestException as err:
                st.error(f"Deploy falló: {err}")
    with col2:
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
    with st.expander("Respuesta completa (raw)"):
        st.json(info)


# --------------------------------------------------------------------------- #
# Layout principal
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Analista de Fraude", page_icon="🔎", layout="wide")
    st.title("Interfaz del analista de fraude")

    with st.sidebar:
        st.subheader("Configuración")
        st.write(f"REST_URL: `{REST_URL}`")
        st.write(f"API_KEY: `{'*' * max(0, len(API_KEY) - 4)}{API_KEY[-4:]}`")
        try:
            salud = requests.get(f"{REST_URL}/health", timeout=HTTP_TIMEOUT)
            if salud.ok:
                st.success("API REST: disponible")
            else:
                st.warning(f"API REST respondió {salud.status_code}")
        except requests.RequestException:
            st.error("API REST: no disponible")

    tab_scoring, tab_revision, tab_realizadas, tab_mlops, tab_modelo = st.tabs(
        ["Scoring manual", "Cola de revisión", "Transacciones realizadas", "MLOps", "Modelo"]
    )
    with tab_scoring:
        vista_scoring_manual()
    with tab_revision:
        vista_cola_revision()
    with tab_realizadas:
        vista_transacciones()
    with tab_mlops:
        vista_mlops()
    with tab_modelo:
        vista_modelo()


if __name__ == "__main__":
    main()
