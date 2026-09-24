"""Interfaz del analista de fraude (Streamlit).

Este servicio es la cara humana del sistema de detección de fraude con tarjeta.
Consume la API REST (FastAPI) para scorear transacciones y consultar el modelo,
y permite al analista revisar alertas y **etiquetar** cada caso como fraude o no,
generando los labels reales que luego alimentan el reentrenamiento.

Vistas:
    - Scoring manual: formulario -> POST /v1/predict.
    - Bandeja de alertas: tabla de sospechosas + botones Confirmar/Descartar.
    - Modelo: GET /v1/model-info (versión y métricas).

Variables de entorno:
    REST_URL  URL base de la API REST (default: http://rest:8080)
    API_KEY   Token para el header X-API-KEY (default: token-secreto-123)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #

# La URL de la REST y el token se leen del entorno para poder cambiarlos
# entre local y Docker sin tocar el código.
REST_URL = os.getenv("REST_URL", "http://rest:8080").rstrip("/")
API_KEY = os.getenv("API_KEY", "token-secreto-123")

# Timeout corto para que la UI no se cuelgue si la REST no responde.
HTTP_TIMEOUT = 5

# Directorio de datos local (placeholder del sistema real).
DATA_DIR = Path(__file__).parent / "data"
ALERTS_CSV = DATA_DIR / "alerts.csv"
LABELS_CSV = DATA_DIR / "labels.csv"

# Features del modelo (deben coincidir con el contrato de la API REST).
NUMERIC_FEATURES = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "hour", "age"]
CATEGORICAL_FEATURES = ["category", "gender"]

# Valores válidos de las categóricas (mismos enums que la API).
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
    """Header de autenticación esperado por la API REST."""
    return {"X-API-KEY": API_KEY}


def predecir(transaccion: dict) -> dict:
    """Llama a POST /v1/predict y devuelve la respuesta como dict.

    Lanza requests.RequestException si la REST no está disponible o
    responde con un error HTTP.
    """
    resp = requests.post(
        f"{REST_URL}/v1/predict",
        json=transaccion,
        headers=_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def obtener_model_info() -> dict:
    """Llama a GET /v1/model-info y devuelve la respuesta como dict."""
    resp = requests.get(
        f"{REST_URL}/v1/model-info",
        headers=_headers(),
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Helpers de alertas y labels (placeholder de los stores del sistema real)
# --------------------------------------------------------------------------- #

def _ejemplos_alertas() -> pd.DataFrame:
    """Genera alertas de ejemplo cuando no existe data/alerts.csv.

    En el sistema real estas alertas llegarían desde el topic `fraud-alerts`.
    """
    filas = [
        {
            "transaction_id": "tx-1001", "amt": 980.50, "category": "shopping_net",
            "gender": "M", "city_pop": 12000, "lat": 40.71, "long": -74.00,
            "merch_lat": 34.05, "merch_long": -118.24, "hour": 3, "age": 27,
            "probability": 0.91,
        },
        {
            "transaction_id": "tx-1002", "amt": 1250.00, "category": "misc_net",
            "gender": "F", "city_pop": 5000, "lat": 41.88, "long": -87.63,
            "merch_lat": 25.76, "merch_long": -80.19, "hour": 2, "age": 44,
            "probability": 0.87,
        },
        {
            "transaction_id": "tx-1003", "amt": 45.20, "category": "grocery_pos",
            "gender": "F", "city_pop": 800000, "lat": 34.05, "long": -118.24,
            "merch_lat": 34.06, "merch_long": -118.25, "hour": 14, "age": 61,
            "probability": 0.62,
        },
    ]
    return pd.DataFrame(filas)


def cargar_alertas() -> pd.DataFrame:
    """Lee las alertas desde data/alerts.csv o genera ejemplos si no existe.

    En producción, la bandeja se alimentaría del topic `fraud-alerts` del broker.
    """
    if ALERTS_CSV.exists():
        return pd.read_csv(ALERTS_CSV)
    return _ejemplos_alertas()


def cargar_labels() -> pd.DataFrame:
    """Lee los labels ya registrados por el analista (o un DataFrame vacío)."""
    columnas = ["transaction_id", "label", "analyst_ts"]
    if LABELS_CSV.exists():
        return pd.read_csv(LABELS_CSV)
    return pd.DataFrame(columns=columnas)


def guardar_label(transaction_id: str, label: int) -> None:
    """Persiste el veredicto del analista en data/labels.csv.

    label: 1 = fraude confirmado, 0 = descartado.
    En el sistema real esto alimentaría el store etiquetado (labeled store)
    usado para reentrenar el modelo con feedback humano.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    labels = cargar_labels()

    # Si ya existe un label para esa transacción, lo reemplazamos (última decisión).
    labels = labels[labels["transaction_id"] != transaction_id]

    nueva = pd.DataFrame([{
        "transaction_id": transaction_id,
        "label": label,
        "analyst_ts": datetime.now(timezone.utc).isoformat(),
    }])
    labels = pd.concat([labels, nueva], ignore_index=True)
    labels.to_csv(LABELS_CSV, index=False)


# --------------------------------------------------------------------------- #
# Vistas de la UI
# --------------------------------------------------------------------------- #

def vista_scoring_manual() -> None:
    """Formulario de scoring: arma una transacción y la envía a /v1/predict."""
    st.header("Scoring manual")
    st.caption("Cargá las features de una transacción y consultá al modelo en vivo.")

    with st.form("form_scoring"):
        col1, col2 = st.columns(2)

        with col1:
            amt = st.number_input("Monto (amt)", min_value=0.01, value=120.50, step=1.0)
            category = st.selectbox("Rubro (category)", CATEGORIES, index=CATEGORIES.index("grocery_pos"))
            gender = st.selectbox("Género (gender)", GENDERS)
            city_pop = st.number_input("Población ciudad (city_pop)", min_value=0, value=50000, step=1000)
            hour = st.slider("Hora del día (hour)", min_value=0, max_value=23, value=2)

        with col2:
            age = st.slider("Edad del titular (age)", min_value=0, max_value=120, value=35)
            lat = st.number_input("Latitud titular (lat)", min_value=-90.0, max_value=90.0, value=40.1)
            long = st.number_input("Longitud titular (long)", min_value=-180.0, max_value=180.0, value=-74.5)
            merch_lat = st.number_input("Latitud comercio (merch_lat)", min_value=-90.0, max_value=90.0, value=40.3)
            merch_long = st.number_input("Longitud comercio (merch_long)", min_value=-180.0, max_value=180.0, value=-74.2)

        enviado = st.form_submit_button("Predecir")

    if not enviado:
        return

    transaccion = {
        "amt": amt, "category": category, "gender": gender, "city_pop": int(city_pop),
        "lat": lat, "long": long, "merch_lat": merch_lat, "merch_long": merch_long,
        "hour": int(hour), "age": int(age),
    }

    try:
        resultado = predecir(transaccion)
    except requests.RequestException as err:
        st.error(f"No se pudo contactar la API REST en {REST_URL}. Detalle: {err}")
        return

    probabilidad = resultado.get("probability", 0.0)
    es_fraude = resultado.get("is_fraud", False)

    st.metric("Probabilidad de fraude", f"{probabilidad:.2%}")
    if es_fraude:
        st.error("Veredicto del modelo: FRAUDE")
    else:
        st.success("Veredicto del modelo: legítima")
    st.caption(f"Modelo: {resultado.get('model_version', 'desconocido')}")


def vista_bandeja_alertas() -> None:
    """Bandeja de alertas: revisar sospechosas y registrar el label real."""
    st.header("Bandeja de alertas")
    st.caption(
        "Transacciones marcadas como sospechosas. En el sistema real llegarían "
        "del topic `fraud-alerts`. Confirmá o descartá para generar los labels reales."
    )

    alertas = cargar_alertas()
    if alertas.empty:
        st.info("No hay alertas pendientes.")
        return

    labels = cargar_labels()
    ya_etiquetadas = set(labels["transaction_id"].astype(str)) if not labels.empty else set()

    # Tabla resumen de las alertas.
    st.dataframe(alertas, use_container_width=True, hide_index=True)

    st.subheader("Revisión")
    for _, alerta in alertas.iterrows():
        tx_id = str(alerta["transaction_id"])
        prob = alerta.get("probability", None)

        etiqueta_previa = ""
        if tx_id in ya_etiquetadas:
            valor = labels.loc[labels["transaction_id"].astype(str) == tx_id, "label"].iloc[-1]
            etiqueta_previa = " (fraude)" if int(valor) == 1 else " (descartada)"

        prob_txt = f" — prob. {float(prob):.2%}" if prob is not None else ""
        col_info, col_ok, col_no = st.columns([4, 1, 1])
        with col_info:
            st.write(f"**{tx_id}**{prob_txt} — monto {alerta.get('amt', '?')} / {alerta.get('category', '?')}{etiqueta_previa}")
        with col_ok:
            if st.button("Confirmar fraude", key=f"ok_{tx_id}"):
                guardar_label(tx_id, 1)
                st.success(f"{tx_id} etiquetada como fraude.")
                st.rerun()
        with col_no:
            if st.button("Descartar", key=f"no_{tx_id}"):
                guardar_label(tx_id, 0)
                st.info(f"{tx_id} descartada.")
                st.rerun()

    # Historial de labels ya registrados.
    labels = cargar_labels()
    if not labels.empty:
        with st.expander("Labels registrados (store etiquetado)"):
            st.caption(f"Se persisten en `{LABELS_CSV.name}`. En producción alimentan el store etiquetado para reentrenar.")
            st.dataframe(labels, use_container_width=True, hide_index=True)


def vista_modelo() -> None:
    """Muestra la información del modelo desde /v1/model-info."""
    st.header("Modelo")
    st.caption("Información del modelo servido actualmente por la API REST.")

    try:
        info = obtener_model_info()
    except requests.RequestException as err:
        st.error(f"No se pudo contactar la API REST en {REST_URL}. Detalle: {err}")
        return

    col1, col2 = st.columns(2)
    col1.metric("Nombre", str(info.get("name", "desconocido")))
    col2.metric("Versión", str(info.get("version", "desconocida")))

    metricas = info.get("metrics")
    if metricas:
        st.subheader("Métricas")
        st.dataframe(
            pd.DataFrame([metricas]).T.rename(columns={0: "valor"}),
            use_container_width=True,
        )

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
        # No mostramos el token completo por seguridad.
        st.write(f"API_KEY: `{'*' * max(0, len(API_KEY) - 4)}{API_KEY[-4:]}`")

        # Chequeo rápido de salud de la REST.
        try:
            salud = requests.get(f"{REST_URL}/health", timeout=HTTP_TIMEOUT)
            if salud.ok:
                st.success("API REST: disponible")
            else:
                st.warning(f"API REST respondió {salud.status_code}")
        except requests.RequestException:
            st.error("API REST: no disponible")

    tab_scoring, tab_alertas, tab_modelo = st.tabs(
        ["Scoring manual", "Bandeja de alertas", "Modelo"]
    )
    with tab_scoring:
        vista_scoring_manual()
    with tab_alertas:
        vista_bandeja_alertas()
    with tab_modelo:
        vista_modelo()


if __name__ == "__main__":
    main()
