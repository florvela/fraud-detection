"""Adapter que hace que el MLP federado se vea como un pipeline sklearn.

El camino federado (``services/federated``) produce un MLP de PyTorch cuyos pesos
globales se guardan en ``models/mlp_federado.npz``. El serving (champion/challenger),
en cambio, espera artefactos con la interfaz sklearn: ``pipeline.predict_proba`` /
``pipeline.predict`` sobre un ``DataFrame`` con las features crudas. Son dos mundos
distintos.

Este módulo los une con :class:`FederatedPipeline`: envuelve (preprocesador + pesos
del MLP) detrás de esa misma interfaz. Así, un federado empaquetado como

    {"pipeline": FederatedPipeline(...), "feature_order": [...], "version", "metrics"}

es **indistinguible de un champion/challenger** para ``ModelStore``, ``Scorer``,
``deploy`` y el núcleo gRPC: se puede correr en sombra y promover a champion sin
tocar el resto del código.

Decisiones de diseño
--------------------
* **Pesos como numpy, torch diferido**: el pipeline guarda los pesos como lista de
  ``np.ndarray`` (no un ``nn.Module``), así el ``joblib`` es portable y no depende de
  la versión de torch al deserializar. El modelo de torch se construye *lazy* en la
  primera predicción (y se cachea).
* **Contrato de features duplicado a propósito**: este módulo vive en el paquete
  ``fraud`` (instalado en las imágenes de serving) y NO puede importar de
  ``services/federated`` (que corre como otra imagen). Por eso replica el contrato
  de ``services/federated/model.py``. Si cambia uno, cambiar el otro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Contrato de features — DEBE coincidir con services/federated/model.py
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "hour", "age"]
CATEGORICAL_FEATURES = ["category", "gender"]

CATEGORY_VOCAB = {
    "category": [
        "entertainment", "food_dining", "gas_transport", "grocery_net",
        "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
        "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
    ],
    "gender": ["F", "M"],
}

# Orden canónico de las features crudas (igual que el champion), para que el serving
# arme el DataFrame con todas las columnas que el preprocesador necesita por nombre.
FEATURE_ORDER = [
    "amt", "category", "gender", "city_pop", "lat",
    "long", "merch_lat", "merch_long", "hour", "age",
]


class _Preprocessor:
    """Estandariza numéricas (mean/std fijos) + one-hot de categóricas (vocab fijo)."""

    def __init__(self, means: dict[str, float], stds: dict[str, float]) -> None:
        self.means = means
        self.stds = stds
        self.feature_names: list[str] = list(NUMERIC_FEATURES)
        for col in CATEGORICAL_FEATURES:
            for val in CATEGORY_VOCAB[col]:
                self.feature_names.append(f"{col}={val}")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        bloques: list[np.ndarray] = []
        num = np.empty((len(df), len(NUMERIC_FEATURES)), dtype=np.float32)
        for j, col in enumerate(NUMERIC_FEATURES):
            std = self.stds[col] or 1.0
            num[:, j] = (df[col].to_numpy(dtype=np.float32) - self.means[col]) / std
        bloques.append(num)
        for col in CATEGORICAL_FEATURES:
            vals = df[col].astype(str).to_numpy()
            oh = np.zeros((len(df), len(CATEGORY_VOCAB[col])), dtype=np.float32)
            for k, cat in enumerate(CATEGORY_VOCAB[col]):
                oh[:, k] = (vals == cat).astype(np.float32)
            bloques.append(oh)
        return np.concatenate(bloques, axis=1).astype(np.float32)


def _build_torch_mlp(input_dim: int):
    """Construye el MLP de torch (import diferido). Arquitectura == model.py."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


class FederatedPipeline:
    """Envuelve (preprocesador + pesos del MLP) con interfaz sklearn.

    Expone ``predict_proba`` / ``predict`` sobre un ``DataFrame`` de features crudas,
    igual que el pipeline del champion XGBoost. Es picklable: solo guarda numpy + dicts;
    torch se importa y el modelo se arma en la primera inferencia.
    """

    def __init__(
        self,
        means: dict[str, float],
        stds: dict[str, float],
        weights: list[np.ndarray],
        threshold: float = 0.5,
    ) -> None:
        self.means = means
        self.stds = stds
        # Guardamos copias en float32 para un pickle estable y chico.
        self.weights = [np.asarray(w, dtype=np.float32) for w in weights]
        self.threshold = threshold
        # Estado NO serializado: se reconstruye lazy tras deserializar.
        self._pre: _Preprocessor | None = None
        self._model = None  # torch.nn.Module

    # -- pickling: nunca persistir el modelo de torch ni el preprocesador --
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_pre"] = None
        state["_model"] = None
        return state

    def _ensure_ready(self) -> None:
        import torch

        if self._pre is None:
            self._pre = _Preprocessor(self.means, self.stds)
        if self._model is None:
            model = _build_torch_mlp(self._pre.n_features)
            # Cargar los pesos en el orden del state_dict (== orden del .npz).
            state_dict = model.state_dict()
            new_state = {
                k: torch.tensor(np.asarray(w))
                for k, w in zip(state_dict.keys(), self.weights)
            }
            model.load_state_dict(new_state, strict=True)
            model.eval()
            self._model = model

    def _probs(self, X: pd.DataFrame) -> np.ndarray:
        import torch

        self._ensure_ready()
        assert self._pre is not None
        mat = self._pre.transform(X[FEATURE_ORDER] if set(FEATURE_ORDER).issubset(X.columns) else X)
        with torch.no_grad():
            logits = self._model(torch.from_numpy(mat))
            return torch.sigmoid(logits).cpu().numpy().ravel()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve (n, 2) con [P(no-fraude), P(fraude)] (convención sklearn)."""
        p = self._probs(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Clase 0/1 aplicando el umbral sobre P(fraude)."""
        return (self._probs(X) >= self.threshold).astype(int)


def load_weights_npz(path) -> list[np.ndarray]:
    """Lee los pesos del modelo global federado desde un .npz (en orden)."""
    data = np.load(path)
    return [data[k] for k in data.files]


def build_pipeline(spec: dict, weights: list[np.ndarray], threshold: float = 0.5) -> FederatedPipeline:
    """Construye el :class:`FederatedPipeline` desde el spec (means/stds) + pesos."""
    return FederatedPipeline(
        means=spec["means"], stds=spec["stds"], weights=weights, threshold=threshold
    )
