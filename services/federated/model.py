"""Modelo MLP tabular y preprocesamiento compartido para el camino federado.

Este módulo concentra el "contrato" del modelo federado para que servidor y
clientes usen exactamente la misma representación de features y la misma
arquitectura de red. Es la pieza clave para que FedAvg tenga sentido: solo se
pueden promediar pesos si todos los bancos entrenan la MISMA red sobre el MISMO
espacio de features.

Decisiones de diseño
--------------------
* **MLP y no XGBoost**: el modelo champion (``models/model.joblib``) es un
  XGBoost. Los árboles de gradient boosting no tienen un vector de "pesos"
  homogéneo que se pueda promediar entre clientes (FedAvg promedia tensores).
  Por eso el camino federado usa un MLP paramétrico: sus pesos SÍ se promedian.
* **Preprocesamiento fijo y consistente**: one-hot de categóricas con
  vocabulario fijo (mismas columnas en todos los bancos) + estandarización de
  numéricas con medias/desvíos acordados. Estas constantes de normalización se
  tratan como METADATA PÚBLICA del esquema (no son transacciones): se calculan
  una sola vez sobre el split público de train y se guardan en
  ``preprocess_spec.json``. Los bancos NO comparten sus filas, solo comparten
  este contrato de normalización y, durante el entrenamiento, los pesos del MLP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Contrato de features (idéntico al del resto del repo, ver services/streaming)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "hour", "age"]
CATEGORICAL_FEATURES = ["category", "gender"]
TARGET = "is_fraud"

# Vocabulario FIJO de las categóricas. Fijarlo garantiza que el one-hot produzca
# exactamente las mismas columnas (en el mismo orden) en Banco A y Banco B,
# incluso si algún silo no tuviera todas las categorías. Sin esto, los vectores
# de pesos tendrían dimensiones distintas y FedAvg fallaría.
CATEGORY_VOCAB = {
    "category": [
        "entertainment", "food_dining", "gas_transport", "grocery_net",
        "grocery_pos", "health_fitness", "home", "kids_pets", "misc_net",
        "misc_pos", "personal_care", "shopping_net", "shopping_pos", "travel",
    ],
    "gender": ["F", "M"],
}

# Rutas por defecto (relativas a la raíz del repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = Path(__file__).resolve().parent / "preprocess_spec.json"
DEFAULT_TRAIN_PATH = _REPO_ROOT / "data" / "processed" / "train.parquet"


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
class Preprocessor:
    """Transforma un DataFrame de features al vector numérico que consume el MLP.

    Aplica:
    * Estandarización ``(x - mean) / std`` de las numéricas con constantes fijas.
    * One-hot de las categóricas usando ``CATEGORY_VOCAB`` (columnas fijas).

    Las constantes (mean/std) provienen de un "spec" acordado y público; no se
    recalculan por silo para que la representación sea idéntica entre bancos.
    """

    def __init__(self, means: dict[str, float], stds: dict[str, float]) -> None:
        self.means = means
        self.stds = stds
        # Nombres de columnas de salida, en orden estable y reproducible.
        self.feature_names: list[str] = list(NUMERIC_FEATURES)
        for col in CATEGORICAL_FEATURES:
            for val in CATEGORY_VOCAB[col]:
                self.feature_names.append(f"{col}={val}")

    @property
    def n_features(self) -> int:
        """Dimensión del vector de entrada del MLP."""
        return len(self.feature_names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Convierte features crudas en una matriz float32 lista para el MLP."""
        bloques: list[np.ndarray] = []

        # 1) Numéricas estandarizadas.
        num = np.empty((len(df), len(NUMERIC_FEATURES)), dtype=np.float32)
        for j, col in enumerate(NUMERIC_FEATURES):
            std = self.stds[col] or 1.0  # evita división por cero
            num[:, j] = (df[col].to_numpy(dtype=np.float32) - self.means[col]) / std
        bloques.append(num)

        # 2) Categóricas one-hot con vocabulario fijo.
        for col in CATEGORICAL_FEATURES:
            vals = df[col].astype(str).to_numpy()
            oh = np.zeros((len(df), len(CATEGORY_VOCAB[col])), dtype=np.float32)
            for k, cat in enumerate(CATEGORY_VOCAB[col]):
                oh[:, k] = (vals == cat).astype(np.float32)
            bloques.append(oh)

        return np.concatenate(bloques, axis=1).astype(np.float32)


def build_preprocess_spec(
    train_path: Path = DEFAULT_TRAIN_PATH,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> dict:
    """Calcula medias/desvíos de las numéricas sobre el train público y los guarda.

    Se ejecuta una sola vez (idempotente). El resultado es metadata pública del
    esquema, no transacciones de ningún banco.
    """
    df = pd.read_parquet(train_path, columns=NUMERIC_FEATURES)
    means = {c: float(df[c].mean()) for c in NUMERIC_FEATURES}
    stds = {c: float(df[c].std(ddof=0)) for c in NUMERIC_FEATURES}
    spec = {"means": means, "stds": stds}
    spec_path.write_text(json.dumps(spec, indent=2))
    logger.info("Spec de preprocesamiento guardado en {}", spec_path)
    return spec


def load_preprocessor(spec_path: Path = DEFAULT_SPEC_PATH) -> Preprocessor:
    """Carga el ``Preprocessor`` desde el spec; lo construye si no existe."""
    if not spec_path.exists():
        logger.warning("No existe {}; se construye desde el train público.", spec_path)
        build_preprocess_spec(spec_path=spec_path)
    spec = json.loads(spec_path.read_text())
    return Preprocessor(means=spec["means"], stds=spec["stds"])


# ---------------------------------------------------------------------------
# Arquitectura del MLP (PyTorch)
# ---------------------------------------------------------------------------
# La import de torch se hace acá para que el módulo pueda importarse (y validar
# el preprocesamiento) aún sin torch instalado. El MLP solo se necesita al
# entrenar/evaluar.
try:  # pragma: no cover - depende del entorno
    import torch
    import torch.nn as nn

    _TORCH_OK = True
except Exception:  # pragma: no cover
    _TORCH_OK = False


if _TORCH_OK:

    class MLP(nn.Module):
        """MLP tabular chico: entrada -> 32 -> 16 -> 1 (logit).

        Devuelve el logit sin sigmoide para poder usar ``BCEWithLogitsLoss``
        (numéricamente estable y con soporte de ``pos_weight`` para el
        desbalance). La probabilidad se obtiene aplicando sigmoide afuera.
        """

        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
            return self.net(x)


def get_weights(model: "MLP") -> list[np.ndarray]:  # noqa: F821
    """Extrae los pesos del modelo como lista de ndarrays (formato Flower)."""
    return [p.detach().cpu().numpy() for p in model.state_dict().values()]


def set_weights(model: "MLP", weights: list[np.ndarray]) -> None:  # noqa: F821
    """Carga en el modelo una lista de ndarrays (formato Flower)."""
    state_dict = model.state_dict()
    new_state = {k: torch.tensor(w) for k, w in zip(state_dict.keys(), weights)}
    model.load_state_dict(new_state, strict=True)


def build_model(input_dim: int) -> "MLP":  # noqa: F821
    """Fábrica del modelo (falla con mensaje claro si no hay torch)."""
    if not _TORCH_OK:
        raise RuntimeError(
            "PyTorch no está instalado. Instalá las dependencias de "
            "requirements.txt para entrenar/evaluar el MLP federado."
        )
    return MLP(input_dim)
