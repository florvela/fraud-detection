"""Carga el modelo entrenado desde models/ y se expone a la API"""

from __future__ import annotations

import joblib

from fraud.config import MODELS_DIR

MODEL_FILE = MODELS_DIR / "model.joblib"


class ModelStore:
    def __init__(self) -> None:
        self._artifact: dict | None = None

    @property
    def loaded(self) -> bool:
        return self._artifact is not None

    def load(self) -> bool:
        if not MODEL_FILE.exists():
            return False
        self._artifact = joblib.load(MODEL_FILE)
        return True

    @property
    def pipeline(self):
        return self._require()["pipeline"]

    @property
    def version(self) -> str:
        return self._require()["version"]

    @property
    def feature_order(self) -> list[str]:
        return self._require()["feature_order"]

    @property
    def metrics(self) -> dict:
        return self._require()["metrics"]

    @property
    def info(self) -> dict:
        return {k: v for k, v in self._require().items() if k != "pipeline"}

    def _require(self) -> dict:
        if self._artifact is None:
            raise RuntimeError(f"Modelo no cargado, se esperaba {MODEL_FILE}")
        return self._artifact
