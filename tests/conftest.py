"""Fixtures compartidas de los tests.

Aísla el estado del serving para no tocar el repo:
- BD SQLite en un directorio temporal (`FRAUD_DB`).
- Umbral de revisión en 0.0 para que las transacciones vayan a `PENDING` y se pueda
  ejercitar el flujo del analista (reviews / decision / report-fraud).
- Un challenger local (copia del champion) para ejercitar el scoring en sombra.

Todo se setea ANTES de importar la app, y se limpia al terminar la sesión.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="fraud-tests-"))
os.environ["FRAUD_DB"] = str(_TMP / "test_fraud.db")
os.environ["FRAUD_REVIEW_THRESHOLD"] = "0.0"

_MODELS = Path(__file__).resolve().parents[1] / "models"
_CHAMPION = _MODELS / "model.joblib"
_CHALLENGER = _MODELS / "challenger.joblib"
_PREVIOUS = _MODELS / "previous.joblib"

# Activamos un challenger en sombra (copia del champion) si hay modelo disponible.
_created_challenger = False
if _CHAMPION.exists() and not _CHALLENGER.exists():
    shutil.copyfile(_CHAMPION, _CHALLENGER)
    _created_challenger = True


@pytest.fixture(scope="session", autouse=True)
def _cleanup_artifacts():
    yield
    # Limpieza: sacamos los artefactos que crearon los tests (no el champion real).
    for f in (_CHALLENGER, _PREVIOUS):
        if f.exists():
            f.unlink()
    shutil.rmtree(_TMP, ignore_errors=True)
