"""Scoring con champion + challenger en sombra (núcleo del UC1).

El **champion** es el único que decide: su probabilidad contra un umbral define si
la transacción se aprueba (`OK`) o se retiene a revisión (`PENDING`). El
**challenger**, si está cargado, recibe exactamente las mismas features y deja su
predicción registrada, pero **no participa del veredicto**. Ambas predicciones se
persisten en el store para poder evaluar después al challenger contra la ground truth.

Decisión (igual que el diagrama de secuencia UC1):
    probability_champion ≤ umbral  -> APRUEBA           (status OK)
    probability_champion > umbral   -> EN REVISIÓN       (status PENDING)
La resolución final aprobado/rechazado la hace el analista (UC2).
"""

from __future__ import annotations

import os

import pandas as pd

from fraud.api import grpc_client
from fraud.api.model_loader import ModelStore, make_challenger, make_champion
from fraud.api.store import STATUS_OK, STATUS_PENDING, TransactionStore, get_store

# Umbral de retención: por encima, la tx se manda a revisión del analista.
REVIEW_THRESHOLD = float(os.getenv("FRAUD_REVIEW_THRESHOLD", "0.5"))

DECISION_APPROVE = "approve"   # aprobada automáticamente
DECISION_REVIEW = "review"     # retenida, va a la cola de revisión


def _predict_with(store: ModelStore, row: dict) -> dict:
    """Puntúa una fila con el pipeline de un ModelStore ya cargado."""
    X = pd.DataFrame([row], columns=store.feature_order)
    probability = float(store.pipeline.predict_proba(X)[0][1])
    is_fraud = bool(store.pipeline.predict(X)[0])
    return {
        "is_fraud": is_fraud,
        "probability": round(probability, 4),
        "model_version": store.version,
    }


class Scorer:
    """Orquesta champion (decide) + challenger (sombra) y persiste el resultado."""

    def __init__(
        self,
        champion: ModelStore | None = None,
        challenger: ModelStore | None = None,
        store: TransactionStore | None = None,
    ) -> None:
        self.champion = champion if champion is not None else make_champion()
        if not self.champion.loaded:
            self.champion.load()
        # El challenger es opcional: si no hay artefacto, queda en None (bloque `opt`).
        if challenger is not None:
            self.challenger = challenger
        else:
            cand = make_challenger()
            self.challenger = cand if cand.load() else None
        self.store = store if store is not None else get_store()

    @property
    def has_challenger(self) -> bool:
        return self.challenger is not None and self.challenger.loaded

    def reload(self) -> None:
        """Recarga ambos modelos (p.ej. tras un deploy que promovió un challenger)."""
        self.champion.reload()
        cand = make_challenger()
        self.challenger = cand if cand.load() else None

    def score(self, row: dict, transaction_id: str | None = None) -> dict:
        """Puntúa una transacción, decide con el champion y registra todo.

        Si `GRPC_HOST` está configurado, el score del champion se delega al núcleo
        gRPC (un solo motor de inferencia); si no, se usa el champion in-process.
        El challenger siempre se evalúa in-process (en sombra).
        """
        # --- champion (decide) ---
        if grpc_client.delegates_to_grpc():
            is_fraud, probability, version = grpc_client.predict(row)
            champion = {
                "is_fraud": bool(is_fraud),
                "probability": round(float(probability), 4),
                "model_version": version,
            }
        else:
            champion = _predict_with(self.champion, row)

        # --- challenger (en sombra, no decide) ---
        challenger = _predict_with(self.challenger, row) if self.has_challenger else None

        # --- decisión del champion contra el umbral ---
        if champion["probability"] > REVIEW_THRESHOLD:
            decision, status = DECISION_REVIEW, STATUS_PENDING
        else:
            decision, status = DECISION_APPROVE, STATUS_OK

        tx_id = self.store.record_scoring(
            features=row,
            champion=champion,
            challenger=challenger,
            decision=decision,
            status=status,
            transaction_id=transaction_id,
        )

        return {
            "transaction_id": tx_id,
            "is_fraud": champion["is_fraud"],
            "probability": champion["probability"],
            "model_version": champion["model_version"],
            "decision": decision,
            "status": status,
            "challenger": challenger,
        }


# Instancia compartida del serving (lazy).
_scorer: Scorer | None = None


def get_scorer() -> Scorer:
    global _scorer
    if _scorer is None:
        _scorer = Scorer()
    return _scorer


def reset_scorer() -> None:
    """Fuerza recrear el Scorer en la próxima llamada (tras un deploy)."""
    global _scorer
    _scorer = None
