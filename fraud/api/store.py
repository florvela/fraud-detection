"""Store de transacciones y labels (ground truth) del sistema de fraude.

Materializa dos piezas del diagrama de arquitectura:

- **BBDD transacciones** (`transactions`): cada scoring deja la transacción cruda
  (features), el score del **champion** (el que decide) y, si existe, el score del
  **challenger** (en sombra). El estado es `OK` (aprobada) o `PENDING` (retenida a
  revisión), con lo cual la propia tabla funciona como **cola de revisión**.
- **Store etiquetado / ground truth** (`labels`): el label real por `transaction_id`,
  que llega después por revisión manual (analista) o por denuncia post-mortem
  (contracargo). Es el insumo del reentrenamiento y del evaluador challenger-vs-champion.

Implementación con SQLite (stdlib): sin dependencias nuevas, persistente e
inspeccionable. La ruta se toma de la env `FRAUD_DB` (default `data/fraud.db`);
`:memory:` sirve para tests. Pensado para el serving in-process (un solo proceso),
con un lock para tolerar los workers de FastAPI/uvicorn.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import os
import sqlite3
import threading
import uuid

from fraud.config import DATA_DIR

# Estados de una transacción
STATUS_OK = "OK"            # aprobada automáticamente (score del champion ≤ umbral)
STATUS_PENDING = "PENDING"  # retenida: va a la cola de revisión del analista
STATUS_APPROVED = "APPROVED"  # el analista la resolvió como legítima
STATUS_REJECTED = "REJECTED"  # el analista la resolvió como fraude

# Origen del label (ground truth)
SOURCE_REVIEW = "review"          # decisión del analista revisor (UC2)
SOURCE_POST_MORTEM = "post_mortem"  # denuncia tardía / contracargo (UC3)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _db_path() -> str:
    raw = os.getenv("FRAUD_DB")
    if raw:
        return raw
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR / "fraud.db")


class TransactionStore:
    """Persistencia de transacciones + labels. Seguro para uso concurrente básico."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or _db_path()
        self._lock = threading.Lock()
        # check_same_thread=False: lo protegemos nosotros con el lock.
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id      TEXT PRIMARY KEY,
                    ts                  TEXT NOT NULL,
                    features            TEXT NOT NULL,
                    amt                 REAL,
                    champion_version    TEXT,
                    champion_prob       REAL,
                    champion_is_fraud   INTEGER,
                    challenger_version  TEXT,
                    challenger_prob     REAL,
                    challenger_is_fraud INTEGER,
                    decision            TEXT,
                    status              TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS labels (
                    transaction_id TEXT PRIMARY KEY,
                    label          INTEGER NOT NULL,
                    source         TEXT NOT NULL,
                    ts             TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # ---------------------------------------------------------------- scoring
    def record_scoring(
        self,
        *,
        features: dict,
        champion: dict,
        challenger: dict | None,
        decision: str,
        status: str,
        transaction_id: str | None = None,
    ) -> str:
        """Guarda una transacción puntuada. Devuelve el transaction_id."""
        tx_id = transaction_id or f"tx-{uuid.uuid4().hex[:12]}"
        row = (
            tx_id,
            _now(),
            json.dumps(features),
            float(features.get("amt", 0.0) or 0.0),
            champion.get("model_version"),
            _as_float(champion.get("probability")),
            _as_int(champion.get("is_fraud")),
            (challenger or {}).get("model_version"),
            _as_float((challenger or {}).get("probability")),
            _as_int((challenger or {}).get("is_fraud")),
            decision,
            status,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO transactions (
                    transaction_id, ts, features, amt,
                    champion_version, champion_prob, champion_is_fraud,
                    challenger_version, challenger_prob, challenger_is_fraud,
                    decision, status
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
            self._conn.commit()
        return tx_id

    # ------------------------------------------------------------- consultas
    def get(self, transaction_id: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM transactions WHERE transaction_id = ?", (transaction_id,)
            )
            tx = cur.fetchone()
            lbl = self._conn.execute(
                "SELECT label, source, ts FROM labels WHERE transaction_id = ?",
                (transaction_id,),
            ).fetchone()
        if tx is None:
            return None
        return _tx_to_dict(tx, lbl)

    def list_by_status(self, status: str = STATUS_PENDING, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM transactions WHERE status = ? ORDER BY ts DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        return [_tx_to_dict(r, None) for r in rows]

    def exists(self, transaction_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "SELECT 1 FROM transactions WHERE transaction_id = ?", (transaction_id,)
            )
            return cur.fetchone() is not None

    # --------------------------------------------------------------- labels
    def resolve(self, transaction_id: str, decision: str) -> bool:
        """Resuelve una tx retenida (UC2): decision in {approve, reject}.

        approve -> legítima (label 0), reject -> fraude (label 1).
        Devuelve False si la transacción no existe.
        """
        if not self.exists(transaction_id):
            return False
        if decision == "approve":
            status, label = STATUS_APPROVED, 0
        elif decision == "reject":
            status, label = STATUS_REJECTED, 1
        else:
            raise ValueError(f"decisión inválida: {decision!r} (esperado approve|reject)")
        with self._lock:
            self._conn.execute(
                "UPDATE transactions SET status = ?, decision = ? WHERE transaction_id = ?",
                (status, decision, transaction_id),
            )
            self._upsert_label(transaction_id, label, SOURCE_REVIEW)
            self._conn.commit()
        return True

    def add_label(self, transaction_id: str, label: int, source: str) -> None:
        """Pega un label por transaction_id (ground truth). Usado por el post-mortem."""
        with self._lock:
            self._upsert_label(transaction_id, int(label), source)
            self._conn.commit()

    def _upsert_label(self, transaction_id: str, label: int, source: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO labels (transaction_id, label, source, ts) VALUES (?,?,?,?)",
            (transaction_id, int(label), source, _now()),
        )

    # ------------------------------------------------------------ evaluación
    def rows_with_ground_truth(self) -> list[dict]:
        """Transacciones que ya tienen label real y score de ambos modelos.

        Es el insumo del evaluador challenger-vs-champion (UC5): solo tiene sentido
        comparar sobre las tx para las que ya llegó la ground truth.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT t.transaction_id, t.amt,
                       t.champion_prob, t.champion_is_fraud,
                       t.challenger_prob, t.challenger_is_fraud,
                       l.label
                FROM transactions t
                JOIN labels l ON l.transaction_id = t.transaction_id
                WHERE t.champion_prob IS NOT NULL
                ORDER BY t.ts
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def counts(self) -> dict:
        with self._lock:
            n_tx = self._conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
            n_pending = self._conn.execute(
                "SELECT COUNT(*) FROM transactions WHERE status = ?", (STATUS_PENDING,)
            ).fetchone()[0]
            n_labels = self._conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        return {"transactions": n_tx, "pending": n_pending, "labels": n_labels}


def _as_float(v) -> float | None:
    return None if v is None else float(v)


def _as_int(v) -> int | None:
    return None if v is None else int(bool(v))


def _tx_to_dict(tx: sqlite3.Row, lbl: sqlite3.Row | None) -> dict:
    d = {
        "transaction_id": tx["transaction_id"],
        "ts": tx["ts"],
        "features": json.loads(tx["features"]),
        "amt": tx["amt"],
        "champion": {
            "model_version": tx["champion_version"],
            "probability": tx["champion_prob"],
            "is_fraud": None if tx["champion_is_fraud"] is None else bool(tx["champion_is_fraud"]),
        },
        "decision": tx["decision"],
        "status": tx["status"],
    }
    if tx["challenger_version"] is not None or tx["challenger_prob"] is not None:
        d["challenger"] = {
            "model_version": tx["challenger_version"],
            "probability": tx["challenger_prob"],
            "is_fraud": None
            if tx["challenger_is_fraud"] is None
            else bool(tx["challenger_is_fraud"]),
        }
    else:
        d["challenger"] = None
    if lbl is not None:
        d["label"] = {"label": lbl["label"], "source": lbl["source"], "ts": lbl["ts"]}
    return d


# Instancia compartida del serving (lazy): un solo store por proceso.
_store: TransactionStore | None = None


def get_store() -> TransactionStore:
    global _store
    if _store is None:
        _store = TransactionStore()
    return _store
