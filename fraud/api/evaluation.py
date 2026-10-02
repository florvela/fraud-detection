"""Evaluador challenger vs champion contra la ground truth (UC5).

Totalmente offline: toma las transacciones que **ya tienen label real** (revisión
del analista + denuncias post-mortem) y compara las predicciones que dejaron
registradas el champion y el challenger. No vuelve a inferir: usa lo que se guardó
en el store en el momento del scoring (sombra incluida).

Devuelve métricas por modelo y un veredicto: *¿el challenger mejora al champion?*.
Si hay MLflow configurado, loguea las métricas para dejar trazabilidad de la
decisión de deploy (best-effort).
"""

from __future__ import annotations

import os

from loguru import logger
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from fraud.api.store import TransactionStore, get_store


def _metrics_for(rows: list[dict], prob_key: str, pred_key: str) -> dict | None:
    """Métricas de un modelo sobre las filas que tienen su predicción."""
    usable = [r for r in rows if r.get(prob_key) is not None]
    if not usable:
        return None

    y_true = [int(r["label"]) for r in usable]
    y_pred = [int(r[pred_key]) for r in usable]
    y_score = [float(r[prob_key]) for r in usable]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0, labels=[0, 1]
    )

    # PR-AUC solo tiene sentido si hay ambas clases en la ground truth.
    pr_auc = None
    if 0 < sum(y_true) < len(y_true):
        pr_auc = float(average_precision_score(y_true, y_score))

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    fraud_total = sum(float(r["amt"] or 0.0) for r in usable if int(r["label"]) == 1)
    fraud_captured = sum(
        float(r["amt"] or 0.0)
        for r in usable
        if int(r["label"]) == 1 and int(r[pred_key]) == 1
    )

    return {
        "n": len(usable),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "pr_auc": None if pr_auc is None else round(pr_auc, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positives": fp,
        "fraud_amount_total": round(fraud_total, 2),
        "fraud_amount_captured": round(fraud_captured, 2),
    }


def _verdict(champion: dict | None, challenger: dict | None) -> dict:
    if challenger is None:
        return {"challenger_better": False, "reason": "no hay predicciones del challenger todavía"}
    if champion is None:
        return {"challenger_better": True, "reason": "no hay predicciones del champion"}

    # Criterio principal: recall (en fraude importa no dejar pasar). Desempate: PR-AUC.
    if challenger["recall"] != champion["recall"]:
        better = challenger["recall"] > champion["recall"]
        return {
            "challenger_better": better,
            "reason": f"recall challenger {challenger['recall']} vs champion {champion['recall']}",
        }
    c_auc = challenger["pr_auc"] or 0.0
    champ_auc = champion["pr_auc"] or 0.0
    better = c_auc >= champ_auc
    return {
        "challenger_better": better,
        "reason": f"desempate por PR-AUC: challenger {c_auc} vs champion {champ_auc}",
    }


def evaluate(store: TransactionStore | None = None, log_to_mlflow: bool = True) -> dict:
    """Compara challenger vs champion sobre las tx con ground truth."""
    store = store or get_store()
    rows = store.rows_with_ground_truth()

    champion = _metrics_for(rows, "champion_prob", "champion_is_fraud")
    challenger = _metrics_for(rows, "challenger_prob", "challenger_is_fraud")
    verdict = _verdict(champion, challenger)

    result = {
        "n_with_ground_truth": len(rows),
        "champion": champion,
        "challenger": challenger,
        **verdict,
    }

    if log_to_mlflow and os.getenv("MLFLOW_TRACKING_URI"):
        _log_to_mlflow(result)

    return result


def _log_to_mlflow(result: dict) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "fraud-detection"))
        with mlflow.start_run(run_name="challenger-vs-champion"):
            mlflow.log_metric("n_with_ground_truth", result["n_with_ground_truth"])
            mlflow.log_param("challenger_better", result["challenger_better"])
            for model in ("champion", "challenger"):
                m = result.get(model)
                if not m:
                    continue
                for key in ("precision", "recall", "f1", "pr_auc", "fraud_amount_captured"):
                    if m.get(key) is not None:
                        mlflow.log_metric(f"{model}_{key}", float(m[key]))
    except Exception as exc:  # noqa: BLE001 - logging es best-effort
        logger.warning(f"No se pudo loguear la evaluación a MLflow: {exc}")
