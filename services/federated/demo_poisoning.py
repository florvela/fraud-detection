"""Demo de ataque de envenenamiento (label-flipping) y su mitigación — SAIF clase 7.

Simula el aprendizaje federado **in-process** (sin Flower ni Docker, como el
notebook de la clase de seguridad) para mostrar, en segundos y lado a lado:

1. **FedAvg limpio** — dos bancos honestos. Línea base del PR-AUC.
2. **FedAvg + cliente malicioso** — un tercer banco voltea sus etiquetas
   (*label flipping*). Con FedAvg el promedio arrastra el modelo global y el
   desempeño cae.
3. **Agregación robusta + clipping + privacidad diferencial** — las mismas tres
   partes, con el mismo cliente malicioso, pero el servidor se defiende
   (mediana/trimmed + clip L2 + ruido gaussiano). El PR-AUC se recupera.

Es el "antes y después" para la presentación: demuestra el riesgo del federado y
que las defensas de la capa 5 de SAIF funcionan.

Uso (dentro de la imagen federada, que tiene torch):
    docker compose run --rm fed-server python demo_poisoning.py
    # o local con el venv del servicio:
    python services/federated/demo_poisoning.py --rounds 5 --sample 20000
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from model import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model,
    get_weights,
    load_preprocessor,
    set_weights,
)
import numpy as np
from robust import robust_aggregate
import typer

app = typer.Typer(add_completion=False, help="Demo de envenenamiento federado y su mitigación.")

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent
DEFAULT_TRAIN = _REPO_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_TEST = _REPO_ROOT / "data" / "processed" / "test.parquet"
_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _prep(df, pre):
    X = pre.transform(df[_FEATURES])
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y


def _train_local(global_w, X, y, epochs: int, lr: float, n_features: int):
    """Entrena un MLP local partiendo de los pesos globales. Devuelve pesos nuevos."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = build_model(n_features)
    set_weights(model, global_w)
    model.train()
    pos_weight = torch.tensor([ (y == 0).sum() / max((y == 1).sum(), 1) ], dtype=torch.float32)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)),
        batch_size=256, shuffle=True,
    )
    for _ in range(epochs):
        for xb, yb in loader:
            optim.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optim.step()
    return get_weights(model)


def _evaluate(global_w, Xte, yte, n_features: int) -> dict:
    """Evalúa los pesos globales sobre el test. Devuelve PR-AUC / ROC-AUC / recall."""
    from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
    import torch

    model = build_model(n_features)
    set_weights(model, global_w)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(Xte))).numpy().ravel()
    pred = (probs >= 0.5).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(yte, probs)), 4),
        "roc_auc": round(float(roc_auc_score(yte, probs)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
    }


def _run_federated(clients, Xte, yte, n_features, *, rounds, epochs, lr, defense: dict) -> dict:
    """Corre `rounds` de federado sobre `clients` = [(X, y), ...]. `defense` → robust_aggregate."""
    rng = np.random.default_rng(0)
    global_w = get_weights(build_model(n_features))
    for r in range(rounds):
        locals_w = [_train_local(global_w, X, y, epochs, lr, n_features) for X, y in clients]
        n_examples = [len(y) for _, y in clients]
        global_w = robust_aggregate(
            locals_w, global_w, n_examples, rng=rng, **defense,
        )
        met = _evaluate(global_w, Xte, yte, n_features)
        logger.info("  round {}/{} | PR-AUC={} ROC-AUC={}", r + 1, rounds, met["pr_auc"], met["roc_auc"])
    return _evaluate(global_w, Xte, yte, n_features)


@app.command()
def run(
    train_path: Path = typer.Option(DEFAULT_TRAIN, help="Parquet de train (se parte en silos)."),
    test_path: Path = typer.Option(DEFAULT_TEST, help="Parquet de test (evaluación)."),
    rounds: int = typer.Option(5, help="Rounds federados por escenario."),
    epochs: int = typer.Option(2, help="Épocas locales por round."),
    lr: float = typer.Option(1e-3, help="Learning rate."),
    sample: int = typer.Option(20000, help="Submuestra del train para que la demo sea rápida (0 = todo)."),
    poison_ratio: float = typer.Option(1.0, help="Fracción de etiquetas que voltea el cliente malicioso."),
    clip_norm: float = typer.Option(3.0, help="Clipping L2 de la defensa."),
    dp_sigma: float = typer.Option(0.01, help="Desvío del ruido gaussiano (privacidad diferencial)."),
    agg: str = typer.Option("median", help="Agregación robusta de la defensa: median | trimmed."),
) -> None:
    """Corre los 3 escenarios y los compara en una tabla."""
    import pandas as pd

    pre = load_preprocessor()
    n_features = pre.n_features

    df = pd.read_parquet(train_path)
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)
    dte = pd.read_parquet(test_path)

    # Dos silos honestos por mediana de longitud (no-IID, como data_partition.py).
    umbral = float(df["long"].median())
    banco_a = df[df["long"] <= umbral]
    banco_b = df[df["long"] > umbral]
    Xa, ya = _prep(banco_a, pre)
    Xb, yb = _prep(banco_b, pre)
    Xte, yte = _prep(dte, pre)

    # Cliente malicioso: copia del banco A con etiquetas volteadas (label flipping).
    ym = ya.copy()
    k = int(poison_ratio * len(ym))
    idx = np.random.default_rng(0).choice(len(ym), size=k, replace=False)
    ym[idx] = 1.0 - ym[idx]

    honest = [(Xa, ya), (Xb, yb)]
    attacked = [(Xa, ya), (Xb, yb), (Xa, ym)]  # + banco malicioso

    no_defense = {"method": "fedavg", "clip_norm": 0.0, "dp_sigma": 0.0}
    defense = {"method": agg, "clip_norm": clip_norm, "dp_sigma": dp_sigma, "trim_ratio": 0.2}

    logger.info("Escenario 1/3 — FedAvg LIMPIO (2 bancos honestos)")
    r_clean = _run_federated(honest, Xte, yte, n_features, rounds=rounds, epochs=epochs, lr=lr, defense=no_defense)

    logger.info("Escenario 2/3 — FedAvg + CLIENTE MALICIOSO (label-flip, sin defensa)")
    r_attack = _run_federated(attacked, Xte, yte, n_features, rounds=rounds, epochs=epochs, lr=lr, defense=no_defense)

    logger.info("Escenario 3/3 — DEFENSA ({}+clip+DP) + CLIENTE MALICIOSO", agg)
    r_defended = _run_federated(attacked, Xte, yte, n_features, rounds=rounds, epochs=epochs, lr=lr, defense=defense)

    print("\n" + "=" * 68)
    print("  ATAQUE DE ENVENENAMIENTO FEDERADO Y MITIGACIÓN (SAIF capa 5)")
    print("=" * 68)
    fmt = "  {:<42} {:>8} {:>8}"
    print(fmt.format("Escenario", "PR-AUC", "recall"))
    print("  " + "-" * 60)
    print(fmt.format("1) FedAvg limpio (honestos)", r_clean["pr_auc"], r_clean["recall"]))
    print(fmt.format("2) FedAvg + malicioso (SIN defensa)", r_attack["pr_auc"], r_attack["recall"]))
    print(fmt.format(f"3) {agg}+clip+DP + malicioso", r_defended["pr_auc"], r_defended["recall"]))
    print("=" * 68)
    limpio = r_clean["pr_auc"]
    atacado = r_attack["pr_auc"]
    defendido = r_defended["pr_auc"]
    caida = limpio - atacado          # cuánto bajó el ataque (positivo = bajó)
    recup = defendido - atacado       # cuánto recuperó la defensa (positivo = subió)
    pct_caida = (caida / limpio * 100) if limpio else 0.0
    pct_recup = (recup / caida * 100) if caida > 0 else 0.0
    print(f"  ATAQUE  : PR-AUC {limpio:.4f} -> {atacado:.4f}  "
          f"(baja {caida:.4f}, -{pct_caida:.0f}%)")
    print(f"  DEFENSA : PR-AUC {atacado:.4f} -> {defendido:.4f}  "
          f"(sube {recup:.4f}, recupera {pct_recup:.0f}% de lo perdido)")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    app()
