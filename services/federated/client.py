"""Cliente Flower (NumPyClient) para un banco: entrena el MLP local sobre su silo.

Cada banco levanta este cliente apuntando a SU parquet (por env ``FED_SILO`` o
argumento). El cliente:
* Carga y preprocesa su silo (nunca lo comparte).
* Recibe los pesos globales, entrena localmente unas épocas y los devuelve.
* Reporta métricas de evaluación local.

Rebalanceo del desbalance (~1% fraude)
--------------------------------------
Se aplica SOLO en entrenamiento vía ``pos_weight`` de ``BCEWithLogitsLoss``
(equivalente a class weight): ``pos_weight = n_neg / n_pos`` del silo. La
evaluación NO se rebalancea para reflejar la distribución real.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from loguru import logger

from model import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    build_model,
    get_weights,
    load_preprocessor,
    set_weights,
)

app = typer.Typer(add_completion=False, help="Cliente federado (un banco).")

_SERVICE_DIR = Path(__file__).resolve().parent
_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _cargar_silo(silo_path: Path):
    """Lee el parquet del banco y devuelve (X preprocesado, y)."""
    df = pd.read_parquet(silo_path)
    pre = load_preprocessor()
    X = pre.transform(df[_FEATURES])
    y = df[TARGET].to_numpy(dtype=np.float32)
    return X, y, pre.n_features


def _make_client(silo_path: Path, local_epochs: int, lr: float, batch_size: int):
    """Construye la instancia de NumPyClient (import de torch/flwr diferido)."""
    import flwr as fl
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    X, y, n_features = _cargar_silo(silo_path)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)
    logger.info(
        "Silo {} | n={} | fraude={:.3%} | pos_weight={:.1f}",
        silo_path.name, len(y), y.mean(), float(pos_weight),
    )

    device = torch.device("cpu")
    model = build_model(n_features).to(device)

    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    def _entrenar() -> None:
        """Entrena ``local_epochs`` con pos_weight para el desbalance."""
        model.train()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optim.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optim.step()

    def _evaluar():
        """Evalúa sobre el silo local (sin rebalanceo). Devuelve (loss, métricas)."""
        from sklearn.metrics import average_precision_score, roc_auc_score

        model.eval()
        criterion = torch.nn.BCEWithLogitsLoss()  # sin pos_weight en eval
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            loss = float(criterion(logits, torch.from_numpy(y).unsqueeze(1).to(device)))
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        # PR-AUC/ROC-AUC solo si hay ambas clases en el silo.
        pr_auc = float(average_precision_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
        roc_auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.0
        return loss, {"pr_auc": pr_auc, "roc_auc": roc_auc}

    class BancoClient(fl.client.NumPyClient):
        """NumPyClient que entrena el MLP sobre el silo de un banco."""

        def get_parameters(self, config):  # noqa: D401
            return get_weights(model)

        def set_parameters(self, parameters):
            set_weights(model, parameters)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            _entrenar()
            return get_weights(model), len(y), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, metrics = _evaluar()
            logger.info("Eval local {} | loss={:.4f} | {}", silo_path.name, loss, metrics)
            return loss, len(y), metrics

    return BancoClient()


@app.command()
def run(
    silo: Path = typer.Option(
        None, help="Parquet del silo del banco. Por defecto usa env FED_SILO.",
    ),
    server_address: str = typer.Option(
        None, help="host:port del servidor Flower. Por defecto env FED_SERVER o 127.0.0.1:8080.",
    ),
    local_epochs: int = typer.Option(3, help="Épocas de entrenamiento local por round."),
    lr: float = typer.Option(1e-3, help="Learning rate (Adam)."),
    batch_size: int = typer.Option(256, help="Tamaño de batch local."),
) -> None:
    """Levanta el cliente y lo conecta al servidor Flower."""
    import flwr as fl

    silo_path = silo or Path(os.getenv("FED_SILO", _SERVICE_DIR / "data" / "banco_a.parquet"))
    address = server_address or os.getenv("FED_SERVER", "127.0.0.1:8080")

    logger.info("Cliente federado -> servidor {} | silo {}", address, silo_path)
    client = _make_client(silo_path, local_epochs=local_epochs, lr=lr, batch_size=batch_size)
    fl.client.start_client(server_address=address, client=client.to_client())


if __name__ == "__main__":
    app()
