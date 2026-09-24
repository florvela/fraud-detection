"""Servidor Flower con estrategia FedAvg para el MLP tabular de fraude.

Coordina N rounds de aprendizaje federado entre los bancos (clientes). En cada
round: envía los pesos globales, recibe los pesos locales entrenados y los
promedia (FedAvg, ponderado por número de ejemplos). Al terminar, guarda los
pesos del **modelo global** en ``models/mlp_federado.npz`` para que
``evaluate_vs_central.py`` los use.

Los datos de los bancos nunca llegan al servidor: solo viajan vectores de pesos.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from model import build_model, get_weights, load_preprocessor

app = typer.Typer(add_completion=False, help="Servidor federado FedAvg.")

_SERVICE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_OUT = _SERVICE_DIR / "models" / "mlp_federado.npz"


def _pesos_iniciales() -> list[np.ndarray]:
    """Inicializa el MLP global para arrancar FedAvg desde pesos comunes."""
    pre = load_preprocessor()
    model = build_model(pre.n_features)
    return get_weights(model)


def _guardar_pesos(weights: list[np.ndarray], out_path: Path) -> None:
    """Persiste la lista de ndarrays del modelo global en un .npz."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, *weights)
    logger.success("Modelo global federado guardado en {}", out_path)


@app.command()
def run(
    rounds: int = typer.Option(
        None, help="Cantidad de rounds federados. Por defecto env FED_ROUNDS o 5.",
    ),
    server_address: str = typer.Option(
        None, help="host:port de escucha. Por defecto env FED_SERVER o 0.0.0.0:8080.",
    ),
    min_clients: int = typer.Option(2, help="Mínimo de clientes (bancos) por round."),
    model_out: Path = typer.Option(DEFAULT_MODEL_OUT, help="Ruta de salida del modelo global."),
) -> None:
    """Arranca el servidor FedAvg y guarda el modelo global al finalizar."""
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

    n_rounds = rounds if rounds is not None else int(os.getenv("FED_ROUNDS", "5"))
    address = server_address or os.getenv("FED_SERVER", "0.0.0.0:8080")

    initial_parameters = ndarrays_to_parameters(_pesos_iniciales())

    def _promedio_metricas(metrics):
        """Agrega métricas de evaluación ponderando por nº de ejemplos."""
        total = sum(n for n, _ in metrics) or 1
        pr = sum(n * m.get("pr_auc", 0.0) for n, m in metrics) / total
        roc = sum(n * m.get("roc_auc", 0.0) for n, m in metrics) / total
        return {"pr_auc": pr, "roc_auc": roc}

    class SaveFedAvg(fl.server.strategy.FedAvg):
        """FedAvg que recuerda los últimos pesos globales agregados.

        FedAvg no expone los pesos finales; sobreescribimos ``aggregate_fit``
        para capturarlos round a round y así poder guardarlos al terminar.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.latest_parameters = kwargs.get("initial_parameters")

        def aggregate_fit(self, server_round, results, failures):
            params, metrics = super().aggregate_fit(server_round, results, failures)
            if params is not None:
                self.latest_parameters = params
                logger.info("Round {} agregado (FedAvg).", server_round)
            return params, metrics

    strategy = SaveFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=_promedio_metricas,
    )

    logger.info("Servidor FedAvg en {} | rounds={} | min_clients={}", address, n_rounds, min_clients)
    fl.server.start_server(
        server_address=address,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )

    # Guardar los pesos globales finales capturados por la estrategia.
    if strategy.latest_parameters is not None:
        _guardar_pesos(parameters_to_ndarrays(strategy.latest_parameters), model_out)
    else:
        logger.warning("No hay pesos globales para guardar (¿ningún round completó?).")


if __name__ == "__main__":
    app()
