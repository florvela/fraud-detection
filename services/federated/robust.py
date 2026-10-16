"""Defensas del aprendizaje federado contra clientes maliciosos (clases 5 y 7).

En FedAvg el servidor promedia los pesos que mandan los bancos. Un cliente
**malicioso** (p. ej. que entrena con etiquetas volteadas — *label flipping*) puede
arrastrar el modelo global y degradar su desempeño round a round. La clase de
seguridad (SAIF, capa 5 "entrenamiento") propone tres mitigaciones, que acá se
implementan en numpy puro para poder testearlas y reusarlas tanto en el servidor
Flower real (`server.py`) como en la demo in-process (`demo_poisoning.py`):

1. **Clipping L2 de la actualización** de cada cliente: se acota cuánto puede mover
   un cliente al modelo global. Limita el daño que un único cliente puede hacer.
   Protege la **integridad**.
2. **Agregación robusta**: en vez del promedio (sensible a outliers), **mediana** o
   **media recortada** (*trimmed mean*) coordenada a coordenada. Un cliente
   envenenado queda como outlier y pesa poco o nada.
3. **Privacidad diferencial**: **ruido gaussiano** sobre los pesos agregados. Evita
   que de los pesos se pueda inferir información de los datos de un banco. Protege la
   **confidencialidad**.

Representación: los "pesos" de un modelo son una ``list[np.ndarray]`` (una entrada
por capa/tensor), el formato que usa Flower (`get_weights`/`set_weights`).
"""

from __future__ import annotations

import numpy as np


def _global_l2(delta: list[np.ndarray]) -> float:
    """Norma L2 de una actualización completa (todos los tensores aplanados juntos)."""
    return float(np.sqrt(sum(float(np.sum(d.astype(np.float64) ** 2)) for d in delta)))


def clip_update(
    client_weights: list[np.ndarray],
    global_weights: list[np.ndarray],
    clip_norm: float,
) -> list[np.ndarray]:
    """Recorta la actualización ``cliente - global`` a norma L2 <= ``clip_norm``.

    Devuelve los pesos del cliente ya recortados (``global + delta_recortada``). Si
    ``clip_norm`` <= 0, no recorta (devuelve los pesos tal cual).
    """
    if clip_norm <= 0:
        return client_weights
    delta = [c - g for c, g in zip(client_weights, global_weights)]
    norm = _global_l2(delta)
    if norm <= clip_norm or norm == 0.0:
        return client_weights
    factor = clip_norm / norm
    return [g + d * factor for g, d in zip(global_weights, delta)]


def aggregate(
    client_weights: list[list[np.ndarray]],
    num_examples: list[int] | None = None,
    method: str = "fedavg",
    trim_ratio: float = 0.1,
) -> list[np.ndarray]:
    """Agrega los pesos de N clientes con la estrategia elegida.

    - ``fedavg``  : promedio ponderado por nº de ejemplos (FedAvg clásico).
    - ``mean``    : promedio simple.
    - ``median``  : mediana coordenada a coordenada (robusta a outliers).
    - ``trimmed`` : media recortada (descarta ``trim_ratio`` de cada extremo).
    """
    n_clients = len(client_weights)
    if n_clients == 0:
        raise ValueError("No hay clientes para agregar.")
    n_layers = len(client_weights[0])

    if method == "fedavg":
        weights = np.asarray(num_examples, dtype=np.float64) if num_examples else np.ones(n_clients)
        weights = weights / weights.sum()
        return [
            sum(weights[i] * client_weights[i][layer] for i in range(n_clients))
            for layer in range(n_layers)
        ]

    out: list[np.ndarray] = []
    for layer in range(n_layers):
        stacked = np.stack([client_weights[i][layer] for i in range(n_clients)], axis=0)
        if method == "mean":
            agg = stacked.mean(axis=0)
        elif method == "median":
            agg = np.median(stacked, axis=0)
        elif method == "trimmed":
            agg = _trimmed_mean(stacked, trim_ratio)
        else:
            raise ValueError(f"method desconocido: {method!r}")
        out.append(agg.astype(client_weights[0][layer].dtype))
    return out


def _trimmed_mean(stacked: np.ndarray, trim_ratio: float) -> np.ndarray:
    """Media recortada por coordenada: ordena sobre el eje de clientes y descarta
    ``trim_ratio`` de cada extremo antes de promediar."""
    n = stacked.shape[0]
    k = int(np.floor(trim_ratio * n))
    if 2 * k >= n:  # recorte demasiado agresivo -> caemos a la mediana
        return np.median(stacked, axis=0)
    ordered = np.sort(stacked, axis=0)
    return ordered[k : n - k].mean(axis=0)


def add_gaussian_noise(
    weights: list[np.ndarray],
    sigma: float,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Suma ruido gaussiano N(0, sigma) a cada tensor (privacidad diferencial).

    ``sigma`` es el desvío del ruido ("noise multiplier" del notebook de clase). Si
    es <= 0, no agrega ruido. Un sigma alto protege más pero degrada el modelo.
    """
    if sigma <= 0:
        return weights
    gen = rng or np.random.default_rng()
    return [w + gen.normal(0.0, sigma, size=w.shape).astype(w.dtype) for w in weights]


def robust_aggregate(
    client_weights: list[list[np.ndarray]],
    global_weights: list[np.ndarray],
    num_examples: list[int] | None = None,
    *,
    method: str = "median",
    clip_norm: float = 0.0,
    trim_ratio: float = 0.1,
    dp_sigma: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Pipeline de defensa completo: clipping -> agregación robusta -> ruido DP.

    Es lo que usa el servidor cuando se activan las defensas. Con ``method='fedavg'``,
    ``clip_norm=0`` y ``dp_sigma=0`` equivale al FedAvg clásico (sin defensa).
    """
    if clip_norm > 0:
        client_weights = [clip_update(cw, global_weights, clip_norm) for cw in client_weights]
    aggregated = aggregate(client_weights, num_examples, method=method, trim_ratio=trim_ratio)
    if dp_sigma > 0:
        aggregated = add_gaussian_noise(aggregated, dp_sigma, rng=rng)
    return aggregated
